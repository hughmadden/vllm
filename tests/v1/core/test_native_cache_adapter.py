# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU command/ACK ownership tests; the fake endpoint allocates no KV pages.

Purpose: keep scheduler intent separate from acknowledged native cache state.
I/O: schema-1 envelopes and snapshots, with no process-local native pointer.
Failures: lost/stale ACK, publication before acceptance, canceled readers and
restart credit leaks. A scripted CPU endpoint is the cheapest sufficient gate.
"""

import copy
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# Keep this seam test independent of the GPU runtime and root model fixtures.
SPEC = importlib.util.spec_from_file_location(
    "native_cache_adapter",
    Path(__file__).parents[3] / "vllm/v1/core/native_cache_adapter.py",
)
assert SPEC and SPEC.loader
native = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = native
SPEC.loader.exec_module(native)


def handle(slot=0, generation=1, owner=17):
    return {"owner": owner, "slot": slot, "generation": generation}


def request(slot=0, request_id=1, end=0, prepared=None, readers=0, closing=False):
    return {
        "request": handle(slot),
        "request_id": request_id,
        "committed_end": end,
        "prepared_end": end if prepared is None else prepared,
        "readers": readers,
        "closing": closing,
    }


class Endpoint:
    """Script native outcomes/credits; exact retry never reapplies a mutation."""

    def __init__(self, owner=17):
        self.snapshot = {
            "owner": owner,
            "epoch": 0,
            "source_page_capacity": [4, 4, 4, 4],
            "source_pages_free": [4, 4, 4, 4],
            "cache_bytes": 123456,
            "transactions": 0,
            "source_prefixes": 0,
            "source_prefix_capacity": 2,
            "requests": [],
        }
        self.last = None
        self.applied = 0

    def reply(self, envelope, result=None, **updates):
        if self.last and envelope == self.last[0]:
            return copy.deepcopy(self.last[1])
        assert result is not None
        assert envelope["owner"] == self.snapshot["owner"]
        assert envelope["epoch"] == self.snapshot["epoch"]
        self.snapshot.update(copy.deepcopy(updates))
        self.snapshot["epoch"] += 1
        ack = {
            "command_id": envelope["command_id"],
            "epoch": self.snapshot["epoch"],
            "result": result,
            "snapshot": copy.deepcopy(self.snapshot),
        }
        self.last = (copy.deepcopy(envelope), copy.deepcopy(ack))
        self.applied += 1
        return ack


def pair():
    endpoint = Endpoint()
    return native.NativeCacheAdapter(endpoint.snapshot), endpoint


def admit(adapter, endpoint, name="a", slot=0):
    command = adapter.admit(name, slot)
    current = endpoint.snapshot["requests"] + [
        request(slot, command["command"]["request_id"])
    ]
    adapter.acknowledge(
        endpoint.reply(
            command, {"state": "admitted", "request": handle(slot)}, requests=current
        )
    )


def begin(adapter, endpoint, name="a", lane=0, rows=8):
    command = adapter.begin_batch(name, lane, rows)
    transaction = {
        "owner": 17,
        "id": endpoint.applied + 1,
        "lane": lane,
        "request": command["command"]["request"],
    }
    states = copy.deepcopy(endpoint.snapshot["requests"])
    for state in states:
        if state["request"] == transaction["request"]:
            state["prepared_end"] = state["committed_end"] + rows
    adapter.acknowledge(
        endpoint.reply(
            command,
            {"state": "begun", "transaction": transaction},
            requests=states,
            transactions=endpoint.snapshot["transactions"] + 1,
        )
    )
    return transaction


def test_backend_refuses_serving_and_never_constructs_a_physical_allocator():
    adapter, _ = pair()
    with pytest.raises(native.IntegrationDisabled):
        adapter.require_serving_ready()
    with pytest.raises(native.IntegrationDisabled):
        adapter.get_computed_blocks("a")
    assert adapter.ledger["cache_bytes"] == 123456
    assert not hasattr(adapter, "block_pool")


def test_lost_admit_ack_retries_exact_command_without_double_admission():
    adapter, endpoint = pair()
    command = adapter.admit("a", 0)
    ack = endpoint.reply(
        command, {"state": "admitted", "request": handle()}, requests=[request()]
    )
    assert adapter.ledger["requests"] == []
    with pytest.raises(native.CommandPending):
        adapter.admit("b", 1)
    command["command"]["slot"] = 15
    retried = adapter.retry()
    assert retried["command"]["slot"] == 0
    adapter.acknowledge(endpoint.reply(retried))
    assert endpoint.applied == 1
    assert adapter.source_committed_tokens("a") == 0
    assert adapter.acknowledge(ack) is False


def test_only_accepted_publication_ack_advances_visible_extent():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    begin(adapter, endpoint, rows=8)
    command = adapter.reserve_publish("a", 3)
    adapter.acknowledge(
        endpoint.reply(command, {"state": "reserved"}, source_pages_free=[3, 3, 3, 3])
    )
    assert adapter.source_committed_tokens("a") == 0
    adapter.acknowledge(endpoint.reply(adapter.publish("a"), {"state": "pending"}))
    assert adapter.source_committed_tokens("a") == 0
    command = adapter.publish("a")
    ack = endpoint.reply(
        command,
        {"state": "published", "committed_end": 3},
        requests=[request(end=3)],
        transactions=0,
    )
    assert adapter.source_committed_tokens("a") == 0
    adapter.acknowledge(ack)
    assert adapter.source_committed_tokens("a") == 3
    assert adapter.ledger["source_pages_free"] == [3, 3, 3, 3]


def test_two_lanes_complete_out_of_order_without_same_request_overlap():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    admit(adapter, endpoint, "b", 1)
    begin(adapter, endpoint, "a", 0)
    begin(adapter, endpoint, "b", 1)
    with pytest.raises(native.AdapterError):
        adapter.begin_batch("a", 1, 1)
    for name in ("b", "a"):
        adapter.acknowledge(
            endpoint.reply(adapter.reserve_publish(name, 2), {"state": "reserved"})
        )
        states = copy.deepcopy(endpoint.snapshot["requests"])
        slot = 1 if name == "b" else 0
        states[slot].update(committed_end=2, prepared_end=2)
        adapter.acknowledge(
            endpoint.reply(
                adapter.publish(name),
                {"state": "published", "committed_end": 2},
                requests=states,
                transactions=endpoint.snapshot["transactions"] - 1,
            )
        )
    assert adapter.source_committed_tokens("a") == 2
    assert adapter.source_committed_tokens("b") == 2


def test_cancel_during_lost_ack_hides_use_and_waits_for_native_readers():
    adapter, endpoint = pair()
    command = adapter.admit("a", 0)
    ack = endpoint.reply(
        command,
        {"state": "admitted", "request": handle()},
        requests=[request(readers=1)],
    )
    adapter.revoke("a")
    adapter.acknowledge(ack)
    with pytest.raises(native.AdapterError):
        adapter.begin_batch("a", 0, 1)
    adapter.acknowledge(
        endpoint.reply(
            adapter.free("a"),
            {"state": "closing"},
            requests=[request(readers=1, closing=True)],
            source_pages_free=[3, 3, 3, 3],
        )
    )
    assert adapter.ledger["source_pages_free"] == [3, 3, 3, 3]
    adapter.acknowledge(
        endpoint.reply(
            adapter.drain("a"),
            {"state": "released"},
            requests=[],
            source_pages_free=[4, 4, 4, 4],
        )
    )
    assert adapter.ledger["requests"] == []


def test_rejected_reservation_does_not_invent_credits_or_consume_transaction():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    begin(adapter, endpoint)
    command = adapter.reserve_publish("a", 8)
    before = adapter.ledger
    adapter.reject(command["owner"], command["epoch"], command["command_id"])
    assert adapter.ledger == before
    assert adapter.reserve_publish("a", 2)["command"]["accepted"] == 2


def test_start_failure_requires_abort_and_never_publishes():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    begin(adapter, endpoint)
    adapter.acknowledge(
        endpoint.reply(
            adapter.reserve_publish("a", 4),
            {"state": "failed", "reason": "device failure", "draining": True},
            source_pages_free=[3, 3, 3, 3],
        )
    )
    with pytest.raises(native.AdapterError):
        adapter.publish("a")
    adapter.acknowledge(endpoint.reply(adapter.abort_batch("a"), {"state": "pending"}))
    assert adapter.ledger["source_pages_free"] == [3, 3, 3, 3]
    adapter.acknowledge(
        endpoint.reply(
            adapter.abort_batch("a"),
            {"state": "aborted"},
            requests=[request()],
            transactions=0,
            source_pages_free=[4, 4, 4, 4],
        )
    )
    assert adapter.source_committed_tokens("a") == 0


def test_worker_restart_invalidates_requests_prefixes_and_old_ack():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    old_ack = endpoint.reply(
        adapter.retain_source_prefix("a", b"x" * 32),
        {
            "state": "source_prefix",
            "prefix": {"owner": 17, "id": 7},
            "committed_end": 0,
        },
        source_prefixes=1,
    )
    adapter.acknowledge(old_ack)
    restarted = Endpoint(owner=18)
    assert adapter.restart(restarted.snapshot) == ("a",)
    assert adapter.lookup_source_prefix(b"x" * 32) is None
    with pytest.raises(native.AdapterError):
        adapter.acknowledge(old_ack)
    with pytest.raises(native.AdapterError):
        adapter.restart(restarted.snapshot)
    assert adapter.ledger["owner"] == 18


def test_source_forks_and_cow_credits_never_become_a_model_prefix_hit():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    begin(adapter, endpoint, rows=257)
    adapter.acknowledge(
        endpoint.reply(
            adapter.reserve_publish("a", 257),
            {"state": "reserved"},
            source_pages_free=[3, 3, 3, 2],
        )
    )
    adapter.acknowledge(
        endpoint.reply(
            adapter.publish("a"),
            {"state": "published", "committed_end": 257},
            requests=[request(end=257, readers=1)],
            transactions=0,
        )
    )
    adapter.acknowledge(
        endpoint.reply(
            adapter.retain_source_prefix("a", b"x" * 32),
            {
                "state": "source_prefix",
                "prefix": {"owner": 17, "id": 7},
                "committed_end": 257,
            },
            source_prefixes=1,
        )
    )
    for name, slot in (("b", 1), ("c", 2)):
        command = adapter.restore_source_prefix(b"x" * 32, name, slot)
        assert command["command"]["prefix"] == {"owner": 17, "id": 7}
        states = endpoint.snapshot["requests"] + [request(slot, slot + 1, end=257)]
        adapter.acknowledge(
            endpoint.reply(
                command, {"state": "admitted", "request": handle(slot)}, requests=states
            )
        )
    begin(adapter, endpoint, "b", 1)
    adapter.acknowledge(
        endpoint.reply(
            adapter.reserve_publish("b", 2),
            {"state": "reserved"},
            source_pages_free=[2, 2, 2, 1],
        )
    )
    assert adapter.ledger["source_pages_free"] == [2, 2, 2, 1]
    adapter.acknowledge(
        endpoint.reply(
            adapter.reset_prefix_cache(),
            {"state": "prefixes_dropped"},
            source_prefixes=0,
        )
    )
    assert adapter.lookup_source_prefix(b"x" * 32) is None
    assert adapter.ledger["source_pages_free"] == [2, 2, 2, 1]
    with pytest.raises(native.IntegrationDisabled):
        adapter.get_computed_blocks("b")


@pytest.mark.parametrize("field", ["command_id", "epoch", "owner", "credits", "extent"])
def test_malformed_or_stale_ack_cannot_partially_update_state(field):
    adapter, endpoint = pair()
    command = adapter.admit("a", 0)
    ack = endpoint.reply(
        command, {"state": "admitted", "request": handle()}, requests=[request()]
    )
    if field in ("command_id", "epoch"):
        ack[field] += 1
    elif field == "owner":
        ack["snapshot"]["owner"] += 1
    elif field == "credits":
        ack["snapshot"]["source_pages_free"][0] = 5
    else:
        ack["snapshot"]["requests"][0]["committed_end"] = 1
    before = adapter.ledger
    with pytest.raises(native.AdapterError):
        adapter.acknowledge(ack)
    assert adapter.ledger == before
    assert adapter.retry() == command


def test_concurrent_producers_cannot_issue_two_unacknowledged_commands():
    adapter, _ = pair()

    def submit(slot):
        try:
            return adapter.admit(str(slot), slot)
        except native.CommandPending:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, [0, 1]))
    assert sum(result is not None for result in results) == 1


def test_cancel_while_publish_ack_is_lost_never_exposes_canceled_progress():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    begin(adapter, endpoint)
    adapter.acknowledge(
        endpoint.reply(adapter.reserve_publish("a", 2), {"state": "reserved"})
    )
    command = adapter.publish("a")
    ack = endpoint.reply(
        command,
        {"state": "published", "committed_end": 2},
        requests=[request(end=2)],
        transactions=0,
    )
    with pytest.raises(native.CommandPending):
        adapter.free("a")
    adapter.acknowledge(ack)
    with pytest.raises(native.AdapterError):
        adapter.source_committed_tokens("a")
    assert adapter.free("a")["command"]["op"] == "release"


def test_drained_write_failure_returns_authoritative_credits_without_publication():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    begin(adapter, endpoint)
    adapter.acknowledge(
        endpoint.reply(
            adapter.reserve_publish("a", 4),
            {"state": "reserved"},
            source_pages_free=[3, 3, 3, 3],
        )
    )
    adapter.acknowledge(
        endpoint.reply(
            adapter.publish("a"),
            {
                "state": "failed",
                "reason": "write failed and drained",
                "draining": False,
            },
            requests=[request()],
            transactions=0,
            source_pages_free=[4, 4, 4, 4],
        )
    )
    assert adapter.source_committed_tokens("a") == 0
    assert adapter.ledger["source_pages_free"] == [4, 4, 4, 4]
    assert adapter.begin_batch("a", 0, 1)["command"]["rows"] == 1


def test_accepted_ack_is_owned_and_slot_reuse_requires_new_generation():
    adapter, endpoint = pair()
    command = adapter.admit("a", 0)
    ack = endpoint.reply(
        command, {"state": "admitted", "request": handle()}, requests=[request()]
    )
    adapter.acknowledge(ack)
    ack["result"]["request"]["generation"] = 9
    assert adapter.free("a")["command"]["request"] == handle()
    adapter.acknowledge(
        endpoint.reply(adapter.retry(), {"state": "released"}, requests=[])
    )
    command = adapter.admit("a", 0)
    resumed = request(request_id=2)
    resumed["request"] = handle(generation=2)
    valid = endpoint.reply(
        command,
        {"state": "admitted", "request": handle(generation=2)},
        requests=[resumed],
    )
    stale = copy.deepcopy(valid)
    stale["result"]["request"]["generation"] = 1
    stale["snapshot"]["requests"][0]["request"]["generation"] = 1
    with pytest.raises(native.AdapterError):
        adapter.acknowledge(stale)
    assert adapter.retry() == command
    adapter.acknowledge(valid)
    assert adapter.begin_batch("a", 0, 1)["command"]["request"] == handle(generation=2)


def test_capacity_changes_and_invalid_acceptance_never_issue_or_publish():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    begin(adapter, endpoint, rows=2)
    for accepted in (-1, 3, True):
        with pytest.raises(native.AdapterError):
            adapter.reserve_publish("a", accepted)
    command = adapter.reserve_publish("a", 1)
    ack = endpoint.reply(command, {"state": "reserved"}, cache_bytes=999999)
    with pytest.raises(native.AdapterError):
        adapter.acknowledge(ack)
    assert adapter.ledger["cache_bytes"] == 123456
    assert adapter.retry() == command


def test_overpublication_ack_cannot_commit_rejected_speculative_suffix():
    adapter, endpoint = pair()
    admit(adapter, endpoint)
    begin(adapter, endpoint, rows=8)
    adapter.acknowledge(
        endpoint.reply(adapter.reserve_publish("a", 1), {"state": "reserved"})
    )
    command = adapter.publish("a")
    ack = endpoint.reply(
        command,
        {"state": "published", "committed_end": 8},
        requests=[request(end=8)],
        transactions=0,
    )
    with pytest.raises(native.AdapterError):
        adapter.acknowledge(ack)
    assert adapter.source_committed_tokens("a") == 0
    assert adapter.retry() == command
