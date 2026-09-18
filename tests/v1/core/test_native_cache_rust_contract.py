# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real serde/CacheCommands boundary with CPU-only device completion controls.

Purpose: catch agreement bugs hidden by independently scripted endpoint tests.
I/O: actual schema-1 envelopes/ACKs over a bounded local JSON-lines process.
Failures: retry mutation, false accepted publication, lost credits, stale
generations, premature reader free. No GPU or duplicated Python page allocator.
"""

import copy
import importlib.util
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "native_cache_adapter",
    Path(__file__).parents[3] / "vllm/v1/core/native_cache_adapter.py",
)
assert SPEC and SPEC.loader
native = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = native
SPEC.loader.exec_module(native)

BINARY = os.environ.get("AFD_NATIVE_CACHE_CONTRACT")
pytestmark = pytest.mark.skipif(
    not BINARY, reason="set AFD_NATIVE_CACHE_CONTRACT to the built Rust CPU example"
)


class RustEndpoint:
    def __init__(self, owner=17, pages=(4, 4, 4, 4)):
        assert BINARY and Path(BINARY).is_file()
        self.process = subprocess.Popen(
            [BINARY, str(owner), json.dumps(pages)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.buffer = b""
        try:
            self.initial = self.read()["snapshot"]
        except BaseException:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + 5
        assert self.process.stdout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([self.process.stdout], [], [], remaining)[0]
            ):
                raise AssertionError("Rust contract endpoint timed out")
            data = os.read(self.process.stdout.fileno(), 16384)
            if not data:
                raise AssertionError("Rust contract endpoint exited before response")
            self.buffer += data
            assert len(self.buffer) <= 131072
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def exchange(self, value):
        assert self.process.stdin
        data = json.dumps(value).encode() + b"\n"
        assert len(data) <= 131072
        self.process.stdin.write(data)
        self.process.stdin.flush()
        return self.read()

    def ack(self, adapter, envelope):
        result = self.exchange(envelope)
        assert "error" not in result, result
        adapter.acknowledge(result)
        return result

    def control(self, control, **fields):
        result = self.exchange({"control": control, **fields})
        assert "error" not in result, result
        return result

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            if self.process.stdout:
                self.process.stdout.close()
            if self.process.stderr:
                self.process.stderr.close()


@pytest.fixture
def rust():
    processes = []

    def create(owner=17, pages=(4, 4, 4, 4)):
        endpoint = RustEndpoint(owner, pages)
        processes.append(endpoint)
        return native.NativeCacheAdapter(endpoint.initial, capacity_rows=1024), endpoint

    yield create
    for endpoint in processes:
        endpoint.close()


def admit(adapter, endpoint, name="a", slot=0):
    return endpoint.ack(adapter, adapter.admit(name, slot))["result"]["request"]


def begin(adapter, endpoint, name="a", lane=0, rows=8):
    return endpoint.ack(adapter, adapter.begin_batch(name, lane, rows))["result"][
        "transaction"
    ]


def commit(adapter, endpoint, name="a", lane=0, rows=8, accepted=8):
    transaction = begin(adapter, endpoint, name, lane, rows)
    endpoint.ack(adapter, adapter.reserve_publish(name, accepted))
    endpoint.control("ready", transaction_id=transaction["id"])
    return endpoint.ack(adapter, adapter.publish(name))


def test_real_serde_and_last_ack_retry_do_not_reapply_admission(rust):
    adapter, endpoint = rust()
    command = adapter.admit("a", 0)
    lost_ack = endpoint.exchange(command)
    assert lost_ack["result"]["state"] == "admitted"
    assert adapter.ledger["requests"] == []
    changed = copy.deepcopy(command)
    changed["command"]["slot"] = 1
    rejected = endpoint.exchange(changed)
    assert "error" in rejected
    assert rejected["snapshot"] == lost_ack["snapshot"]
    retried_ack = endpoint.exchange(adapter.retry())
    assert retried_ack == lost_ack
    assert adapter.acknowledge(retried_ack)
    assert adapter.acknowledge(lost_ack) is False
    assert len(adapter.ledger["requests"]) == 1


def test_actual_native_publication_commits_only_accepted_rows_after_completion(rust):
    adapter, endpoint = rust()
    admit(adapter, endpoint)
    transaction = begin(adapter, endpoint, rows=257)
    endpoint.ack(adapter, adapter.reserve_publish("a", 127))
    assert adapter.ledger["source_pages_free"] == [3, 3, 3, 3]
    pending = endpoint.ack(adapter, adapter.publish("a"))
    assert pending["result"]["state"] == "pending"
    assert adapter.source_committed_tokens("a") == 0
    endpoint.control("ready", transaction_id=transaction["id"])
    lost = endpoint.exchange(adapter.publish("a"))
    assert adapter.source_committed_tokens("a") == 0
    assert endpoint.exchange(adapter.retry()) == lost
    adapter.acknowledge(lost)
    assert adapter.source_committed_tokens("a") == 127
    commit(adapter, endpoint, rows=4, accepted=0)
    assert adapter.source_committed_tokens("a") == 127


def test_actual_two_lanes_publish_b_before_a_and_cancel_keeps_reserved_pages(rust):
    adapter, endpoint = rust()
    admit(adapter, endpoint)
    admit(adapter, endpoint, "b", 1)
    first = begin(adapter, endpoint, "a", 0, 257)
    second = begin(adapter, endpoint, "b", 1, 8)
    endpoint.ack(adapter, adapter.reserve_publish("a", 255))
    endpoint.ack(adapter, adapter.reserve_publish("b", 4))
    endpoint.control("ready", transaction_id=second["id"])
    endpoint.ack(adapter, adapter.publish("b"))
    assert adapter.source_committed_tokens("b") == 4
    assert adapter.source_committed_tokens("a") == 0
    before = adapter.ledger["source_pages_free"]
    closing = endpoint.ack(adapter, adapter.free("a"))
    assert closing["result"]["state"] == "closing"
    assert adapter.ledger["source_pages_free"] == before
    endpoint.control("ready", transaction_id=first["id"])
    endpoint.ack(adapter, adapter.drain("a"))
    assert adapter.ledger["source_pages_free"] == [3, 3, 3, 3]
    endpoint.ack(adapter, adapter.free("b"))
    assert adapter.ledger["source_pages_free"] == [4, 4, 4, 4]


def test_real_shared_tail_forks_readers_reset_and_drain_preserve_native_credits(rust):
    adapter, endpoint = rust()
    original = admit(adapter, endpoint)
    commit(adapter, endpoint, rows=255, accepted=255)
    key = b"x" * 32
    endpoint.ack(adapter, adapter.retain_source_prefix("a", key))
    for name, slot in (("b", 1), ("c", 2)):
        endpoint.ack(adapter, adapter.restore_source_prefix(key, name, slot))
    assert adapter.ledger["source_pages_free"] == [3, 3, 3, 3]
    transaction = begin(adapter, endpoint, "b", 1, 2)
    endpoint.ack(adapter, adapter.reserve_publish("b", 2))
    queued = endpoint.control("snapshot")["device"]["queued"]
    copies = queued[str(transaction["id"])]["tail_copies"]
    assert copies == [1, 1, 1, 1]
    assert adapter.ledger["source_pages_free"] == [2, 2, 2, 1]
    endpoint.control("ready", transaction_id=transaction["id"])
    endpoint.ack(adapter, adapter.publish("b"))
    endpoint.control("hold_reader", reader_id=1, request=original)
    endpoint.ack(adapter, adapter.reset_prefix_cache())
    assert adapter.lookup_source_prefix(key) is None
    endpoint.ack(adapter, adapter.free("a"))
    endpoint.ack(adapter, adapter.free("c"))
    assert adapter.ledger["source_pages_free"] == [2, 2, 2, 1]
    endpoint.control("drop_reader", reader_id=1)
    endpoint.ack(adapter, adapter.drain("a"))
    assert adapter.ledger["source_pages_free"] == [3, 3, 3, 2]
    endpoint.ack(adapter, adapter.free("b"))
    assert adapter.ledger["source_pages_free"] == [4, 4, 4, 4]
    with pytest.raises(native.IntegrationDisabled):
        adapter.get_computed_blocks("b")


def test_actual_fourth_pool_failure_rolls_back_before_smaller_accepted_retry(rust):
    adapter, endpoint = rust(pages=(2, 2, 2, 1))
    admit(adapter, endpoint)
    transaction = begin(adapter, endpoint, rows=257)
    command = adapter.reserve_publish("a", 257)
    rejected = endpoint.exchange(command)
    assert "error" in rejected
    assert rejected["snapshot"] == adapter.ledger
    adapter.reject(command["owner"], command["epoch"], command["command_id"])
    endpoint.ack(adapter, adapter.reserve_publish("a", 127))
    endpoint.control("ready", transaction_id=transaction["id"])
    endpoint.ack(adapter, adapter.publish("a"))
    endpoint.ack(adapter, adapter.free("a"))
    assert adapter.ledger["source_pages_free"] == [2, 2, 2, 1]


def test_actual_start_failure_and_drained_write_failure_never_publish(rust):
    adapter, endpoint = rust()
    admit(adapter, endpoint)
    transaction = begin(adapter, endpoint)
    endpoint.control("fail_next_start")
    failed = endpoint.ack(adapter, adapter.reserve_publish("a", 4))
    assert failed["result"]["draining"] is True
    pending = endpoint.ack(adapter, adapter.abort_batch("a"))
    assert pending["result"]["state"] == "pending"
    endpoint.control("ready", transaction_id=transaction["id"])
    endpoint.ack(adapter, adapter.abort_batch("a"))
    transaction = begin(adapter, endpoint)
    endpoint.ack(adapter, adapter.reserve_publish("a", 4))
    endpoint.control("fail", transaction_id=transaction["id"])
    failed = endpoint.ack(adapter, adapter.publish("a"))
    assert failed["result"]["draining"] is False
    assert adapter.source_committed_tokens("a") == 0
    assert adapter.ledger["source_pages_free"] == [4, 4, 4, 4]


def test_actual_process_restart_rejects_old_owner_and_requires_recompute(rust):
    adapter, first = rust()
    admit(adapter, first)
    command = adapter.retain_source_prefix("a", b"x" * 32)
    old_ack = first.exchange(command)
    adapter.acknowledge(old_ack)
    first.close()
    _, restarted = rust(owner=18)
    assert adapter.restart(restarted.initial) == ("a",)
    assert adapter.lookup_source_prefix(b"x" * 32) is None
    assert "error" in restarted.exchange(command)
    with pytest.raises(native.AdapterError):
        adapter.acknowledge(old_ack)
    request = admit(adapter, restarted)
    assert request["owner"] == 18
    assert request["generation"] == 1
