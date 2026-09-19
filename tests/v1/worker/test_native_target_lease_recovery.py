# SPDX-License-Identifier: Apache-2.0
"""Regression tests: timed-out logits acquisition recovery must converge.

Fault model (CPU only, no CUDA/fleet/library installs): one acquire_logits ACK
is delayed past its client deadline, then delivered during owner-thread
cancellation. The real production classes are exercised end to end:

- vllm.v1.worker.native_target.NativeTargetRunner     
- vllm.v1.worker.native_target_binding.NativeClientBackend
- vllm_afd.native_target.client.NativeTargetClient / NativeLogitsLease
 

Only the native C ABI, the clock and the CUDA/torch surface are faked; they
are adapted from tests/unit/test_native_target_client.py in the vllm-afd
package. Wire-op ordering asserted here is the native consumer fence: a
recovered lease must observe release_logits (drain) before cancel, exactly
once, and a failed drain must retain entry + lease (fail closed).

Scope: cleanup/recovery only. This does not exercise and cannot explain the
separately-owned 512/1024 numerical divergence or the completed controls.
"""

import ctypes as ct
import importlib.util
import json
import sys
import types
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load real implementations without unrelated GPU infrastructure. Restore
# public aliases immediately to avoid polluting other tests during collection.
# The native client package must be installed or supplied on PYTHONPATH.
native = _load(ROOT / "vllm/v1/worker/native_target.py", "lease_recovery_native_subject")
_aliases = ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.native_target")
_previous = {name: sys.modules.get(name) for name in _aliases}
try:
    for name in _aliases[:-1]:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = []
            sys.modules[name] = stub
    sys.modules[_aliases[-1]] = native
    binding_module = _load(
        ROOT / "vllm/v1/worker/native_target_binding.py",
        "lease_recovery_binding_subject",
    )
finally:
    for name, module in _previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

from vllm_afd.errors import LeaseError  # noqa: E402
from vllm_afd.native_target.client import (  # noqa: E402
    NativeLogitsLease,
    NativeTargetClient,
    NativeTargetTimeout,
    VOCABULARY,
)


# ---------------------------------------------------------------------------
# CPU fakes for the native C ABI, monotonic clock and CUDA surface.
# Adapted from vllm-afd tests/unit/test_native_target_client.py.
# ---------------------------------------------------------------------------
class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ABI:
    """C-shaped fake: input JSON copied; result consumption and fences explicit."""

    def __init__(self):
        self.version = 1
        self.handle = 1009
        self.next_id = 1
        self.replies, self.command_ops = {}, {0: "initialize"}
        self.commands, self.result_calls, self.events = [], [], []
        self.hold, self.failure, self.mutate = set(), {}, {}
        self.status, self.destroy_codes = {}, deque()
        self.poll_states = deque()
        self.requests, self.jobs, self.generations = {}, {}, Counter()
        self.closed = self.destroyed = False
        self.capacity_available = True
        self.bank = None
        for name in ("create", "command", "result", "destroy"):
            setattr(self, "ds41rt_target_" + name, Function(getattr(self, name)))
        self.ds41rt_target_abi_version = Function(lambda: self.version)

    def create(self, buffer, length, out):
        self.config = json.loads(ct.string_at(buffer, length))
        if self.status.get("create"):
            return self.status["create"]
        ct.cast(out, ct.POINTER(ct.c_uint64))[0] = self.handle
        self.bank = dict(
            owner=self.config["owner"],
            capacity_rows=self.config["batch_tokens"],
            source_page_capacity=[100, 50, 25, 12],
            source_pages_free=[100, 50, 25, 12],
            source_payload_bytes=1000,
            cache_bytes=20000,
        )
        self.envelope(0, "initialize", dict(state="initialized", bank=self.bank))
        return 0

    def envelope(self, identifier, op, result):
        value = dict(command_id=identifier, ok=True, result=result)
        if op in self.failure:
            value = dict(
                command_id=identifier, ok=False, code=self.failure[op], error="injected " + op
            )
        if op in self.mutate:
            self.mutate[op](value)
        self.replies[identifier] = json.dumps(value).encode()

    def command(self, handle, buffer, length, out):
        assert handle == self.handle and not self.destroyed
        value = json.loads(ct.string_at(buffer, length))
        op = value["op"]
        if self.status.get(op):
            return self.status[op]
        identifier = self.next_id
        self.next_id += 1
        ct.cast(out, ct.POINTER(ct.c_uint64))[0] = identifier
        self.command_ops[identifier] = op
        self.commands.append(value)
        self.events.append(("command", identifier, op))
        request = value.get("request")
        ticket = value.get("ticket")
        if op == "info":
            end = None if request is None else self.requests[request["slot"]]["end"]
            result = dict(state="info", bank=self.bank, request=request, committed_end=end)
        elif op == "admit":
            slot = value["slot"]
            self.generations[slot] += 1
            request = dict(owner=self.config["owner"], slot=slot, generation=self.generations[slot])
            self.requests[slot] = dict(request=request, end=0)
            result = dict(state="admitted", request=request, committed_end=0, bank=self.bank)
        elif op == "submit":
            ticket = dict(request=request, lane=value["lane"], id=identifier + 100)
            self.jobs[ticket["id"]] = dict(
                ticket=ticket, work=value["work"], end=value["expected_committed_end"]
            )
            result = dict(state="prepared", ticket=ticket)
        elif op == "execute":
            result = dict(state="executing", ticket=ticket)
        elif op == "poll":
            state = self.poll_states.popleft() if self.poll_states else "ready"
            result = dict(
                state=state, ticket=ticket, error="native kernel failed" if state == "failed" else None
            )
        elif op == "acquire_logits":
            job = self.jobs[ticket["id"]]
            selected = job["work"]["selected"]
            result = dict(
                state="leased", ticket=ticket, lease=identifier + 500, device_pointer="0x1000",
                device_id=0, rows=len(selected), vocabulary=VOCABULARY, dtype="float32",
                bytes=len(selected) * VOCABULARY * 4, selected=selected,
                positions=[job["end"] + i for i in selected],
            )
        elif op == "release_logits":
            result = dict(state="consumed", ticket=ticket, lease=value["lease"])
        elif op == "cancel":
            self.jobs.pop(ticket["id"])
            result = dict(state="cancelled", ticket=ticket, bank=self.bank)
        elif op == "release":
            del self.requests[request["slot"]]
            result = dict(state="released", request=request, bank=self.bank)
        elif op == "shutdown":
            assert not self.jobs and not self.requests
            self.closed = True
            result = dict(state="closed")
        else:
            raise AssertionError(op)
        self.envelope(identifier, op, result)
        return 0

    def result(self, handle, identifier, buffer, capacity, written):
        assert handle == self.handle and not self.destroyed
        self.result_calls.append((identifier, capacity))
        op = self.command_ops[identifier]
        if self.status.get("result"):
            return self.status["result"]
        if op in self.hold or identifier in self.hold:
            return 4
        if identifier not in self.replies:
            return 3
        raw = self.replies[identifier]
        ct.cast(written, ct.POINTER(ct.c_size_t))[0] = len(raw)
        if capacity < len(raw):
            return 5
        ct.memmove(buffer, raw, len(raw))
        del self.replies[identifier]
        self.events.append(("consumed", identifier, op))
        return 0

    def destroy(self, handle):
        assert handle == self.handle
        self.events.append(("destroy",))
        code = self.destroy_codes.popleft() if self.destroy_codes else 0
        if not code:
            assert self.closed or "initialize" in self.failure
            self.destroyed = True
        return code

    # Test observability helpers -------------------------------------------
    def sent(self):
        return [event[2] for event in self.events if event[0] == "command"]

    def count(self, op):
        return self.sent().count(op)

    def tickets_of(self, op):
        return [
            command["ticket"]["id"] for command in self.commands if command["op"] == op
        ]


class LaneHoldABI(ABI):
    """Hold every acquire_logits reply except the first-sent until released.

    Models a second active lane whose acquisition ACK is delayed past its
    deadline while the first lane acquired normally.
    """

    def __init__(self):
        super().__init__()
        self.acquire_holds_released = False

    def result(self, handle, identifier, buffer, capacity, written):
        if (
            self.command_ops.get(identifier) == "acquire_logits"
            and not self.acquire_holds_released
        ):
            acquires = [
                i for i, op in self.command_ops.items() if op == "acquire_logits"
            ]
            if acquires.index(identifier) > 0:
                return 4  # NOT_READY: reply withheld
        return super().result(handle, identifier, buffer, capacity, written)


@dataclass(frozen=True)
class Device:
    type: str = "cuda"
    index: int = 0


class Torch:
    float32 = "float32"

    def __init__(self):
        self.stream = SimpleNamespace(device=Device(), cuda_stream=1234)
        self.device_id, self.capturing, self.copy_pointer = 0, False, False
        self.views = []
        self.cuda = SimpleNamespace(
            current_device=lambda: self.device_id,
            current_stream=lambda device: self.stream,
            is_current_stream_capturing=lambda: self.capturing,
        )

    def as_tensor(self, view, *, device):
        self.views.append(view)
        array = view.__cuda_array_interface__
        assert array["typestr"] == "<f4" and array["data"][1] is False
        assert device == Device()
        return SimpleNamespace(
            data_ptr=lambda: array["data"][0] + (4 if self.copy_pointer else 0),
            shape=array["shape"],
            dtype=self.float32,
            device=device,
            is_contiguous=lambda: True,
            source=view,
        )


# ---------------------------------------------------------------------------
# Harness: real runner + real binding backend + real client, faked ABI only.
# ---------------------------------------------------------------------------
TARGET_CONFIG = dict(
    owner=7001,
    snapshot="/checkpoint",
    native_lib="/native.so",
    peers=["a:1", "b:1", "c:1", "d:1"],
    batch_tokens=80,
    max_context_tokens=4096,
    slots=16,
    source_pool_budget_bytes=1024,
)

CLIENT_TIMEOUT_S = 0.01
CLIENT_POLL_S = 0.001


class FakeSampler:
    def __init__(self, events):
        self.events = events

    def update_requests(self, output):
        self.events.append("sampling_requests")

    def sample(self, grants, leases, grammar_output):  # pragma: no cover
        raise AssertionError("sampling is out of scope for these tests")

    def close(self):
        self.events.append("sampler_close")


class RunnerBinding:
    """Binds the real NativeClientBackend over the real client into the runner."""

    def __init__(self, harness):
        self.harness = harness
        self.events = harness.events

    def validate_config(self, config):  # pragma: no cover
        raise AssertionError("not used via runner construction here")

    def create_sampler(self, config, device):
        self.events.append("sampler_init")
        return FakeSampler(self.events)

    def create_backend(self, config, device, owner):
        assert owner == self.harness.client.owner
        self.events.append("backend_init")
        return self.harness.backend

    def create_scheduler(self, **kwargs):  # pragma: no cover
        raise AssertionError("scheduler is out of scope for these tests")


class Harness:
    def __init__(self, abi=None):
        self.abi = abi if abi is not None else ABI()
        self.clock = Clock()
        self.torch = Torch()
        self.client = NativeTargetClient(
            TARGET_CONFIG,
            enabled=True,
            library=self.abi,
            timeout_s=CLIENT_TIMEOUT_S,
            poll_interval_s=CLIENT_POLL_S,
            clock=self.clock,
            sleep=self.clock.sleep,
            torch_module=self.torch,
        )
        self.backend = binding_module.NativeClientBackend(self.client)
        self.client.initialize()  # idempotent; runner.load_model re-calls it
        self.events = []
        self._binding = RunnerBinding(self)

    def runner(self):
        config = SimpleNamespace(
            additional_config={"afd_native_target": {"enabled": True}},
            model_config=SimpleNamespace(max_model_len=1024),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        )
        runner = native.NativeTargetRunner(
            config, "cuda:0", self._binding, owner=TARGET_CONFIG["owner"]
        )
        runner.load_model()
        return runner

    def work(self, runner, lanes=1, step=1, rows=3):
        grants = tuple(
            native.NativeGrant(
                request_id=str(i),
                request=runner.admit(i, i + 1),
                lane=i,
                committed_end=0,
                tokens=tuple(range(rows)),
                selected=(rows - 1,),
                kind="prefill",
                placement=0,
            )
            for i in range(lanes)
        )
        return SimpleNamespace(
            native_target=native.NativeSchedule(TARGET_CONFIG["owner"], step, grants),
            num_scheduled_tokens={g.request_id: rows for g in grants},
            total_num_scheduled_tokens=rows * lanes,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )


def client_grant(request, lane):
    return SimpleNamespace(
        request_id=f"api-{lane}",
        request=request,
        lane=lane,
        committed_end=0,
        tokens=[7, 8, 9],
        selected=[0, 2],
        phase="full_target",
        kind="prefill",
        placement=0,
    )


def wire_order(abi, ops):
    """(op, ticket-id-or-None) in native command arrival order."""
    return [
        (command["op"], command.get("ticket", {}).get("id"))
        for command in abi.commands
        if command["op"] in ops
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_acquire_timeout_recovers_lease_cancels_and_shuts_down_cleanly():
    """timeout -> ACK recovery -> drain/close exactly once -> cancel -> shutdown.

    Before the fix this scenario raised LeaseError('cannot cancel before
    consumer lease drain and close') from every cancel/shutdown attempt,
    masked the original timeout, and stranded the recovered lease.
    """
    harness = Harness()
    abi = harness.abi
    runner = harness.runner()
    work = harness.work(runner, lanes=1)

    abi.hold.add("acquire_logits")
    with pytest.raises(NativeTargetTimeout):
        runner.execute_model(work)

    # Deliver the delayed ACK; recovery happens inside cancellation.
    abi.hold.clear()
    runner.cancel_step()
    runner.shutdown()

    # The native fence was honored before reclaim, exactly once.
    assert abi.count("acquire_logits") == 1  # never resubmitted
    assert abi.count("release_logits") == 1
    assert abi.count("cancel") == 1
    assert abi.count("release") == 1  # admission bookkeeping intact
    assert abi.count("shutdown") == 1
    assert abi.destroyed
    # Fence order for the recovered lease: drain strictly before reclaim,
    # admission released after cancel, shutdown last.
    released = abi.tickets_of("release_logits")[0]
    cancelled = abi.tickets_of("cancel")[0]
    assert wire_order(abi, ("release_logits", "cancel", "release", "shutdown")) == [
        ("release_logits", released),
        ("cancel", cancelled),
        ("release", None),
        ("shutdown", None),
    ]
    assert not harness.client.active_leases
    assert not harness.client.active_tickets


def test_second_lane_timeout_recovered_lease_and_consumed_lane_preserved():
    """Lane 0 leases and is consumed normally; lane 1's delayed ACK is
    recovered during cancellation. Each lease is drained/closed exactly once;
    the already-consumed lane 0 lease is never re-drained by recovery.
    """
    harness = Harness(LaneHoldABI())
    abi = harness.abi
    runner = harness.runner()
    work = harness.work(runner, lanes=2)

    with pytest.raises(NativeTargetTimeout):
        runner.execute_model(work)

    abi.acquire_holds_released = True  # lane 1 ACK delivered during cancel
    runner.cancel_step()
    runner.shutdown()

    assert abi.count("acquire_logits") == 2
    assert abi.count("release_logits") == 2  # one per lane, no duplicates
    assert abi.count("cancel") == 2
    assert abi.count("release") == 2
    assert abi.count("shutdown") == 1
    assert abi.destroyed
    released = [c["ticket"]["id"] for c in abi.commands if c["op"] == "release_logits"]
    cancelled = [c["ticket"]["id"] for c in abi.commands if c["op"] == "cancel"]
    assert len(set(released)) == 2 and len(set(cancelled)) == 2
    assert set(released) == set(cancelled)
    # Fence order: lane 0's runner-path consumption first; then the recovered
    # lane 1 lease is drained strictly before its reclaim.
    assert wire_order(abi, ("release_logits", "cancel")) == [
        ("release_logits", released[0]),
        ("cancel", cancelled[0]),
        ("release_logits", released[1]),
        ("cancel", cancelled[1]),
    ]
    assert not harness.client.active_leases


def test_recovery_timeout_resumes_same_command_and_entry_is_preserved():
    """Recovery failing again: the acquire ACK is still withheld during the
    first cancel attempt. The timeout must propagate (not LeaseError), the
    entry must be retained, no resubmission may occur, and a later attempt
    converges with exactly-once drain/close.
    """
    harness = Harness()
    abi = harness.abi
    runner = harness.runner()
    work = harness.work(runner, lanes=1)

    abi.hold.add("acquire_logits")
    with pytest.raises(NativeTargetTimeout):
        runner.execute_model(work)

    # First cancel attempt: ACK still withheld.
    with pytest.raises(NativeTargetTimeout):
        runner.cancel_step()
    assert abi.count("acquire_logits") == 1  # resumed, never resubmitted
    assert len(runner._active) == 1  # entry preserved for retry
    assert abi.count("cancel") == 0  # no reclaim bypassed the missing lease

    # ACK lands; the retry converges.
    abi.hold.clear()
    runner.cancel_step()
    assert len(runner._active) == 0
    runner.shutdown()

    assert abi.count("release_logits") == 1
    assert abi.count("cancel") == 1
    assert abi.destroyed


def test_failed_recovered_drain_preserves_fence_entry_and_lease():
    """No consumer-fence bypass: if the recovered lease cannot be drained
    (wrong CUDA surface), cancel must fail closed — no native cancel, no
    bookkeeping deletion — and a retry on a healthy surface converges.
    """
    harness = Harness()
    abi = harness.abi
    runner = harness.runner()
    work = harness.work(runner, lanes=1)

    abi.hold.add("acquire_logits")
    with pytest.raises(NativeTargetTimeout):
        runner.execute_model(work)
    abi.hold.clear()

    harness.torch.device_id = 1  # drain_consumers must reject this surface
    with pytest.raises(LeaseError):
        runner.cancel_step()
    assert abi.count("release_logits") == 0  # fence refused before any wire op
    assert abi.count("cancel") == 0  # reclaim not bypassed
    assert len(runner._active) == 1  # entry retained
    assert len(harness.client.active_leases) == 1  # lease retained, not dropped

    harness.torch.device_id = 0
    runner.cancel_step()
    runner.shutdown()
    assert abi.count("release_logits") == 1
    assert abi.count("cancel") == 1
    assert abi.destroyed


def test_backend_cancel_retains_other_recovered_lease_for_its_own_ticket():
    """Adversarial: TWO acquisitions pending when one backend.cancel recovers
    both replies. The canceled ticket's lease is drained/closed and the ticket
    reclaimed; the unrelated recovered lease must stay attached to its client
    job (retention/re-delivery via active_leases) — undrained, unclosed —
    until its own cancel call reclaims it exactly once.
    """
    harness = Harness()
    abi, client, backend = harness.abi, harness.client, harness.backend
    first = client.admit(0, 1)
    second = client.admit(1, 2)
    ticket0 = client.submit(client_grant(first, 0))
    ticket1 = client.submit(client_grant(second, 1))
    client.execute((ticket0, ticket1))

    abi.hold.add("acquire_logits")
    with pytest.raises(NativeTargetTimeout):
        backend.acquire_logits(ticket0)
    with pytest.raises(NativeTargetTimeout):
        backend.acquire_logits(ticket1)
    assert len(client.pending_commands) == 2
    abi.hold.clear()

    assert backend.cancel(ticket0) is False  # full target: admission retained
    # ticket0's lease drained/closed and its ticket reclaimed...
    assert abi.count("release_logits") == 1
    assert abi.count("cancel") == 1
    assert ticket0 not in client.active_tickets
    # ...while the unrelated recovered lease is retained, untouched:
    leases = client.active_leases
    assert len(leases) == 1
    survivor = leases[0]
    assert isinstance(survivor, NativeLogitsLease)
    assert survivor.ticket == ticket1
    assert survivor._drained is False and survivor._closed is False

    # Its own cancel call re-delivers cleanup exactly once.
    assert backend.cancel(ticket1) is False
    assert abi.count("release_logits") == 2
    assert abi.count("cancel") == 2
    assert ticket1 not in client.active_tickets
    assert not client.active_leases
    released = [c["ticket"]["id"] for c in abi.commands if c["op"] == "release_logits"]
    assert sorted(released) == sorted([ticket0.id, ticket1.id])
    client.close()
    assert abi.destroyed


def test_normal_cancel_path_unaffected_by_recovery_scan():
    """Guard: a healthy step's cancel still drains the runner-held lease once
    and cancels; the recovery scan must not interfere or double-drain.
    """
    harness = Harness()
    abi = harness.abi
    runner = harness.runner()
    work = harness.work(runner, lanes=1)

    runner.execute_model(work)  # acquire succeeds; lease held by entry
    runner.cancel_step()
    runner.shutdown()

    assert abi.count("acquire_logits") == 1
    assert abi.count("release_logits") == 1
    assert abi.count("cancel") == 1
    assert abi.destroyed
    assert not harness.client.active_leases


def test_minimal_client_standins_keep_existing_cancel_semantics():
    """Compatibility with the minimal client fakes used by the vLLM serving
    tests (pending_commands/active_tickets/active_requests only, no lease
    bookkeeping): the recovery scan must be inert for them.
    """
    request = SimpleNamespace(owner=17, slot=0, generation=1)
    ticket = SimpleNamespace(request=request)

    # Stream cancel: revocation reported once, no lease surface touched.
    stream_client = SimpleNamespace(
        pending_commands=(), active_tickets=(ticket,), active_requests=(request,)
    )

    def cancel(value):
        assert value is ticket
        stream_client.active_tickets = stream_client.active_requests = ()

    stream_client.cancel = cancel
    assert binding_module.NativeClientBackend(stream_client).cancel(ticket) is True

    # Lost commit ACK recovered: recovery consumes the pending command without
    # reporting revocation, and the missing active_leases surface is inert.
    commit_client = SimpleNamespace(
        pending_commands=(7,), active_tickets=(ticket,), active_requests=(request,)
    )

    def wait(command):
        assert command == 7
        commit_client.active_tickets = ()

    commit_client.wait = wait
    assert binding_module.NativeClientBackend(commit_client).cancel(ticket) is False
