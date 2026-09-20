# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for NativeScheduler slot-lease lifecycle ordering.

Purpose: pin the scheduler-side admission contract against the real worker
runner: occupied slots derive from scheduler-held leases, not worker state, so
a worker-side revocation alone never readmits a slot; retirement and
preemption release-and-validate the receipt before the lease leaves the pool;
a failed or invalid release keeps the lease reserved.

What is real here: the entire ``NativeScheduler`` class body is compiled
unchanged from ``vllm/v1/core/sched/native_scheduler.py`` (only module-level
import statements are stripped by AST; no method body is edited or
reimplemented), and it talks to a real ``NativeTargetRunner`` from
``vllm/v1/worker/native_target.py`` holding real ``NativeRequest`` leases and
``NativeCacheInfo`` receipts. What is stubbed: the executor transport (a fake
``collective_rpc`` mirroring the one-line ``native_*`` endpoints of
``gpu_worker.py:775-786``), engine-core value types the exercised paths merely
construct (PrefillStats, EngineCoreOutput, RequestStatus, PauseState),
request objects, and the fake native backend/binding shared with
``tests/v1/worker/test_native_target.py``. ``schedule()`` and
``update_from_output()`` need the full SchedulerOutput machinery and are not
exercised; ``_admit_waiting`` is driven directly. No CUDA, no sampler parity,
no full-stack claim.
"""

import ast
import importlib.util
import sys
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
BASE_SPEC = importlib.util.spec_from_file_location(
    "native_scheduler_contract_base", ROOT / "tests/v1/worker/test_native_target.py"
)
assert BASE_SPEC and BASE_SPEC.loader
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)

native = base.native
output_type = base.output_type


class ContractBackend(base.Backend):
    """The base fake backend plus the capacity query the scheduler RPCs."""

    def can_prepare(self, work):
        self.events.append(
            ("can_prepare", tuple((r.slot, r.generation, rows) for r, rows in work))
        )
        return True


base.Backend = ContractBackend
binding = base.binding


class PauseState(Enum):
    UNPAUSED = 0
    PAUSED_ALL = 1


class PrefillStats:
    def set(self, *args):
        return None


def _unused(*args, **kwargs):
    raise AssertionError("stub outside the exercised scheduler paths was called")


RequestStatus = SimpleNamespace(
    RUNNING="running",
    PREEMPTED="preempted",
    FINISHED_ERROR="finished_error",
    FINISHED_ABORTED="finished_aborted",
)

_SCHEDULER_PATH = ROOT / "vllm/v1/core/sched/native_scheduler.py"
_TREE = ast.parse(_SCHEDULER_PATH.read_text())
_TREE.body = [
    node for node in _TREE.body if not isinstance(node, (ast.Import, ast.ImportFrom))
]
# Only module-level imports may be stripped; the executed bodies stay intact.
assert not any(
    isinstance(inner, (ast.Import, ast.ImportFrom))
    for node in _TREE.body
    for inner in ast.walk(node)
)
_SCOPE = {
    "defaultdict": defaultdict,
    "deque": deque,
    "PauseState": PauseState,
    "SchedulerInterface": object,
    "NewRequestData": _unused,
    "SchedulerOutput": _unused,
    "check_stop": _unused,
    "EngineCoreEventType": SimpleNamespace(QUEUED="queued", SCHEDULED="scheduled"),
    "EngineCoreOutput": lambda **kwargs: SimpleNamespace(**kwargs),
    "EngineCoreOutputs": _unused,
    "PrefillStats": PrefillStats,
    "SchedulerStats": _unused,
    "SpecDecodingStats": _unused,
    "RequestStatus": RequestStatus,
    "NativeCacheInfo": native.NativeCacheInfo,
    "NativeGrant": native.NativeGrant,
    "NativeSchedule": native.NativeSchedule,
    "NativeStepResult": native.NativeStepResult,
}
exec(compile(_TREE, str(_SCHEDULER_PATH), "exec"), _SCOPE)
NativeScheduler = _SCOPE["NativeScheduler"]


class Executor:
    """Fake transport; endpoints mirror gpu_worker.py:775-786 onto a real runner."""

    def __init__(self, runner):
        self.runner, self.log = runner, []
        self.fail_release = self.corrupt_receipt = False

    def collective_rpc(self, method, args=()):
        self.log.append((method, *args))
        if method == "native_cache_info":
            return [self.runner.cache_info()]
        if method == "native_admit":
            return [self.runner.admit(*args)]
        if method == "native_can_prepare":
            return [self.runner.can_prepare(*args)]
        assert method == "native_release"
        if self.fail_release:
            raise RuntimeError("native release transport failed")
        self.runner.release(args[0])
        receipt = self.runner.cache_info()
        return [SimpleNamespace()] if self.corrupt_receipt else [receipt]


class FakeRequest:
    def __init__(self, name, prompt=4, max_tokens=4):
        self.request_id, self.client_index = name, 0
        self.sampling_params = SimpleNamespace(
            prompt_logprobs=None,
            logprobs=None,
            logprob_token_ids=(),
            thinking_token_budget=None,
            trace_decode_token_ids=False,
            extra_args=None,
        )
        self.pooling_params = self.prompt_embeds = self.lora_request = None
        self.prompt_token_ids = [3] * prompt
        self.mm_features = ()
        self.resumable = self.use_structured_output = False
        self.priority = 0
        self.recurrent_instruction_boundary = self.trace_headers = None
        self.num_prompt_tokens, self.max_tokens = prompt, max_tokens
        self.num_computed_tokens = 0
        self.status, self.events = None, []

    def record_event(self, event):
        self.events.append(event)

    def take_events(self):
        return []

    def get_finished_reason(self):
        return "error"


def vllm_config(slots, stream):
    options = {"slots": slots, "batch_tokens": 256 if stream else 16}
    if stream:
        options["prefill_mode"] = "encoder_stream"
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        additional_config={"afd_native_target": options},
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=256),
    )


def build(binding, slots, stream=False):
    runner = (base.stream_ready if stream else base.ready)(binding)
    executor = Executor(runner)
    scheduler = NativeScheduler(
        vllm_config(slots, stream), executor, runner.cache_info(), None
    )
    return runner, executor, scheduler


def stream_step(step, name, lease, rows=370):
    grant = native.NativeGrant(
        name,
        lease,
        0,
        0,
        tuple(range(rows)),
        (rows - 1,),
        "prefill",
        0,
        phase="encoder_stream",
        chunk_rows=80,
    )
    return SimpleNamespace(
        native_target=native.NativeSchedule(17, step, (grant,)),
        num_scheduled_tokens={name: rows},
        total_num_scheduled_tokens=rows,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )


def admits(executor):
    return [entry for entry in executor.log if entry[0] == "native_admit"]


def test_revoked_but_held_lease_blocks_slot_until_successful_release(binding):
    runner, executor, scheduler = build(binding, slots=2, stream=True)
    a = FakeRequest("a", prompt=370, max_tokens=2)
    scheduler.add_request(a)
    scheduler._admit_waiting()
    lease_a = scheduler.leases["a"]
    assert lease_a.slot == 0 and a.status == RequestStatus.RUNNING
    # Worker-side stream cancellation revokes the bank natively; the scheduler
    # still holds the lease and has not been told anything.
    runner.execute_model(stream_step(1, "a", lease_a))
    runner.cancel_step()
    assert lease_a not in binding.backend.ends
    assert ("free", 0) not in binding.events
    b = FakeRequest("b")
    scheduler.add_request(b)
    scheduler._admit_waiting()
    # Occupancy comes from scheduler leases, so the revoked slot 0 stays
    # reserved and the successor lands on slot 1.
    assert scheduler.leases["b"].slot == 1
    assert [entry[1] for entry in admits(executor)] == [0, 1]
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert "a" not in scheduler.leases and "a" not in scheduler.requests
    # The consumed revocation receipt suppressed a second native free.
    assert ("free", 0) not in binding.events
    c = FakeRequest("c")
    scheduler.add_request(c)
    scheduler._admit_waiting()
    lease_c = scheduler.leases["c"]
    assert lease_c.slot == 0 and lease_c.generation == lease_a.generation + 1
    release_at = executor.log.index(("native_release", lease_a))
    assert release_at < executor.log.index(admits(executor)[2])
    runner.shutdown()


def test_retirement_releases_before_successor_admission(binding):
    runner, executor, scheduler = build(binding, slots=1)
    a, b = FakeRequest("a"), FakeRequest("b")
    scheduler.add_request(a)
    scheduler.add_request(b)
    scheduler._admit_waiting()
    lease_a = scheduler.leases["a"]
    assert list(scheduler.leases) == ["a"] and list(scheduler.waiting) == [b]
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    scheduler._admit_waiting()
    lease_b = scheduler.leases["b"]
    assert lease_b.slot == 0 and lease_b.generation == lease_a.generation + 1
    release_at = executor.log.index(("native_release", lease_a))
    assert release_at < executor.log.index(admits(executor)[1])
    # The native free itself lands before the successor admission.
    assert binding.events.count(("free", 0)) == 1
    assert binding.events.index(("free", 0)) < binding.events.index(("admit", 0, 2))
    runner.shutdown()


def test_failed_release_keeps_lease_reserved_and_is_retryable(binding):
    runner, executor, scheduler = build(binding, slots=1)
    a, b = FakeRequest("a"), FakeRequest("b")
    scheduler.add_request(a)
    scheduler.add_request(b)
    scheduler._admit_waiting()
    lease_a = scheduler.leases["a"]
    executor.fail_release = True
    with pytest.raises(RuntimeError, match="transport"):
        scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    # The transport failed before the worker freed anything: the lease stays
    # reserved, retirement did not proceed, and no admission can take slot 0.
    assert scheduler.leases["a"] == lease_a and "a" in scheduler.requests
    assert ("free", 0) not in binding.events
    scheduler._admit_waiting()
    assert "b" not in scheduler.leases and len(admits(executor)) == 1
    executor.fail_release = False
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    scheduler._admit_waiting()
    assert scheduler.leases["b"].slot == 0
    assert binding.events.count(("free", 0)) == 1
    runner.shutdown()


def test_invalid_release_receipt_keeps_slot_out_of_the_pool(binding):
    runner, executor, scheduler = build(binding, slots=3)
    a, b = FakeRequest("a"), FakeRequest("b")
    scheduler.add_request(a)
    scheduler.add_request(b)
    scheduler._admit_waiting()
    lease_a = scheduler.leases["a"]
    assert (lease_a.slot, scheduler.leases["b"].slot) == (0, 1)
    executor.corrupt_receipt = True
    with pytest.raises(RuntimeError, match="cache receipt"):
        scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    executor.corrupt_receipt = False
    # The worker did free the bank, but the receipt was rejected, so the
    # scheduler conservatively keeps the lease: slot 0 leaks rather than being
    # readmitted against a bank whose free was never acknowledged.
    assert binding.events.count(("free", 0)) == 1
    assert scheduler.leases["a"] == lease_a and "a" in scheduler.requests
    c = FakeRequest("c")
    scheduler.add_request(c)
    scheduler._admit_waiting()
    assert scheduler.leases["c"].slot == 2
    assert 0 not in {entry[1] for entry in admits(executor)[2:]}


def test_repeated_revoke_release_readmit_cycles_reuse_one_slot(binding):
    runner, executor, scheduler = build(binding, slots=1, stream=True)
    generation = 0
    for cycle in range(1, 4):
        name = f"r{cycle}"
        request = FakeRequest(name, prompt=370, max_tokens=2)
        scheduler.add_request(request)
        scheduler._admit_waiting()
        lease = scheduler.leases[name]
        assert lease.slot == 0 and lease.generation == generation + 1
        generation = lease.generation
        runner.execute_model(stream_step(cycle, name, lease))
        runner.cancel_step()
        scheduler.finish_requests(name, RequestStatus.FINISHED_ABORTED)
        assert name not in scheduler.leases
        release_at = executor.log.index(("native_release", lease))
        assert release_at == len(executor.log) - 1
    # Every cycle's release consumed its revocation receipt: no native free
    # ever fired for the slot, and each release preceded the next admission.
    assert ("free", 0) not in binding.events
    assert len(admits(executor)) == 3
    runner.shutdown()


def test_preemption_releases_before_slot_returns_to_pool(binding):
    runner, executor, scheduler = build(binding, slots=1)
    a = FakeRequest("a")
    scheduler.add_request(a)
    scheduler._admit_waiting()
    first = scheduler.leases["a"]
    assert scheduler.reset_prefix_cache(reset_running_requests=True) is True
    assert not scheduler.leases and list(scheduler.waiting) == [a]
    assert a.status == RequestStatus.PREEMPTED and a.num_computed_tokens == 0
    assert binding.events.count(("free", 0)) == 1
    scheduler._admit_waiting()
    second = scheduler.leases["a"]
    assert second.slot == 0 and second.generation == first.generation + 1
    assert binding.events.index(("free", 0)) < binding.events.index(("admit", 0, 2))
    runner.shutdown()
