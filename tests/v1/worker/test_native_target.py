# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for allocation delegation and target-result ownership.

Purpose: select the native owner before ordinary model/cache construction.
I/O: explicit token grants, native handles/capacity and ModelRunnerOutput.
Failures: duplicate allocation, premature logits release/publication, failed
construction and stale grants. Fake native completion is the cheapest gate;
these tests do not claim CUDA execution or sampler numerical parity.
"""

import ast
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location(
    "native_target_test_subject", ROOT / "vllm/v1/worker/native_target.py"
)
assert SPEC and SPEC.loader
native = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = native
SPEC.loader.exec_module(native)


@pytest.fixture
def output_type(monkeypatch):
    """Load the real output dataclasses with only unused GPU imports stubbed."""
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.native_target", native)
    for name, attributes in {
        "torch": {"Tensor": type("Tensor", (), {})},
        "vllm.compilation.cuda_graph": {"CUDAGraphStat": object},
        "vllm.v1.core.sched.output": {"SchedulerOutput": object},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "vllm.v1.outputs", ROOT / "vllm/v1/outputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.ModelRunnerOutput


def config(enabled=True):
    return SimpleNamespace(
        additional_config={"afd_native_target": {"enabled": enabled}},
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            world_size=1,
            data_parallel_size=1,
            enable_dbo=False,
            enable_fault_tolerance=False,
            distributed_executor_backend="uni",
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(
            runner_type="generate",
            enable_sleep_mode=False,
        ),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        weight_transfer_config=None,
    )


class Lease:
    def __init__(self, backend, ticket):
        self.backend, self.ticket = backend, ticket
        self.drained = False

    def drain_consumers(self):
        self.backend.events.append(("drain", self.ticket))
        if self.backend.fail_drain:
            raise RuntimeError("consumer event incomplete")
        self.drained = True

    def close(self):
        assert self.drained
        self.backend.events.append(("release_logits", self.ticket))
        self.backend.leases.remove(self.ticket)


class Backend:
    def __init__(self, owner, events):
        self.owner, self.events = owner, events
        self.ends, self.active, self.leases = {}, {}, set()
        self.generations = {}
        self.fail_execute = self.fail_drain = self.fail_commit = False

    def info(self):
        return native.NativeCacheInfo(self.owner, 256, (4,) * 4, (4,) * 4, 4096)

    def admit(self, slot, request_id):
        self.generations[slot] = self.generations.get(slot, 0) + 1
        handle = native.NativeRequest(self.owner, slot, self.generations[slot])
        self.ends[handle] = 0
        self.events.append(("admit", slot, request_id))
        return handle

    def committed_end(self, request):
        return self.ends[request]

    def submit(self, grant):
        self.events.append(("submit", grant.lane))
        self.active[grant.lane] = grant
        return grant.lane

    def execute(self, tickets):
        self.events.append(("execute", tickets))
        if self.fail_execute:
            raise RuntimeError("rank failed")

    def acquire_logits(self, ticket):
        self.events.append(("logits", ticket))
        self.leases.add(ticket)
        return Lease(self, ticket)

    def commit(self, ticket, accepted):
        assert ticket not in self.leases
        self.events.append(("commit", ticket, accepted))
        if self.fail_commit:
            raise RuntimeError("publication failed")
        grant = self.active.pop(ticket)
        self.ends[grant.request] += accepted
        return self.ends[grant.request]

    def cancel(self, ticket):
        assert ticket not in self.leases
        self.events.append(("cancel", ticket))
        grant = self.active.pop(ticket, None)
        if grant is not None and grant.phase == "encoder_stream":
            del self.ends[grant.request]
            return True
        return False

    def release(self, request):
        assert not any(g.request == request for g in self.active.values())
        self.events.append(("free", request.slot))
        del self.ends[request]

    def close(self):
        assert not self.leases
        self.events.append("backend_close")


class Binding:
    def __init__(self, output_type):
        self.events = []
        self.output_type = output_type
        self.fail_load = self.fail_sample = self.fail_scheduler = False
        self.accepted = None
        self.generated = None

    def validate_config(self, config):
        self.events.append("validate")

    def create_sampler(self, config, device):
        self.events.append("sampler_init")
        return self

    def create_backend(self, config, device, owner):
        self.events += ["weights_load", "native_plan", "native_cache_init"]
        if self.fail_load:
            raise MemoryError("native cache allocation failed")
        self.backend = Backend(owner, self.events)
        return self.backend

    def create_scheduler(self, **kwargs):
        self.events.append("scheduler_init")
        assert kwargs["cache_info"] == self.backend.info()
        if self.fail_scheduler:
            raise RuntimeError("scheduler binding failed")
        return SimpleNamespace(connector=None, ec_connector=None)

    def sample(self, grants, leases, grammar_output):
        assert len(leases) == len(grants)
        assert self.backend.leases == {g.lane for g in grants}
        self.events.append("sample")
        if self.fail_sample:
            raise RuntimeError("sampler failed after launch")
        names = [g.request_id for g in grants]
        output = self.output_type(
            names,
            dict(zip(names, range(len(names)))),
            self.generated if self.generated is not None else [[7]] * len(names),
        )
        accepted = self.accepted or tuple(len(g.tokens) for g in grants)
        return native.NativeSample(output, accepted)

    def update_requests(self, output):
        self.events.append("sampling_requests")

    def close(self):
        self.events.append("sampler_close")


@pytest.fixture
def binding(output_type, monkeypatch):
    binding = Binding(output_type)
    monkeypatch.setattr(native, "_binding", None)
    native.register_native_target(binding)
    return binding


def ready(binding):
    runner = native.NativeTargetRunner(config(), "cuda:0", binding, owner=17)
    runner.load_model()
    return runner


def schedule(runner, rows=3, lanes=1, step=1):
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
        native_target=native.NativeSchedule(17, step, grants),
        num_scheduled_tokens={g.request_id: rows for g in grants},
        total_num_scheduled_tokens=rows * lanes,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )


def test_default_off_and_missing_binding_fail_before_construction(monkeypatch):
    monkeypatch.setattr(native, "_binding", None)
    assert native.get_native_target_binding(config(False)) is None
    with pytest.raises(native.NativeTargetUnavailable, match="binding"):
        native.get_native_target_binding(config())


@pytest.mark.parametrize("feature", ["prefix", "async", "speculation", "worker"])
def test_unbound_serving_features_reject_before_factory(binding, feature):
    cfg = config()
    if feature == "prefix":
        cfg.cache_config.enable_prefix_caching = True
    elif feature == "async":
        cfg.scheduler_config.async_scheduling = True
    elif feature == "speculation":
        cfg.speculative_config = object()
    else:
        cfg.parallel_config.distributed_executor_backend = "mp"
    with pytest.raises(native.NativeTargetUnavailable):
        native.get_native_target_binding(cfg)
    assert not binding.events


def test_load_plan_cache_once_and_both_lanes_before_execute(binding, output_type):
    runner = ready(binding)
    with pytest.raises(RuntimeError, match="already"):
        runner.load_model()
    work = schedule(runner, lanes=2)
    assert runner.execute_model(work) is None
    output = runner.sample_tokens(None)
    assert isinstance(output, output_type)
    assert output.req_ids == ["0", "1"]
    assert output.native_target.committed_ends == (("0", 3), ("1", 3))
    events = binding.events
    assert events.count("weights_load") == events.count("native_cache_init") == 1
    assert events.index(("submit", 1)) < events.index(("execute", (0, 1)))
    assert events.index("sample") < events.index(("drain", 0))
    assert events.index(("release_logits", 1)) < events.index(("commit", 0, 3))
    runner.release(work.native_target.grants[0].request)
    runner.shutdown()


def test_cache_failure_never_exposes_capacity_and_closes_sampler(binding):
    binding.fail_load = True
    runner = native.NativeTargetRunner(config(), "cuda:0", binding, owner=17)
    with pytest.raises(MemoryError):
        runner.load_model()
    with pytest.raises(RuntimeError, match="ready"):
        runner.cache_info()
    assert binding.events[-1] == "sampler_close"


@pytest.mark.parametrize("failure", ["execute", "sample"])
def test_failed_step_drains_and_cancels_without_publication(binding, failure):
    runner = ready(binding)
    work = schedule(runner, lanes=2)
    if failure == "execute":
        binding.backend.fail_execute = True
    else:
        binding.fail_sample = True
    with pytest.raises(RuntimeError):
        runner.execute_model(work)
        runner.sample_tokens(None)
    assert not any(isinstance(e, tuple) and e[0] == "commit" for e in binding.events)
    assert not binding.backend.active
    assert not binding.backend.leases
    runner.shutdown()


def test_async_constructor_timeout_retains_owner_until_cleanup(binding):
    closed = []

    def construct(*args):
        backend = binding.backend = Backend(17, binding.events)

        def initialize():
            raise TimeoutError("initial native ACK pending")

        backend.initialize = initialize
        backend.close = lambda: closed.append(backend)
        return backend

    binding.create_backend = construct
    runner = native.NativeTargetRunner(config(), "cuda:0", binding, owner=17)
    with pytest.raises(TimeoutError, match="ACK pending"):
        runner.load_model()
    assert closed == [binding.backend]


def test_failed_consumer_drain_retains_leases_until_successful_shutdown(binding):
    runner = ready(binding)
    runner.execute_model(schedule(runner))
    binding.backend.fail_drain = True
    with pytest.raises(RuntimeError, match="consumer event"):
        runner.sample_tokens(None)
    with pytest.raises(RuntimeError, match="consumer event"):
        runner.shutdown()
    assert binding.backend.leases == {0}
    assert "backend_close" not in binding.events
    binding.backend.fail_drain = False
    runner.shutdown()
    assert not binding.backend.leases


def test_concurrent_cancel_cannot_reclaim_sampler_borrow(binding):
    runner = ready(binding)
    runner.execute_model(schedule(runner))
    sampling, cancel_entered, complete = Event(), Event(), Event()
    original_sample = binding.sample

    def held_sample(*args):
        sampling.set()
        assert complete.wait(2)
        return original_sample(*args)

    def cancel():
        cancel_entered.set()
        runner.cancel_step()

    binding.sample = held_sample
    with ThreadPoolExecutor(max_workers=2) as pool:
        sample = pool.submit(runner.sample_tokens, None)
        try:
            assert sampling.wait(2)
            cancellation = pool.submit(cancel)
            assert cancel_entered.wait(2)
            assert binding.backend.leases == {0}
            assert not cancellation.done()
        finally:
            complete.set()
        assert sample.result(timeout=2).native_target.committed_ends == (("0", 3),)
        cancellation.result(timeout=2)
    assert not binding.backend.leases


def test_active_result_blocks_next_execute_and_repeated_sampling(binding):
    runner = ready(binding)
    work = schedule(runner)
    runner.execute_model(work)
    with pytest.raises(RuntimeError, match="still active"):
        runner.execute_model(work)
    runner.sample_tokens(None)
    with pytest.raises(RuntimeError, match="no native result"):
        runner.sample_tokens(None)


def test_empty_step_updates_sampler_metadata_without_native_execute(
    binding, output_type
):
    runner = ready(binding)
    work = schedule(runner, lanes=0)
    output = runner.execute_model(work)
    assert isinstance(output, output_type)
    assert output.native_target.committed_ends == ()
    assert "sampling_requests" in binding.events
    assert not any(isinstance(e, tuple) and e[0] == "execute" for e in binding.events)


@pytest.mark.parametrize("wrong", ["owner", "credits", "mutable"])
def test_bad_bank_receipt_prevents_scheduler_admission(binding, wrong):
    runner = ready(binding)
    info = runner.cache_info()
    if wrong == "owner":
        info = replace(info, owner=0)
    elif wrong == "credits":
        info = replace(info, source_pages_free=(5, 4, 4, 4))
    else:
        info = replace(info, source_pages_free=[4] * 4)
    executor = SimpleNamespace(
        collective_rpc=lambda name: [info],
        shutdown=runner.shutdown,
    )
    with pytest.raises(RuntimeError, match="receipt"):
        native.initialize_native_scheduler(binding, config(), executor)
    assert "scheduler_init" not in binding.events
    assert "backend_close" in binding.events


def test_cancel_drains_logits_before_native_discard_and_free(binding):
    runner = ready(binding)
    work = schedule(runner)
    runner.execute_model(work)
    with pytest.raises(RuntimeError, match="active"):
        runner.release(work.native_target.grants[0].request)
    runner.cancel_step()
    runner.release(work.native_target.grants[0].request)
    assert binding.events.index(("release_logits", 0)) < binding.events.index(
        ("cancel", 0)
    )
    assert binding.events.index(("cancel", 0)) < binding.events.index(("free", 0))


@pytest.mark.parametrize("change", ["owner", "extent", "phase", "rows", "sequence"])
def test_invalid_grant_never_reaches_native_submit(binding, change):
    runner = ready(binding)
    work = schedule(runner)
    grant = work.native_target.grants[0]
    if change == "owner":
        work.native_target = replace(work.native_target, owner=18)
    elif change == "sequence":
        work.native_target = replace(work.native_target, step_id=0)
    else:
        grant = replace(
            grant,
            **{
                "extent": {"committed_end": 1},
                "phase": {"phase": "encoder"},
                "rows": {"tokens": tuple(range(257))},
            }[change],
        )
        work.native_target = replace(work.native_target, grants=(grant,))
    with pytest.raises((RuntimeError, ValueError)):
        runner.execute_model(work)
    assert not any(isinstance(e, tuple) and e[0] == "submit" for e in binding.events)


@pytest.mark.parametrize("accepted", [0, 1])
def test_accepted_publication_uses_native_extent_and_commit_failure_poison(
    binding, accepted
):
    runner = ready(binding)
    work = schedule(runner)
    binding.accepted = (accepted,)
    runner.execute_model(work)
    output = runner.sample_tokens(None)
    assert output.native_target.committed_ends == (("0", accepted),)
    grant = replace(work.native_target.grants[0], committed_end=accepted)
    work.native_target = native.NativeSchedule(17, 2, (grant,))
    runner.execute_model(work)
    binding.backend.fail_commit = True
    with pytest.raises(RuntimeError, match="publication"):
        runner.sample_tokens(None)
    with pytest.raises(RuntimeError, match="ready"):
        runner.execute_model(work)
    runner.shutdown()


def method(path, class_name, method_name, namespace):
    """Execute the actual entry method with CPU dependencies; no body rewrite."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def test_worker_entry_hooks_bypass_ordinary_gpu_constructor_and_allocator(binding):
    namespace = {
        "get_native_target_binding": native.get_native_target_binding,
        "NativeTargetRunner": native.NativeTargetRunner,
        "torch": SimpleNamespace(device=lambda name: name),
    }
    worker = SimpleNamespace(vllm_config=config(), local_rank=0)
    init = method("vllm/v1/worker/gpu_worker.py", "Worker", "init_device", namespace)
    load = method("vllm/v1/worker/gpu_worker.py", "Worker", "load_model", namespace)
    init(worker)
    load(worker)
    assert isinstance(worker.model_runner, native.NativeTargetRunner)
    assert binding.events.count("weights_load") == 1
    # No distributed/workspace/PyTorch-model globals were supplied above:
    # reaching any ordinary initializer would fail this test.
    for name in (
        "get_kv_cache_spec",
        "determine_available_memory",
        "initialize_from_config",
    ):
        call = method("vllm/v1/worker/gpu_worker.py", "Worker", name, namespace)
        with pytest.raises(RuntimeError, match="native"):
            call(worker, *([None] if name == "initialize_from_config" else []))


def test_engine_native_init_uses_bank_receipt_without_ordinary_block_pool(binding):
    runner = ready(binding)
    executor = SimpleNamespace(
        collective_rpc=lambda name: [runner.cache_info()],
        shutdown=runner.shutdown,
    )
    engine = SimpleNamespace(
        vllm_config=config(), model_executor=executor, log_stats=False
    )
    namespace = {
        "get_native_target_binding": native.get_native_target_binding,
        "initialize_native_scheduler": native.initialize_native_scheduler,
        "StructuredOutputManager": lambda cfg: object(),
    }
    initialize = method(
        "vllm/v1/engine/core.py",
        "EngineCore",
        "_initialize_cache_and_scheduler",
        namespace,
    )
    assert initialize(engine, config(), False) == 0
    assert binding.events[-1] == "scheduler_init"
    assert engine.scheduler.connector is None
    binding.fail_scheduler = True
    with pytest.raises(RuntimeError, match="scheduler binding"):
        initialize(engine, config(), False)
    assert "backend_close" in binding.events


def test_engine_missing_binding_rejects_before_executor(monkeypatch):
    monkeypatch.setattr(native, "_binding", None)
    plugins = ModuleType("vllm.plugins")
    plugins.load_general_plugins = lambda: None
    monkeypatch.setitem(sys.modules, plugins.__name__, plugins)
    cfg = config()
    cfg.parallel_config.data_parallel_rank_local = 1
    constructed = []
    initialize = method(
        "vllm/v1/engine/core.py",
        "EngineCore",
        "__init__",
        {"get_native_target_binding": native.get_native_target_binding},
    )
    with pytest.raises(native.NativeTargetUnavailable, match="binding"):
        initialize(SimpleNamespace(), cfg, lambda cfg: constructed.append(cfg), False)
    assert not constructed


def test_engine_structured_initialization_failure_closes_native_bank(binding):
    runner = ready(binding)
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(shutdown=runner.shutdown),
        log_stats=False,
    )

    def fail(config):
        raise RuntimeError("structured initialization failed")

    initialize = method(
        "vllm/v1/engine/core.py",
        "EngineCore",
        "_initialize_cache_and_scheduler",
        {
            "get_native_target_binding": native.get_native_target_binding,
            "StructuredOutputManager": fail,
        },
    )
    with pytest.raises(RuntimeError, match="structured initialization"):
        initialize(engine, config(), False)
    assert "backend_close" in binding.events


def test_engine_default_retains_ordinary_initialization_order(binding):
    events = []
    cfg = config(False)
    cfg.scheduler_config.get_scheduler_cls = lambda: (
        lambda **kwargs: events.append("scheduler")
    )
    cfg.scheduler_config.enable_chunked_prefill = False
    executor = SimpleNamespace(b12x_warmup_control=nullcontext)
    engine = SimpleNamespace(model_executor=executor, log_stats=False)
    engine._initialize_kv_caches = lambda cfg: (
        events.append("ordinary_cache") or SimpleNamespace(kv_cache_groups=[object()])
    )
    namespace = {
        "get_native_target_binding": native.get_native_target_binding,
        "StructuredOutputManager": lambda cfg: events.append("structured"),
        "resolve_kv_cache_block_sizes": lambda *args: (256, 256),
    }
    initialize = method(
        "vllm/v1/engine/core.py",
        "EngineCore",
        "_initialize_cache_and_scheduler",
        namespace,
    )
    assert initialize(engine, cfg, False) == 256
    assert events == ["ordinary_cache", "structured", "scheduler"]


def test_native_admission_validation_uses_request_error_boundary():
    req = SimpleNamespace(use_structured_output=True)

    def reject(request):
        assert request is req
        raise ValueError("structured native request unsupported")

    engine = SimpleNamespace(
        mm_receiver_cache=None,
        request_block_hasher=None,
        scheduler=SimpleNamespace(validate_request=reject),
    )
    preprocess = method(
        "vllm/v1/engine/core.py",
        "EngineCore",
        "preprocess_add_request",
        {"Request": SimpleNamespace(from_engine_core_request=lambda *a: req)},
    )
    with pytest.raises(ValueError, match="unsupported"):
        preprocess(engine, SimpleNamespace(mm_features=None, current_wave=0))


def stream_ready(binding):
    cfg = config()
    cfg.additional_config["afd_native_target"].update(
        prefill_mode="encoder_stream", batch_tokens=256
    )
    cfg.model_config.max_model_len = 1024
    cfg.scheduler_config.max_num_batched_tokens = 256
    runner = native.NativeTargetRunner(cfg, "cuda:0", binding, owner=17)
    runner.load_model()
    return runner


def stream_work(runner, rows=370):
    work = schedule(runner, rows=rows)
    grant = replace(work.native_target.grants[0], phase="encoder_stream", chunk_rows=80)
    work.native_target = replace(work.native_target, grants=(grant,))
    return work


def test_stream_whole_prompt_grant_exceeds_chunk_capacity_and_commits_once(binding):
    runner = stream_ready(binding)
    work = stream_work(runner)
    runner.execute_model(work)
    assert binding.backend.ends[work.native_target.grants[0].request] == 0
    output = runner.sample_tokens(None)
    assert output.native_target.committed_ends == (("0", 370),)
    assert binding.events.count(("commit", 0, 370)) == 1
    runner.shutdown()


@pytest.mark.parametrize(
    "change",
    [
        {"chunk_rows": 79},
        {"chunk_rows": 257},
        {"committed_end": 1},
        {"selected": (0,)},
        {"lane": 1},
        {"kind": "decode"},
    ],
)
def test_stream_invalid_geometry_rejects_before_native_mutation(binding, change):
    runner = stream_ready(binding)
    work = stream_work(runner)
    work.native_target = replace(
        work.native_target, grants=(replace(work.native_target.grants[0], **change),)
    )
    with pytest.raises(ValueError):
        runner.execute_model(work)
    assert not binding.backend.active
    runner.shutdown()


def test_stream_cancel_revokes_admission_then_reentry_uses_new_generation(binding):
    runner = stream_ready(binding)
    work = stream_work(runner)
    old = work.native_target.grants[0].request
    runner.execute_model(work)
    runner.cancel_step()
    assert old not in binding.backend.ends
    runner.release(old)  # consumes revocation receipt; no second native release
    assert ("free", 0) not in binding.events
    next_work = schedule(runner, step=2)
    assert next_work.native_target.grants[0].request.generation == 2
    runner.execute_model(next_work)
    assert runner.sample_tokens(None).native_target.committed_ends == (("0", 3),)
    runner.shutdown()


def test_stream_rejects_partial_acceptance_and_revokes_without_publication(binding):
    runner = stream_ready(binding)
    work = stream_work(runner)
    runner.execute_model(work)
    binding.accepted = (1,)
    with pytest.raises(RuntimeError, match="sampler"):
        runner.sample_tokens(None)
    assert not binding.backend.ends
    assert not any(
        isinstance(event, tuple) and event[0] == "commit" for event in binding.events
    )
    with pytest.raises(RuntimeError, match="ready"):
        runner.execute_model(work)
    runner.shutdown()


def test_stream_cannot_share_step_or_enter_when_opt_in_is_absent(binding):
    runner = stream_ready(binding)
    work = schedule(runner, lanes=2)
    grants = list(work.native_target.grants)
    grants[0] = replace(grants[0], phase="encoder_stream", chunk_rows=80)
    work.native_target = replace(work.native_target, grants=tuple(grants))
    with pytest.raises(ValueError):
        runner.execute_model(work)
    work.native_target = replace(work.native_target, grants=(grants[0],))
    runner.config.additional_config["afd_native_target"]["prefill_mode"] = "full_target"
    with pytest.raises(ValueError):
        runner.execute_model(work)
    assert not binding.backend.active
    runner.shutdown()


def test_stream_execution_failure_revokes_and_requires_a_new_owner_scope(binding):
    runner = stream_ready(binding)
    work = stream_work(runner)
    binding.backend.fail_execute = True
    with pytest.raises(RuntimeError, match="rank failed"):
        runner.execute_model(work)
    assert not binding.backend.ends
    with pytest.raises(RuntimeError, match="not ready"):
        runner.admit(0, 2)
    runner.shutdown()


def test_stream_consumer_timeout_retains_both_contexts_until_drain(binding):
    runner = stream_ready(binding)
    work = stream_work(runner)
    runner.execute_model(work)
    binding.backend.fail_drain = True
    with pytest.raises(RuntimeError, match="consumer"):
        runner.cancel_step()
    assert binding.backend.active
    assert binding.backend.ends
    with pytest.raises(RuntimeError, match="still active"):
        runner.execute_model(work)
    binding.backend.fail_drain = False
    runner.cancel_step()
    assert not binding.backend.active
    assert not binding.backend.ends
    runner.shutdown()
