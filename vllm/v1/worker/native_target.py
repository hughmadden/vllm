# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in retained TargetPass integration; no model or physical KV allocator.

These Python descriptors are binding-local, not the Rust schema-1 protocol.
The installed binding must attach actual NativeBank/TargetContext operations.
It must never create the independent CacheCommands metadata prototype.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput


class NativeTargetUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeRequest:
    owner: int
    slot: int
    generation: int


@dataclass(frozen=True)
class NativeCacheInfo:
    owner: int
    capacity_rows: int
    source_page_capacity: tuple[int, int, int, int]
    source_pages_free: tuple[int, int, int, int]
    cache_bytes: int

    def validate(self, owner: int | None = None) -> None:
        if (
            type(self.owner) is not int
            or not 0 < self.owner < 2**64
            or (owner is not None and self.owner != owner)
            or not 1 <= self.capacity_rows <= 4096
            or self.cache_bytes <= 0
            or not isinstance(self.source_page_capacity, tuple)
            or not isinstance(self.source_pages_free, tuple)
            or len(self.source_page_capacity) != 4
            or len(self.source_pages_free) != 4
            or any(
                type(free) is not int
                or type(capacity) is not int
                or not 0 <= free <= capacity
                for free, capacity in zip(
                    self.source_pages_free, self.source_page_capacity
                )
            )
        ):
            raise RuntimeError("invalid native cache receipt")


@dataclass(frozen=True)
class NativeGrant:
    request_id: str
    request: NativeRequest
    lane: int
    committed_end: int
    tokens: tuple[int, ...]
    selected: tuple[int, ...]
    kind: str
    placement: int
    phase: str = "full_target"


@dataclass(frozen=True)
class NativeSchedule:
    owner: int
    step_id: int
    grants: tuple[NativeGrant, ...]


@dataclass(frozen=True)
class NativeStepResult:
    owner: int
    step_id: int
    committed_ends: tuple[tuple[str, int], ...]
    cache_info: NativeCacheInfo


@dataclass(frozen=True)
class NativeSample:
    output: ModelRunnerOutput
    accepted: tuple[int, ...]


class NativeLogitsLease(Protocol):
    """Binding-owned FP32 view; consumers register their completion on this lease.

    A sampler exception must not lose already-enqueued consumers. drain_consumers
    waits for all their events, including on the exception path. close releases
    the Rust borrow only after successful drain; a timeout retains ownership.
    """

    def drain_consumers(self) -> None: ...
    def close(self) -> None: ...


class NativeTargetBackend(Protocol):
    def info(self) -> NativeCacheInfo: ...
    def admit(self, slot: int, request_id: int) -> NativeRequest: ...
    def release(self, request: NativeRequest) -> None: ...
    def committed_end(self, request: NativeRequest) -> int: ...
    def submit(self, grant: NativeGrant) -> Any:
        """Own copied inputs and retain unresolved native commands on reply loss.

        If a ticket cannot be returned, backend close must still drain its work.
        """
        ...

    def execute(self, tickets: tuple[Any, ...]) -> None:
        """Drive the submitted native futures together to bounded completion."""
        ...

    def acquire_logits(self, ticket: Any) -> NativeLogitsLease: ...
    def commit(self, ticket: Any, accepted: int) -> int: ...
    def cancel(self, ticket: Any) -> None:
        """Drain native work before discarding this ticket's private state."""
        ...

    def close(self) -> None: ...


class NativeSampler(Protocol):
    def update_requests(self, output: SchedulerOutput) -> None:
        """Reuse vLLM request sampling parameters/history, including retirements."""
        ...

    def sample(
        self,
        grants: tuple[NativeGrant, ...],
        leases: tuple[NativeLogitsLease, ...],
        grammar_output: GrammarOutput | None,
    ) -> NativeSample: ...
    def close(self) -> None: ...


class NativeTargetBinding(Protocol):
    def validate_config(self, config: VllmConfig) -> None:
        """Validate checkpoint, device, explicit native budget and supported phases.

        Must also validate scheduler/sampler support before allocating anything.
        Prefix/encoder/replay/speculation support cannot be inferred from having
        callable full-target kernels or source-prefix references.
        """
        ...

    def create_sampler(self, config: VllmConfig, device: Any) -> NativeSampler: ...
    def create_backend(
        self,
        config: VllmConfig,
        device: Any,
        owner: int,
    ) -> NativeTargetBackend:
        """Load weights, plan memory and create ONE actual native bank atomically.

        Must clean up partial construction before raising. Sampler allocations
        already exist, so the native planner sees their device memory usage.
        """
        ...

    def create_scheduler(self, **kwargs: Any) -> Any:
        """Return SchedulerInterface backed by native RPC credits, no BlockPool."""
        ...


_binding: NativeTargetBinding | None = None


def register_native_target(binding: NativeTargetBinding) -> None:
    """Register from a vLLM general plugin in both engine and worker processes."""
    global _binding
    if _binding is not None and _binding is not binding:
        raise RuntimeError("native target binding already registered")
    _binding = binding


def get_native_target_binding(config: VllmConfig) -> NativeTargetBinding | None:
    additional = config.additional_config
    options = (
        additional.get("afd_native_target", {}) if isinstance(additional, dict) else {}
    )
    if not isinstance(options, dict) or type(options.get("enabled", False)) is not bool:
        raise ValueError("afd_native_target.enabled must be a boolean")
    if not options.get("enabled", False):
        return None
    parallel = config.parallel_config
    if (
        not config.use_v2_model_runner
        or parallel.world_size != 1
        or parallel.data_parallel_size != 1
        or parallel.distributed_executor_backend != "uni"
        or parallel.enable_dbo
        or parallel.enable_fault_tolerance
        or config.scheduler_config.async_scheduling
        or config.model_config.runner_type != "generate"
        or config.model_config.enable_sleep_mode
        or config.cache_config.enable_prefix_caching
        or any(
            getattr(config, field) is not None
            for field in (
                "speculative_config",
                "lora_config",
                "kv_transfer_config",
                "ec_transfer_config",
                "weight_transfer_config",
            )
        )
    ):
        raise NativeTargetUnavailable(
            "native target currently requires synchronous V2 generation with one "
            "uni worker; prefixes, speculation, connectors, LoRA, sleep and parallel "
            "workers are not bound"
        )
    if _binding is None:
        raise NativeTargetUnavailable(
            "native target selected but no target/CUDA-lease/sampler/scheduler "
            "binding is registered; ordinary weights and KV must not be loaded"
        )
    _binding.validate_config(config)
    return _binding


def initialize_native_scheduler(binding, config, executor, **kwargs):
    """Read actual worker capacity before allowing scheduler admission."""
    try:
        receipts = executor.collective_rpc("native_cache_info")
        if len(receipts) != 1 or not isinstance(receipts[0], NativeCacheInfo):
            raise RuntimeError("native target requires one authoritative cache bank")
        receipts[0].validate()
        return binding.create_scheduler(
            vllm_config=config, executor=executor, cache_info=receipts[0], **kwargs
        )
    except BaseException:
        executor.shutdown()
        raise


@dataclass
class _Active:
    grant: NativeGrant
    ticket: Any
    logits: NativeLogitsLease | None = None


class NativeTargetRunner:
    """Synchronous V2 worker surface; native futures may use both target lanes.

    The RLock serializes worker calls; cancellation cannot race a sampler into
    freeing its logits. In-flight cancellation belongs in the binding's bounded
    owner-thread command loop, not a second Python call mutating the same scope.
    """

    def __init__(self, config, device, binding, *, owner=None):
        self.config, self.device, self.binding = config, device, binding
        self.owner = owner if owner is not None else (secrets.randbits(64) or 1)
        if not 0 < self.owner < 2**64:
            raise ValueError("invalid native owner nonce")
        self._backend: NativeTargetBackend | None = None
        self._sampler: NativeSampler | None = None
        self._active: list[_Active] = []
        self._last_step = 0
        self._loaded = self._failed = self._closed = False
        self._lock = RLock()

    def _ready(self) -> NativeTargetBackend:
        if not self._loaded or self._failed or self._closed or self._backend is None:
            raise RuntimeError("native target is not ready")
        return self._backend

    def load_model(self, *, load_dummy_weights=False):
        with self._lock:
            if self._loaded or self._failed or self._closed:
                raise RuntimeError("native target already loaded or closed")
            if load_dummy_weights:
                raise NativeTargetUnavailable("native target requires actual weights")
            try:
                self._sampler = self.binding.create_sampler(self.config, self.device)
                self._backend = self.binding.create_backend(
                    self.config, self.device, self.owner
                )
                self._backend.info().validate(self.owner)
                self._loaded = True
            except BaseException:
                self._failed = True
                self.shutdown()
                raise

    def cache_info(self) -> NativeCacheInfo:
        with self._lock:
            info = self._ready().info()
            info.validate(self.owner)
            return info

    def admit(self, slot: int, request_id: int) -> NativeRequest:
        with self._lock:
            request = self._ready().admit(slot, request_id)
            if (
                request.owner != self.owner
                or request.slot != slot
                or request.generation < 1
            ):
                self._failed = True
                raise RuntimeError("invalid native admission handle")
            return request

    def release(self, request: NativeRequest) -> None:
        with self._lock:
            backend = self._ready()
            if any(entry.grant.request == request for entry in self._active):
                raise RuntimeError(
                    "native request still active; cancel or sample first"
                )
            backend.release(request)

    def _validate_schedule(self, output: SchedulerOutput) -> NativeSchedule:
        backend = self._ready()
        work = output.native_target
        if (
            not isinstance(work, NativeSchedule)
            or work.owner != self.owner
            or work.step_id != self._last_step + 1
            or not isinstance(work.grants, tuple)
            or not 0 <= len(work.grants) <= 2
        ):
            raise ValueError("invalid native owner, sequence or grants")
        if output.scheduled_spec_decode_tokens or output.scheduled_encoder_inputs:
            raise NativeTargetUnavailable(
                "native speculation and image work are unbound"
            )
        info = self.cache_info()
        lanes, requests, names = set(), set(), set()
        for grant in work.grants:
            if (
                not isinstance(grant, NativeGrant)
                or grant.phase != "full_target"
                or grant.kind not in ("prefill", "decode")
                or not isinstance(grant.request, NativeRequest)
                or grant.request.owner != self.owner
                or grant.lane not in (0, 1)
                or grant.lane in lanes
                or grant.request in requests
                or grant.request_id in names
                or not isinstance(grant.tokens, tuple)
                or not 1 <= len(grant.tokens) <= info.capacity_rows
                or any(type(t) is not int or not 0 <= t < 2**32 for t in grant.tokens)
                or not isinstance(grant.selected, tuple)
                or not 1 <= len(grant.selected) <= 48
                or any(
                    type(i) is not int or not 0 <= i < len(grant.tokens)
                    for i in grant.selected
                )
                or tuple(sorted(set(grant.selected))) != grant.selected
                or not 0 <= grant.placement < 2**64
                or grant.request_id in output.finished_req_ids
                or grant.request_id in (output.preempted_req_ids or ())
                or grant.committed_end != backend.committed_end(grant.request)
            ):
                raise ValueError("invalid or unsupported native target grant")
            lanes.add(grant.lane)
            requests.add(grant.request)
            names.add(grant.request_id)
        counts = {g.request_id: len(g.tokens) for g in work.grants}
        if (
            output.num_scheduled_tokens != counts
            or output.total_num_scheduled_tokens != sum(counts.values())
        ):
            raise ValueError("native tokens exceed or differ from scheduler grant")
        return work

    def execute_model(self, scheduler_output: SchedulerOutput):
        with self._lock:
            if self._active:
                raise RuntimeError("native result still active; sample or cancel first")
            work = self._validate_schedule(scheduler_output)
            backend = self._ready()
            self._last_step = work.step_id
            try:
                assert self._sampler is not None
                self._sampler.update_requests(scheduler_output)
                if not work.grants:
                    from vllm.v1.outputs import ModelRunnerOutput

                    return ModelRunnerOutput(
                        [],
                        {},
                        native_target=NativeStepResult(
                            self.owner, self._last_step, (), self.cache_info()
                        ),
                    )
                for grant in work.grants:
                    self._active.append(_Active(grant, backend.submit(grant)))
                backend.execute(tuple(entry.ticket for entry in self._active))
                for entry in self._active:
                    entry.logits = backend.acquire_logits(entry.ticket)
            except BaseException:
                self._failed = True
                self.cancel_step()
                raise
            return None

    def _drain_logits(self):
        for entry in self._active:
            if entry.logits is not None:
                entry.logits.drain_consumers()
                entry.logits.close()
                entry.logits = None

    def sample_tokens(self, grammar_output: GrammarOutput | None):
        from vllm.v1.outputs import ModelRunnerOutput

        with self._lock:
            backend = self._ready()
            if not self._active:
                raise RuntimeError("no native result to sample")
            assert self._sampler is not None
            try:
                grants = tuple(entry.grant for entry in self._active)
                leases = tuple(entry.logits for entry in self._active)
                assert all(lease is not None for lease in leases)
                sample = self._sampler.sample(grants, leases, grammar_output)
                if (
                    not isinstance(sample.output, ModelRunnerOutput)
                    or sample.output.req_ids != [g.request_id for g in grants]
                    or sample.output.req_id_to_index
                    != {g.request_id: i for i, g in enumerate(grants)}
                    or len(sample.accepted) != len(grants)
                    or any(
                        type(n) is not int or not 0 <= n <= len(g.tokens)
                        for n, g in zip(sample.accepted, grants)
                    )
                ):
                    raise RuntimeError("invalid native sampler result")
                self._drain_logits()
                committed = []
                for accepted in sample.accepted:
                    entry = self._active[0]
                    end = backend.commit(entry.ticket, accepted)
                    self._active.pop(0)
                    if end != entry.grant.committed_end + accepted:
                        raise RuntimeError("native committed extent mismatch")
                    committed.append((entry.grant.request_id, end))
                sample.output.native_target = NativeStepResult(
                    self.owner, self._last_step, tuple(committed), self.cache_info()
                )
                return sample.output
            except BaseException:
                self._failed = True
                self.cancel_step()
                raise

    def cancel_step(self) -> None:
        """Cancel this whole worker step; scheduler may regrant unaffected work."""
        with self._lock:
            self._drain_logits()
            while self._active:
                assert self._backend is not None
                self._backend.cancel(self._active[0].ticket)
                self._active.pop(0)

    def shutdown(self):
        with self._lock:
            if self._closed:
                return
            self.cancel_step()
            if self._backend is not None:
                self._backend.close()
                self._backend = None
            if self._sampler is not None:
                self._sampler.close()
                self._sampler = None
            self._closed = True

    def get_supported_tasks(self):
        return ("generate",)

    def get_model(self):
        raise NativeTargetUnavailable("native target has no PyTorch backbone")

    def get_encoder_timing_stats(self):
        return {}
