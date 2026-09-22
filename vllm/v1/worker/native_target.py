# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in retained TargetPass integration; no model or physical KV allocator.

These Python descriptors are binding-local, not the Rust schema-1 protocol.
The installed binding must attach actual NativeBank/TargetContext operations.
It must never create the independent CacheCommands metadata prototype.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
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
    chunk_rows: int | None = None
    verification_rows: int = 0

    @property
    def scheduled_rows(self) -> int:
        return self.verification_rows if self.phase == "dspark" else len(self.tokens)


@dataclass(frozen=True)
class NativeProposal:
    ticket: Any
    tokens: tuple[int, ...]
    draft_us: int


@dataclass(frozen=True)
class NativeBatchGrant:
    lane: int
    placement: int
    members: tuple[NativeGrant, ...]


@dataclass(frozen=True)
class NativeBatchMember:
    request: NativeRequest
    tokens: tuple[int, ...]
    selected: tuple[int, ...]
    input_offset: int
    output_offset: int


@dataclass(frozen=True)
class NativeBatchProposal:
    ticket: Any
    members: tuple[NativeBatchMember, ...]
    draft_us: int


@dataclass(frozen=True)
class NativeSchedule:
    owner: int
    step_id: int
    grants: tuple[NativeGrant, ...]
    batching: bool = False


@dataclass(frozen=True)
class NativeStepResult:
    owner: int
    step_id: int
    committed_ends: tuple[tuple[str, int], ...]
    cache_info: NativeCacheInfo
    speculative_counts: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class NativeCompositeRecovery:
    """Binding-validated cleanup decision, never an inferred bank frontier."""
    consumer_complete: bool
    failure: BaseException | None = None


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
    def can_prepare(self, work: tuple[tuple[NativeRequest, int], ...]) -> bool: ...
    def submit(self, grant: NativeGrant) -> Any:
        """Own copied inputs and retain unresolved native commands on reply loss.

        If a ticket cannot be returned, backend close must still drain its work.
        """
        ...

    def propose(self, grant: NativeGrant) -> NativeProposal:
        """Return the native-owned anchor/drafts within the reserved envelope."""
        ...

    def submit_batch(self, grant: NativeBatchGrant) -> Any: ...
    def propose_batch(self, grant: NativeBatchGrant) -> NativeBatchProposal: ...
    def commit_batch(
        self, ticket: Any, accepted: tuple[int, ...]
    ) -> tuple[int, ...]: ...

    def execute(self, tickets: tuple[Any, ...]) -> None:
        """Drive the submitted native futures together to bounded completion."""
        ...

    def acquire_logits(self, ticket: Any) -> NativeLogitsLease: ...
    def commit(self, ticket: Any, accepted: int) -> int: ...
    def cancel(self, ticket: Any) -> bool:
        """Drain work; return whether native revoked the entire admission."""
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
            "uni worker; prefixes, ordinary draft models, connectors, LoRA, sleep "
            "and parallel workers are not bound"
        )
    if _binding is None and options.get("implementation") == "retained":
        from vllm.v1.worker.native_target_binding import RetainedNativeBinding

        register_native_target(RetainedNativeBinding())
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
    members: tuple[NativeGrant, ...] | None = None

    @property
    def grants(self):
        return self.members if self.members is not None else (self.grant,)


@dataclass(frozen=True)
class _LogitsSlice:
    parent: NativeLogitsLease
    start: int
    end: int
    total: int

    @property
    def tensor(self):
        tensor = self.parent.tensor
        if tensor.ndim != 2 or tensor.shape[0] != self.total:
            raise RuntimeError("native batch logits extent differs")
        return tensor[self.start : self.end]


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
        self._request_order: tuple[str, ...] = ()
        self._revoked: dict[int, NativeRequest] = {}
        # p10 (p5-rpc-trim reuse): last frontier the daemon itself confirmed
        # per live request: seeded by admit (zero), advanced only by
        # already-validated commit replies, removed on revocation or release.
        # committed_end is a pure daemon read and no other command mutates a
        # committed frontier, so this mirror answers schedule validation
        # without a per-grant committed_end RPC; capacity_rows is owner-static
        # (daemon info() clones it), cached at load instead of a per-step
        # info command.
        self._frontiers: dict[NativeRequest, int] = {}
        self._capacity_rows = 0
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
                # C ABI creation returns an owned thread immediately; retain it
                # before waiting for its one combined load/plan/cache-init ACK.
                if initialize := getattr(self._backend, "initialize", None):
                    initialize()
                info = self._backend.info()
                info.validate(self.owner)
                # p10: capacity_rows is owner-static; the schedule-validation
                # capacity checks read this cached value (p5 trim).
                self._capacity_rows = info.capacity_rows
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
            self._revoked.pop(slot, None)
            self._frontiers[request] = 0
            return request

    def release(self, request: NativeRequest) -> None:
        with self._lock:
            backend = self._ready()
            if any(
                g.request == request for entry in self._active for g in entry.grants
            ):
                raise RuntimeError(
                    "native request still active; cancel or sample first"
                )
            if self._revoked.get(request.slot) == request:
                # Streaming cancellation already released the actual bank.
                self._revoked.pop(request.slot)
                self._frontiers.pop(request, None)
                return
            backend.release(request)
            self._frontiers.pop(request, None)

    def can_prepare(self, work):
        with self._lock:
            if self._active:
                raise RuntimeError("native capacity query requires idle target lanes")
            return self._ready().can_prepare(work)

    def _validate_schedule(self, output: SchedulerOutput) -> NativeSchedule:
        backend = self._ready()
        work = output.native_target
        if (
            not isinstance(work, NativeSchedule)
            or work.owner != self.owner
            or work.step_id != self._last_step + 1
            or not isinstance(work.grants, tuple)
            or type(work.batching) is not bool
            or not 0 <= len(work.grants) <= (16 if work.batching else 2)
        ):
            raise ValueError("invalid native owner, sequence or grants")
        if output.scheduled_spec_decode_tokens or output.scheduled_encoder_inputs:
            raise NativeTargetUnavailable(
                "ordinary scheduler draft tokens and image work are unbound"
            )
        # p10 (p5 trim): read-only capacity/floor checks run against
        # owner-static capacity and the commit-confirmed frontier mirror: no
        # info or per-grant committed_end commands on the validated step path.
        capacity_rows = self._capacity_rows
        options = self.config.additional_config["afd_native_target"]
        if work.batching and options.get("decode_batching") is not True:
            raise ValueError("native decode batching is not enabled")
        lanes, requests, names = set(), set(), set()
        for grant in work.grants:
            streaming = (
                isinstance(grant, NativeGrant) and grant.phase == "encoder_stream"
            )
            speculative = isinstance(grant, NativeGrant) and grant.phase == "dspark"
            if (
                not isinstance(grant, NativeGrant)
                or grant.phase not in ("full_target", "encoder_stream", "dspark")
                or type(grant.verification_rows) is not int
                or (not speculative and grant.verification_rows != 0)
                or grant.kind not in ("prefill", "decode")
                or not isinstance(grant.request, NativeRequest)
                or grant.request.owner != self.owner
                or grant.lane not in (0, 1)
                or (not work.batching and grant.lane in lanes)
                or grant.request in requests
                or grant.request_id in names
                or not isinstance(grant.tokens, tuple)
                or not grant.tokens
                or (
                    not streaming
                    and (
                        len(grant.tokens) > capacity_rows
                        or grant.chunk_rows is not None
                    )
                )
                or any(type(t) is not int or not 0 <= t < 2**32 for t in grant.tokens)
                or not isinstance(grant.selected, tuple)
                or not 1 <= len(grant.selected) <= 48
                or any(
                    type(i) is not int or not 0 <= i < grant.scheduled_rows
                    for i in grant.selected
                )
                or tuple(sorted(set(grant.selected))) != grant.selected
                or not 0 <= grant.placement < 2**64
                or grant.request_id in output.finished_req_ids
                or grant.request_id in (output.preempted_req_ids or ())
                or grant.committed_end
                != self._known_end(backend, grant.request, grant.committed_end)
            ):
                raise ValueError("invalid or unsupported native target grant")
            if streaming and (
                options.get("prefill_mode", "full_target") != "encoder_stream"
                or len(work.grants) != 1
                or grant.lane != 0
                or grant.committed_end != 0
                or grant.kind != "prefill"
                or len(grant.tokens) > self.config.model_config.max_model_len
                or type(grant.chunk_rows) is not int
                or not 80
                <= grant.chunk_rows
                <= min(
                    capacity_rows,
                    options["batch_tokens"],
                    self.config.scheduler_config.max_num_batched_tokens,
                )
                or grant.selected[0] < max(0, len(grant.tokens) - 128)
            ):
                raise ValueError("invalid or nonexclusive native encoder stream")
            if speculative and (
                not (options.get("dspark") or {}).get("draft_limit", 0)
                or grant.kind != "decode"
                or len(grant.tokens) != 1
                or grant.committed_end <= 0
                or not 1
                <= grant.verification_rows
                <= min(
                    options["dspark"]["draft_limit"] + 1,
                    capacity_rows,
                    self.config.scheduler_config.max_num_batched_tokens,
                    self.config.model_config.max_model_len - grant.committed_end - 1,
                )
                or grant.selected != tuple(range(grant.verification_rows))
            ):
                raise ValueError("invalid native speculative verification envelope")
            if work.batching and (
                grant.kind != "decode"
                or streaming
                or grant.committed_end <= 0
                or (
                    not speculative
                    and (len(grant.tokens) != 1 or grant.selected != (0,))
                )
            ):
                raise ValueError("native batches require one decode anchor per request")
            lanes.add(grant.lane)
            requests.add(grant.request)
            names.add(grant.request_id)
        counts = {g.request_id: g.scheduled_rows for g in work.grants}
        if work.batching:
            for lane in lanes:
                members = [g for g in work.grants if g.lane == lane]
                if (
                    len(members) > 8
                    or len({g.phase for g in members}) != 1
                    or len({g.placement for g in members}) != 1
                    or sum(g.scheduled_rows for g in members)
                    > min(capacity_rows, options["batch_tokens"])
                    or sum(len(g.selected) for g in members) > 48
                ):
                    raise ValueError(
                        "native batch exceeds lane capacity or mixes phases"
                    )
            if (
                sum(counts.values())
                > self.config.scheduler_config.max_num_batched_tokens
            ):
                raise ValueError("native batch exceeds scheduler token budget")
        if (
            output.num_scheduled_tokens != counts
            or output.total_num_scheduled_tokens != sum(counts.values())
        ):
            raise ValueError("native tokens exceed or differ from scheduler grant")
        return work

    def _known_end(self, backend, request: NativeRequest, presented: int) -> int:
        # p10 (p5 trim): the commit-confirmed mirror answers for every
        # request whose presented end matches it (the production path, zero
        # RPCs: every presented end came from this runner's own published
        # receipt). A presented end the mirror does not hold (foreign
        # constructions that seeded daemon state out of band) is verified
        # once against the authoritative read and the mirror self-heals to
        # it; a true mismatch then fails this validation exactly like the
        # pinned per-grant RPC.
        known = self._frontiers.get(request)
        if known == presented:
            return known
        actual = backend.committed_end(request)
        self._frontiers[request] = actual
        return actual

    def _submit_batch(self, members):
        assert self._backend is not None
        first = members[0]
        group = NativeBatchGrant(first.lane, first.placement, members)
        if first.phase != "dspark":
            ticket = self._backend.submit_batch(group)
            self._active.append(_Active(first, ticket, members=members))
            return
        proposal = self._backend.propose_batch(group)
        if not isinstance(proposal, NativeBatchProposal):
            raise RuntimeError("invalid native batch proposal receipt")
        entry = _Active(first, proposal.ticket, members=members)
        self._active.append(entry)
        if (
            not isinstance(proposal.members, tuple)
            or len(proposal.members) != len(members)
            or type(proposal.draft_us) is not int
            or not 0 <= proposal.draft_us < 2**64
        ):
            raise RuntimeError("invalid native batch proposal manifest")
        resolved, input_offset, output_offset = [], 0, 0
        for grant, member in zip(members, proposal.members):
            if (
                not isinstance(member, NativeBatchMember)
                or member.request != grant.request
                or not isinstance(member.tokens, tuple)
                or not 1 <= len(member.tokens) <= grant.verification_rows
                or member.tokens[0] != grant.tokens[0]
                or any(type(t) is not int or not 0 <= t < 2**32 for t in member.tokens)
                or member.selected != tuple(range(len(member.tokens)))
                or type(member.input_offset) is not int
                or member.input_offset != input_offset
                or type(member.output_offset) is not int
                or member.output_offset != output_offset
            ):
                raise RuntimeError("invalid native batch proposal member or offsets")
            resolved.append(
                replace(grant, tokens=member.tokens, selected=member.selected)
            )
            input_offset += len(member.tokens)
            output_offset += len(member.selected)
        entry.members = tuple(resolved)

    def _submit_split_supported(self, backend):
        # p10: multi-lane steps overlap per-lane submits only when the
        # bridge exposes the non-blocking publish/await split for singleton
        # (submit/propose), grouped (submit_batch/propose_batch) entries and
        # the result-scope splits (acquire, commit, commit_batch).
        return all(
            callable(getattr(backend, name, None))
            for name in (
                "begin_submit",
                "finish_submit",
                "begin_propose",
                "finish_propose",
                "begin_submit_batch",
                "finish_submit_batch",
                "begin_propose_batch",
                "finish_propose_batch",
                "begin_acquire_logits",
                "finish_acquire_logits",
                "begin_commit",
                "finish_commit",
                "begin_commit_batch",
                "finish_commit_batch",
            )
        )

    def _begin_lane_submits(self, backend, work):
        # p10: SEND every lane's submit (speculative or not, grouped or
        # singleton) before awaiting any prepared ACK. The daemon admits one
        # pending command per lane slot (ffi/actor.rs reserve()), so both
        # lanes' draft proposals queue on their own lane actors instead of
        # the second lane waiting for the first lane's ACK. Per-entry
        # manifest validation still runs in _finish_lane_submit, in send
        # order, before execute_model proceeds to execute/acquire.
        pending = []
        try:
            begun = set()
            for grant in work.grants:
                if work.batching:
                    if grant.lane in begun:
                        continue
                    begun.add(grant.lane)
                    members = tuple(g for g in work.grants if g.lane == grant.lane)
                    first = members[0]
                    group = NativeBatchGrant(first.lane, first.placement, members)
                    if first.phase == "dspark":
                        command = backend.begin_propose_batch(group)
                        pending.append((grant, "batch_propose", members, command))
                    else:
                        command = backend.begin_submit_batch(group)
                        pending.append((grant, "batch_submit", members, command))
                    continue
                if grant.phase == "dspark":
                    pending.append(
                        (grant, "propose", None, backend.begin_propose(grant))
                    )
                else:
                    pending.append(
                        (grant, "submit", None, backend.begin_submit(grant))
                    )
        except BaseException:
            self._collect_lane_submits(backend, pending)
            raise
        return pending

    def _resilient_submit_finish(self, backend, retry_begin, finish, command):
        # p10 (F1 fix): a submission the daemon rejects with BUSY/NOT_READY
        # never reached lane processing (ffi/actor.rs reserve()/control()
        # reply without committing slot state; the encoder-stream scope is
        # the documented BUSY source, p9 F1). Re-publishing the identical
        # submission is therefore atomic. Retry transient rejections with a
        # bounded backoff instead of letting one reply poison the whole
        # worker (the MIXED crash class: one native error -> _failed=True ->
        # engine dead for every remaining request).
        from vllm_afd.native_target.client import (
            NativeTargetError,
            submit_rejection_retryable,
        )

        client = getattr(backend, "client", None)
        clock = getattr(client, "_clock", None) or __import__("time").monotonic
        sleep = getattr(client, "_sleep", None) or __import__("time").sleep
        timeout = getattr(client, "timeout_s", 60) or 60
        deadline = clock() + timeout
        while True:
            try:
                return finish(command)
            except NativeTargetError as error:
                if not submit_rejection_retryable(error) or clock() >= deadline:
                    raise
                sleep(min(0.001, max(0.0, deadline - clock())))
                command = retry_begin()

    def _finish_lane_submit(self, backend, grant, kind, members, command):
        # Await one lane's prepared ACK and run that entry's manifest
        # validation exactly where the serial loop ran it: after the entry
        # is appended to _active, so a validation failure is recovered by
        # the ordinary cancel_step path.
        if kind == "propose":
            proposal = self._resilient_submit_finish(
                backend,
                lambda: backend.begin_propose(grant),
                backend.finish_propose,
                command,
            )
            if not isinstance(proposal, NativeProposal):
                raise RuntimeError("invalid native proposal receipt")
            entry = _Active(grant, proposal.ticket)
            self._active.append(entry)
            if (
                not isinstance(proposal.tokens, tuple)
                or not 1 <= len(proposal.tokens) <= grant.verification_rows
                or proposal.tokens[0] != grant.tokens[0]
                or any(
                    type(t) is not int or not 0 <= t < 2**32 for t in proposal.tokens
                )
                or type(proposal.draft_us) is not int
                or not 0 <= proposal.draft_us < 2**64
            ):
                raise RuntimeError("native proposal exceeds owned envelope")
            entry.grant = replace(
                grant,
                tokens=proposal.tokens,
                selected=tuple(range(len(proposal.tokens))),
            )
            return entry
        if kind == "submit":
            entry = _Active(
                grant,
                self._resilient_submit_finish(
                    backend,
                    lambda: backend.begin_submit(grant),
                    backend.finish_submit,
                    command,
                ),
            )
            self._active.append(entry)
            return entry
        if kind == "batch_propose":
            group = NativeBatchGrant(members[0].lane, members[0].placement, members)
            proposal = self._resilient_submit_finish(
                backend,
                lambda: backend.begin_propose_batch(group),
                backend.finish_propose_batch,
                command,
            )
            if not isinstance(proposal, NativeBatchProposal):
                raise RuntimeError("invalid native batch proposal receipt")
            entry = _Active(members[0], proposal.ticket, members=members)
            self._active.append(entry)
            if (
                not isinstance(proposal.members, tuple)
                or len(proposal.members) != len(members)
                or type(proposal.draft_us) is not int
                or not 0 <= proposal.draft_us < 2**64
            ):
                raise RuntimeError("invalid native batch proposal manifest")
            resolved, input_offset, output_offset = [], 0, 0
            for member_grant, member in zip(members, proposal.members):
                if (
                    not isinstance(member, NativeBatchMember)
                    or member.request != member_grant.request
                    or not isinstance(member.tokens, tuple)
                    or not 1 <= len(member.tokens) <= member_grant.verification_rows
                    or member.tokens[0] != member_grant.tokens[0]
                    or any(type(t) is not int or not 0 <= t < 2**32 for t in member.tokens)
                    or member.selected != tuple(range(len(member.tokens)))
                    or type(member.input_offset) is not int
                    or member.input_offset != input_offset
                    or type(member.output_offset) is not int
                    or member.output_offset != output_offset
                ):
                    raise RuntimeError(
                        "invalid native batch proposal member or offsets"
                    )
                resolved.append(
                    replace(
                        member_grant, tokens=member.tokens, selected=member.selected
                    )
                )
                input_offset += len(member.tokens)
                output_offset += len(member.selected)
            entry.members = tuple(resolved)
            return entry
        group = NativeBatchGrant(members[0].lane, members[0].placement, members)
        ticket = self._resilient_submit_finish(
            backend,
            lambda: backend.begin_submit_batch(group),
            backend.finish_submit_batch,
            command,
        )
        entry = _Active(members[0], ticket, members=members)
        self._active.append(entry)
        return entry

    def _collect_lane_submits(self, backend, remaining):
        # Failure path: every already-SENT submit is collected so its ticket
        # lands in _active and the existing cancel/recover path cancels it
        # (daemon Cancel accepts prepared tickets, ffi/actor.rs reserve()).
        # A collect that itself fails keeps its original client command
        # pending for wait/cancel recovery — exactly the serial path's state
        # for a failed or timed-out submit. Secondary failures never mask the
        # primary error.
        for grant, kind, members, command in remaining:
            try:
                self._finish_lane_submit(backend, grant, kind, members, command)
            except BaseException:
                continue

    def _finish_lane_submits(self, backend, pending):
        for index, (grant, kind, members, command) in enumerate(pending):
            try:
                self._finish_lane_submit(backend, grant, kind, members, command)
            except BaseException:
                self._collect_lane_submits(backend, pending[index + 1 :])
                raise

    def _acquire_lane_logits(self, backend):
        # p10: publish every entry's logits acquisition before awaiting any
        # descriptor (per-slot pending gate, disjoint tickets); fall back to
        # the serial acquisition for split-less bridges.
        if len(self._active) > 1 and self._submit_split_supported(backend):
            published = [
                (entry, backend.begin_acquire_logits(entry.ticket))
                for entry in self._active
            ]
            for entry, command in published:
                entry.logits = backend.finish_acquire_logits(command)
            return
        for entry in self._active:
            entry.logits = backend.acquire_logits(entry.ticket)

    def execute_model(self, scheduler_output: SchedulerOutput):
        with self._lock:
            if self._active:
                raise RuntimeError("native result still active; sample or cancel first")
            work = self._validate_schedule(scheduler_output)
            backend = self._ready()
            self._last_step = work.step_id
            self._request_order = tuple(g.request_id for g in work.grants)
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
                if len(work.grants) > 1 and self._submit_split_supported(backend):
                    # p10: multi-lane step; every lane's submit is SENT
                    # before any prepared ACK is awaited, then awaited in
                    # send order (per-lane ordering and _active order
                    # unchanged). Transient daemon rejections of an in-flight
                    # submission (BUSY/NOT_READY, e.g. the encoder-stream
                    # scope) are retried atomically; the MIXED-cell crash
                    # class — one rejection -> _failed -> dead engine — is
                    # closed by _resilient_submit_finish.
                    self._finish_lane_submits(
                        backend, self._begin_lane_submits(backend, work)
                    )
                else:
                    for grant in work.grants:
                        if work.batching:
                            if any(
                                entry.grant.lane == grant.lane
                                for entry in self._active
                            ):
                                continue
                            self._submit_batch(
                                tuple(g for g in work.grants if g.lane == grant.lane)
                            )
                            continue
                        if grant.phase == "dspark":
                            proposal = backend.propose(grant)
                            if not isinstance(proposal, NativeProposal):
                                raise RuntimeError("invalid native proposal receipt")
                            entry = _Active(grant, proposal.ticket)
                            self._active.append(entry)
                            if (
                                not isinstance(proposal.tokens, tuple)
                                or not 1 <= len(proposal.tokens) <= grant.verification_rows
                                or proposal.tokens[0] != grant.tokens[0]
                                or any(
                                    type(t) is not int or not 0 <= t < 2**32
                                    for t in proposal.tokens
                                )
                                or type(proposal.draft_us) is not int
                                or not 0 <= proposal.draft_us < 2**64
                            ):
                                raise RuntimeError("native proposal exceeds owned envelope")
                            entry.grant = replace(
                                grant,
                                tokens=proposal.tokens,
                                selected=tuple(range(len(proposal.tokens))),
                            )
                        else:
                            self._active.append(_Active(grant, backend.submit(grant)))
                backend.execute(tuple(entry.ticket for entry in self._active))
                self._acquire_lane_logits(backend)
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

    def _drain_entries_pipelined(self):
        # p10: publish every entry's consumer drain (release_logits) before
        # awaiting any ACK; the daemon records the consumer fences
        # concurrently. Falls back per-step to the serial drain when a lease
        # does not expose the split (split-less bridges, fixture leases).
        entries = [entry for entry in self._active if entry.logits is not None]
        if not entries:
            return
        if not all(
            callable(getattr(entry.logits, "begin_drain", None))
            and callable(getattr(entry.logits, "finish_drain", None))
            for entry in entries
        ):
            self._drain_logits()
            return
        published = [
            (entry, entry.logits, entry.logits.begin_drain()) for entry in entries
        ]
        for entry, lease, command in published:
            lease.finish_drain(command)
            lease.close()
            entry.logits = None

    def _composite_eligible(self, backend):
        if len(self._active) != 1:
            return False
        entry = self._active[0]
        # The fused release_logits+commit route is daemon-gated to grouped
        # FULL_TARGET batches only: ffi/actor.rs result_scope composite arm
        # requires Acceptance::Batch (actor.rs:365) and rejects
        # SpeculativeBatch with a LeaseHeld error, so grouped dSpark stays
        # on the separate drain+commit path (daemon rule, mirrored by
        # composite.py's batch_full_target phase check).
        return (
            entry.members is not None
            and all(grant.phase == "full_target" for grant in entry.grants)
            and entry.logits is not None
            and callable(getattr(backend, "can_drain_commit_batch", None))
            and backend.can_drain_commit_batch(entry.ticket, entry.logits)
        )

    def _sample_composite(self, backend, sample):
        # Only reached after the original complete sampler-result validation.
        entry = self._active[0]
        accepted_by_request = dict(zip(self._request_order, sample.accepted))
        accepted = tuple(accepted_by_request[g.request_id] for g in entry.grants)
        ends = backend.drain_commit_batch(entry.ticket, entry.logits, accepted)
        # Successful return is the bridge's validated consumer+commit proof.
        entry.logits = None
        self._active.pop(0)
        if (
            not isinstance(ends, tuple)
            or len(ends) != len(entry.grants)
            or any(
                type(end) is not int or end != grant.committed_end + count
                for end, grant, count in zip(ends, entry.grants, accepted)
            )
        ):
            raise RuntimeError("native committed extent mismatch")
        self._frontiers.update(
            (g.request, end) for g, end in zip(entry.grants, ends)
        )
        committed = {g.request_id: end for g, end in zip(entry.grants, ends)}
        sample.output.native_target = NativeStepResult(
            self.owner, self._last_step,
            tuple((name, committed[name]) for name in self._request_order),
            self.cache_info(),
        )
        return sample.output

    def _recover_composites_before_drain(self):
        recover = getattr(self._backend, "recover_composite", None)
        if not callable(recover):
            return
        for entry in self._active:
            recovery = recover(entry.ticket)
            if recovery is None:
                continue
            if not isinstance(recovery, NativeCompositeRecovery):
                raise RuntimeError("invalid composite recovery decision")
            if recovery.consumer_complete:
                entry.logits = None
            if recovery.failure is not None:
                # Quarantine keeps the step and native identities; do not guess
                # cancellation/revocation from an unreadable native frontier.
                raise recovery.failure

    def sample_tokens(self, grammar_output: GrammarOutput | None):
        from vllm.v1.outputs import ModelRunnerOutput

        with self._lock:
            backend = self._ready()
            if not self._active:
                raise RuntimeError("no native result to sample")
            assert self._sampler is not None
            try:
                rows = {}
                for entry in self._active:
                    total, offset = sum(len(g.selected) for g in entry.grants), 0
                    for grant in entry.grants:
                        lease = (
                            entry.logits
                            if entry.members is None
                            else _LogitsSlice(
                                entry.logits,
                                offset,
                                offset + len(grant.selected),
                                total,
                            )
                        )
                        rows[grant.request_id] = (grant, lease)
                        offset += len(grant.selected)
                grants = tuple(rows[name][0] for name in self._request_order)
                leases = tuple(rows[name][1] for name in self._request_order)
                assert all(lease is not None for lease in leases)
                sample = self._sampler.sample(grants, leases, grammar_output)
                if (
                    not isinstance(sample.output, ModelRunnerOutput)
                    or sample.output.req_ids != [g.request_id for g in grants]
                    or sample.output.req_id_to_index
                    != {g.request_id: i for i, g in enumerate(grants)}
                    or len(sample.accepted) != len(grants)
                    or len(sample.output.sampled_token_ids) != len(grants)
                    or any(
                        type(n) is not int
                        or not 0 <= n <= len(g.tokens)
                        or (g.phase == "encoder_stream" and n != len(g.tokens))
                        or (
                            g.phase == "dspark"
                            and (n < 1 or len(sample.output.sampled_token_ids[i]) != n)
                        )
                        for i, (n, g) in enumerate(zip(sample.accepted, grants))
                    )
                ):
                    raise RuntimeError("invalid native sampler result")
                if self._composite_eligible(backend):
                    return self._sample_composite(backend, sample)
                # p10: publish every entry's consumer drain, then every
                # entry's commit, before awaiting the replies; each ACK is
                # still consumed in _active order and every validation is
                # byte-for-byte the serial path's. On a mid-phase failure
                # the still-in-flight peer commands are collected so their
                # tickets resolve, then the ordinary cancel_step path
                # recovers exactly the serial failure state.
                self._drain_entries_pipelined()
                committed = {}
                accepted_by_request = dict(zip(self._request_order, sample.accepted))
                if len(self._active) > 1 and self._submit_split_supported(backend):
                    published = []
                    try:
                        for entry in self._active:
                            accepted = tuple(
                                accepted_by_request[g.request_id]
                                for g in entry.grants
                            )
                            command = (
                                backend.begin_commit_batch(entry.ticket, accepted)
                                if entry.members is not None
                                else backend.begin_commit(
                                    entry.ticket, accepted[0]
                                )
                            )
                            published.append((entry, accepted, command))
                    except BaseException:
                        for entry, _, command in published:
                            try:
                                if entry.members is not None:
                                    backend.finish_commit_batch(command)
                                else:
                                    backend.finish_commit(command)
                            except BaseException:
                                continue
                        raise
                    for index, (entry, accepted, command) in enumerate(published):
                        try:
                            ends = (
                                backend.finish_commit_batch(command)
                                if entry.members is not None
                                else (backend.finish_commit(command),)
                            )
                        except BaseException:
                            for peer, _, peer_command in published[index + 1 :]:
                                try:
                                    if peer.members is not None:
                                        backend.finish_commit_batch(peer_command)
                                    else:
                                        backend.finish_commit(peer_command)
                                except BaseException:
                                    continue
                            raise
                        self._active.pop(0)
                        if (
                            not isinstance(ends, tuple)
                            or len(ends) != len(entry.grants)
                            or any(
                                type(end) is not int
                                or end != grant.committed_end + count
                                for end, grant, count in zip(
                                    ends, entry.grants, accepted
                                )
                            )
                        ):
                            raise RuntimeError("native committed extent mismatch")
                        self._frontiers.update(
                            (g.request, end)
                            for g, end in zip(entry.grants, ends)
                        )
                        committed.update(
                            (g.request_id, end) for g, end in zip(entry.grants, ends)
                        )
                else:
                    while self._active:
                        entry = self._active[0]
                        accepted = tuple(
                            accepted_by_request[g.request_id] for g in entry.grants
                        )
                        ends = (
                            backend.commit_batch(entry.ticket, accepted)
                            if entry.members is not None
                            else (backend.commit(entry.ticket, accepted[0]),)
                        )
                        self._active.pop(0)
                        if (
                            not isinstance(ends, tuple)
                            or len(ends) != len(entry.grants)
                            or any(
                                type(end) is not int
                                or end != grant.committed_end + count
                                for end, grant, count in zip(
                                    ends, entry.grants, accepted
                                )
                            )
                        ):
                            raise RuntimeError("native committed extent mismatch")
                        self._frontiers.update(
                            (g.request, end)
                            for g, end in zip(entry.grants, ends)
                        )
                        committed.update(
                            (g.request_id, end) for g, end in zip(entry.grants, ends)
                        )
                sample.output.native_target = NativeStepResult(
                    self.owner,
                    self._last_step,
                    tuple((name, committed[name]) for name in self._request_order),
                    self.cache_info(),
                    tuple(
                        (g.request_id, len(g.tokens) - 1, n - 1)
                        for g, n in zip(grants, sample.accepted)
                        if g.phase == "dspark"
                    ),
                )
                return sample.output
            except BaseException:
                self._failed = True
                self.cancel_step()
                raise

    def cancel_step(self) -> None:
        """Cancel this whole worker step; scheduler may regrant unaffected work."""
        with self._lock:
            self._recover_composites_before_drain()
            self._drain_logits()
            while self._active:
                assert self._backend is not None
                entry = self._active[0]
                if self._backend.cancel(entry.ticket):
                    for grant in entry.grants:
                        self._revoked[grant.request.slot] = grant.request
                        self._frontiers.pop(grant.request, None)
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
            self._revoked.clear()

    def get_supported_tasks(self):
        return ("generate",)

    def get_model(self):
        raise NativeTargetUnavailable("native target has no PyTorch backbone")

    def get_encoder_timing_stats(self):
        return {}

    def reset_mm_cache(self) -> None:
        # API startup invokes this even when multimodal inputs are unsupported.
        with self._lock:
            self._ready()

    def reset_encoder_cache(self) -> None:
        # This is vLLM's vision-output cache, not the native text cache bank.
        with self._lock:
            self._ready()
