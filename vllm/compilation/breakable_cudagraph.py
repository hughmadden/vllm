# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Breakable CUDA graph capture/replay.

This is an alternative to :class:`CUDAGraphWrapper` that replaces vLLM's
torch.compile-based FX graph splitting with runtime stream-capture
breaks.

The idea (inspired by sgl-project/sglang#19102): instead of pre-splitting
the model into many pieces at attention boundaries, a
single capture context drives the whole forward and intercepts
attention / kv-cache custom ops at the dispatcher to end the current
stream capture, run the op eagerly, and resume capture.

The captured artifact is a list of zero-arg callables -- the bound
``CUDAGraph.replay`` for graph segments, or the user fn for eager
segments -- replayed in order at inference time.

Eager segments must operate on the same static buffers used during
capture so subsequent graph segments read the same memory addresses.
"""

from __future__ import annotations

import dataclasses
import functools
import gc
import threading
import weakref
from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

import torch

import vllm.envs as envs
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.utils.torch_utils import weak_ref_tensor, weak_ref_tensors
from vllm.v1.worker.workspace import (
    collect_cuda_graph_capture_resources,
    suspend_cuda_graph_capture_resources,
)

logger = init_logger(__name__)


def is_breakable_cudagraph_enabled() -> bool:
    return bool(envs.VLLM_USE_BREAKABLE_CUDAGRAPH)



# AFD P3 GRAPH HYGIENE
_AFD_CAPTURE_LOCK = threading.RLock()
_AFD_CAPTURE_OPS = {}
_AFD_CAPTURE_AUDIT_GENERATION = 0


def register_capture_safety(name, fn, policy):
    """Scoped registry, not a universal analysis of every dispatched CUDA op.

    Custom host/sync ops must be registered as eager_break after wrapping in
    eager_break_during_capture. Unknown policies and unwrapped host ops fail
    the audit before any stream capture begins.
    """
    if policy not in ("capture_safe", "eager_break", "forbidden"):
        raise ValueError(f"Unknown capture policy for {name}: {policy}")
    global _AFD_CAPTURE_AUDIT_GENERATION
    _AFD_CAPTURE_AUDIT_GENERATION += 1
    # Metadata only: a registry must never pin a per-capture closure/tensor.
    _AFD_CAPTURE_OPS[name] = (policy, bool(getattr(fn, "_afd_eager_break", False)),
                             _AFD_CAPTURE_AUDIT_GENERATION)


def audit_registered_capture_ops():
    for name, (policy, verified_eager, generation) in tuple(_AFD_CAPTURE_OPS.items()):
        if policy == "forbidden" or (policy == "eager_break"
                and not verified_eager) or generation <= 0:
            raise RuntimeError(f"Capture safety audit rejected {name}: {policy}")
    return tuple(sorted(_AFD_CAPTURE_OPS))


# These tensor operations underlie the stable AFD exchange/output planes.
# Host transport operations are registered by the eager-break decorator below.
for _name in ("copy_", "add", "mul"):
    register_capture_safety("torch.Tensor." + _name,
                            getattr(torch.Tensor, _name), "capture_safe")


def _afd_signature(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                str(value.dtype), str(value.device))
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_afd_signature(item) for item in value))
    if type(value) is dict:
        return ("dict", tuple((key, _afd_signature(item)) for key, item in value.items()))
    if value is None or type(value) in (str, int, float, bool):
        return (type(value).__name__, value)
    return (type(value).__qualname__, id(value))


def _afd_exec_update(old, new):
    """Unsupported updates use the fresh executable; never destroy borrowed handles."""
    try:
        from cuda.bindings import runtime
        status, info = runtime.cudaGraphExecUpdate(old.raw_cuda_graph_exec(), new.raw_cuda_graph())
        ok = (status == runtime.cudaError_t.cudaSuccess
              and info.result == runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateSuccess)
        if not ok:
            # cudaErrorGraphExecUpdateFailure is expected for topology changes.
            # Clear the runtime's last-error slot before subsequent torch calls.
            runtime.cudaGetLastError()
        return bool(ok)
    except (ImportError, AttributeError, RuntimeError, TypeError) as error:
        logger.debug("AFD graph update unavailable; using fresh executable: %s", error)
        return False


def _afd_serialized_capture(fn):
    @functools.wraps(fn)
    def locked(*args, **kwargs):
        with _AFD_CAPTURE_LOCK:
            return fn(*args, **kwargs)
    return locked

F = TypeVar("F", bound=Callable[..., Any])


def eager_break_during_capture(fn: F) -> F:
    """Decorator that turns a custom-op Python kernel into a "break point"
    for the breakable cudagraph capture.

    When the decorated function is invoked outside of a
    :class:`BreakableCUDAGraphCapture` context, it executes normally.

    When invoked inside a capture context, it ends the current cudagraph
    segment, runs the function eagerly on the capture stream, records the
    callable for replay, and starts a fresh segment.

    **In-place output buffer required.** Decorated ops must write into a
    caller-provided output tensor; a fresh tensor returned by ``fn`` would
    change address each replay and break downstream graph segments.

    **Decorator order matters.** Apply as the *outermost* decorator if
    there are other decorators that introduce host-side side effects
    around the call -- the canonical example is
    ``@maybe_transfer_kv_layer`` for PD-disaggregation, whose
    ``wait_for_layer_load`` and ``save_kv_layer`` calls must run in the
    eager segment, not inside the captured cudagraph. Putting
    ``@eager_break_during_capture`` *inside* such a decorator would
    record those side effects into the graph and hang on replay.

    The correct order is::

        @eager_break_during_capture   # outermost
        @maybe_transfer_kv_layer
        def unified_attention_with_output(...):
            ...
    """
    if not is_breakable_cudagraph_enabled():
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        capture = BreakableCUDAGraphCapture.current()
        if capture is None:
            return fn(*args, **kwargs)
        if not capture._capturing:
            return fn(*args, **kwargs)
        if is_forward_context_available():
            mode = get_forward_context().cudagraph_runtime_mode
            if mode == CUDAGraphMode.FULL:
                return fn(*args, **kwargs)

        # Weak-ref args: strong refs in the replay lambda pin cudagraph-pool
        # slots across batch descriptors. cudagraph owns the slot, so the
        # weak_ref is safe to deref on replay.
        weak_args = _weak_tensor_arguments(args)
        weak_kwargs = _weak_tensor_arguments(kwargs)
        return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))

    wrapper._afd_eager_break = True
    register_capture_safety(fn.__module__ + "." + fn.__qualname__, wrapper, "eager_break")
    return wrapper  # type: ignore[return-value]


def _weak_tensor_arguments(value: Any) -> Any:
    """Keep tensor arguments weak even inside ordinary tuple/list/dict bundles."""
    if isinstance(value, torch.Tensor):
        return weak_ref_tensor(value)
    if type(value) is tuple:
        return tuple(_weak_tensor_arguments(item) for item in value)
    if type(value) is list:
        return [_weak_tensor_arguments(item) for item in value]
    if type(value) is dict:
        return {key: _weak_tensor_arguments(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Capture context
# ---------------------------------------------------------------------------


class BreakableCUDAGraphCapture:
    """Segmented graphs with process-wide capture serialization and owned cleanup."""
    _tls = threading.local()

    @classmethod
    def current(cls):
        return getattr(cls._tls, "active", None)

    @classmethod
    def is_active(cls):
        return cls.current() is not None

    def __init__(self, pool=None):
        self.pool = pool
        self.segments = []
        self._graphs = []
        self._graph_segment_indices = []
        self._retained_graphs = []
        self._update_generation = 0
        self._num_graphs = 0
        self._num_eager_breaks = 0
        self._current_graph = None
        self._capturing = False
        self._failed = False
        self._afd_pending_cleanups = []
        self.capture_audit = ()
        self.update_stats = {"updated": 0, "fallback": 0}

    def __enter__(self):
        if self.current() is not None:
            raise RuntimeError("Nested BreakableCUDAGraphCapture is not supported.")
        if self.segments or self._failed:
            raise RuntimeError("Capture context must be fresh before entering.")
        _AFD_CAPTURE_LOCK.acquire()
        try:
            self.capture_audit = audit_registered_capture_ops()
            self.capture_audit_generation = _AFD_CAPTURE_AUDIT_GENERATION
            self._tls.active = self
            self._begin_segment()
            return self
        except BaseException:
            self._failed = True
            self._cleanup_pending(suppress=True)
            self._tls.active = None
            _AFD_CAPTURE_LOCK.release()
            raise

    def __exit__(self, exc_type, exc, tb):
        cleaned = False
        try:
            try:
                self._end_segment()
            except BaseException:
                self._failed = True
                self._cleanup_pending(suppress=True)
                cleaned = True
                if exc_type is None:
                    raise
            if exc_type is not None and not cleaned:
                self._failed = True
                self._cleanup_pending(suppress=True)
        finally:
            self._tls.active = None
            _AFD_CAPTURE_LOCK.release()

    def _cleanup_pending(self, suppress=False):
        first_error = None
        for cleanup in tuple(self._afd_pending_cleanups):
            try:
                cleanup()
            except BaseException as error:
                first_error = first_error or error
                logger.exception("AFD pending exchange cleanup failed")
        if first_error is not None and not suppress:
            raise first_error

    def _begin_segment(self):
        assert not self._capturing
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        if self.pool is not None:
            graph.capture_begin(pool=self.pool, capture_error_mode="global")
        else:
            graph.capture_begin(capture_error_mode="global")
        self._current_graph = graph
        self._capturing = True

    def _end_segment(self):
        if not self._capturing:
            return
        graph = self._current_graph
        try:
            graph.capture_end()
            graph.instantiate()
            self._graph_segment_indices.append(len(self.segments))
            self._graphs.append(graph)
            self.segments.append(graph.replay)
            self._num_graphs += 1
            # Materialize the captured producer before its eager consumer/fence.
            graph.replay()
        finally:
            self._current_graph = None
            self._capturing = False

    def add_eager(self, fn):
        self._end_segment()
        with suspend_cuda_graph_capture_resources():
            result = fn()
        self.segments.append(fn)
        self._num_eager_breaks += 1
        self._begin_segment()
        return result

    def replay(self):
        if self._failed:
            raise RuntimeError("Cannot replay a failed AFD capture")
        with _AFD_CAPTURE_LOCK:
            try:
                for segment in self.segments:
                    segment()
            except BaseException:
                self._failed = True
                self._cleanup_pending(suppress=True)
                raise

    def adopt_updated_execs(self, previous):
        """Called only after a synchronized recapture of this descriptor.

        New eager closures/output buffers remain authoritative. A reused exec
        owns its old graph, while the new template and both resource lists
        stay alive. Force a fresh generation after one update to bound retained
        graph-pool memory even if pointers change continually.
        """
        if (previous._failed or previous._update_generation
                or self._graph_segment_indices != previous._graph_segment_indices
                or len(self.segments) != len(previous.segments)):
            self.update_stats["fallback"] += len(self._graphs)
            return False
        for index, (old, new) in enumerate(zip(previous._graphs, self._graphs)):
            if _afd_exec_update(old, new):
                self._graphs[index] = old
                self.segments[self._graph_segment_indices[index]] = old.replay
                self._retained_graphs.append(new)
                self.update_stats["updated"] += 1
            else:
                self.update_stats["fallback"] += 1
        if self.update_stats["updated"]:
            self._retained_graphs.extend(previous._graphs)
            self._retained_graphs.extend(previous._retained_graphs)
            self._update_generation = 1
            return True
        return False

    def reset(self):
        with _AFD_CAPTURE_LOCK:
            if self._capturing:
                raise RuntimeError("Cannot reset an active breakable CUDA graph capture.")
            self._cleanup_pending()
            torch.accelerator.synchronize()
            seen = set()
            for graph in self._graphs + self._retained_graphs:
                if id(graph) not in seen:
                    graph.reset()
                    seen.add(id(graph))
            self._graphs.clear()
            self._retained_graphs.clear()
            self._graph_segment_indices.clear()
            self._afd_pending_cleanups.clear()
            self.segments.clear()

    @property
    def num_graphs(self):
        return self._num_graphs

    @property
    def num_eager_breaks(self):
        return self._num_eager_breaks

    def __repr__(self):
        return (f"BreakableCUDAGraphCapture(graphs={self.num_graphs}, "
                f"eager_breaks={self.num_eager_breaks})")


# ---------------------------------------------------------------------------
# Wrapper that mirrors CUDAGraphWrapper's interface
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _BreakableEntry:
    batch_descriptor: BatchDescriptor
    capture: BreakableCUDAGraphCapture | None = None
    output: Any = None
    input_addresses: list[int] | None = None
    resources: list[Any] | None = None
    input_signature: Any = None
    pending_signature: Any = None
    stable_calls: int = 0


class BreakableCUDAGraphWrapper:
    """Drop-in replacement for :class:`CUDAGraphWrapper` that uses
    :class:`BreakableCUDAGraphCapture` instead of a single monolithic
    ``torch.cuda.graph()`` capture.

    Same dispatch contract as ``CUDAGraphWrapper``:
        * If no ``forward_context`` is available, run the underlying
          callable eagerly.
        * If runtime mode mismatch / NONE, run eagerly.
        * Otherwise, lazily capture per ``batch_descriptor`` and replay
          on subsequent invocations with the same descriptor.
    """

    _all_instances: ClassVar[weakref.WeakSet[BreakableCUDAGraphWrapper]] = (
        weakref.WeakSet()
    )

    @classmethod
    def clear_all_graphs(cls) -> None:
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    @classmethod
    def reset_all_graphs(cls) -> None:
        """Destroy graph segments without releasing entry-owned resources."""
        for instance in list(cls._all_instances):
            instance.reset_graphs()

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
    ) -> None:
        # Unlike the original CUDAGraphWrapper which strictly matches a
        # single runtime_mode, this wrapper captures whatever the
        # dispatcher emits (any non-NONE runtime_mode) -- breakable's
        # capture is identical for prefill and decode, so there's nothing
        # to dispatch on at the runtime_mode level. Entries are keyed by
        # BatchDescriptor which already encodes batch shape / uniformity.
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = current_platform.get_global_graph_pool()
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        self.entries: dict[BatchDescriptor, _BreakableEntry] = {}
        self.afd_graph_stats = {"capture": 0, "replay": 0, "warmup": 0, "updated": 0, "fallback": 0}
        BreakableCUDAGraphWrapper._all_instances.add(self)

        logger.info_once("Breakable CUDA graph enabled")

    # --- vllm-style attribute forwarding ---------------------------------

    def __getattr__(self, key: str) -> Any:
        runnable = self.__dict__.get("runnable")
        if runnable is not None and hasattr(runnable, key):
            return getattr(runnable, key)
        raise AttributeError(key)

    def unwrap(self) -> Callable[..., Any]:
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> BreakableCUDAGraphWrapper:
        return self

    def clear_graphs(self) -> None:
        self.reset_graphs()
        self.entries.clear()

    def reset_graphs(self) -> None:
        """Destroy graph segments while retaining their captured resources."""
        for entry in self.entries.values():
            capture = entry.capture
            if capture is not None:
                capture.reset()
                entry.capture = None

    # --- dispatch --------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not is_forward_context_available():
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # Capture whenever the dispatcher says "some cudagraph mode" --
        # breakable produces the same artifact regardless of PIECEWISE
        # vs FULL, so we match either. Entries are keyed by batch
        # descriptor, which already encodes prefill/decode distinctions.
        if cudagraph_runtime_mode == CUDAGraphMode.NONE:
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        entry = self.entries.get(batch_descriptor)
        if entry is None:
            entry = _BreakableEntry(batch_descriptor=batch_descriptor)
            self.entries[batch_descriptor] = entry

        signature = _afd_signature((args, kwargs))
        if entry.capture is not None and signature == entry.input_signature:
            self.afd_graph_stats["replay"] += 1
            return self._replay(entry, args, kwargs)
        if signature != entry.pending_signature:
            entry.pending_signature = signature
            entry.stable_calls = 1
        else:
            entry.stable_calls += 1
        if entry.stable_calls < 2:
            self.afd_graph_stats["warmup"] += 1
            return self.runnable(*args, **kwargs)
        return self._capture(entry, args, kwargs)

    # --- capture / replay paths -----------------------------------------

    @staticmethod
    def _collect_tensor_addresses(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> list[int]:
        """Flatten tensor data_ptrs from positional and keyword args in a
        stable order (positionals first, then kwargs in insertion order).

        Used for the DEBUG-mode address-stability check; covers both call
        styles since vLLM models are typically invoked with kwargs.
        """
        addrs = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        addrs.extend(
            v.data_ptr() for v in kwargs.values() if isinstance(v, torch.Tensor)
        )
        return addrs

    @_afd_serialized_capture
    def _capture(
        self,
        entry: _BreakableEntry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        validate_cudagraph_capturing_enabled()

        entry.input_addresses = self._collect_tensor_addresses(args, kwargs)

        if self.graph_pool is not None:
            set_graph_pool_id(self.graph_pool)
        else:
            set_graph_pool_id(current_platform.graph_pool_handle())

        # Match torch.cuda.graph()'s pre-capture barrier and cleanup once per
        # descriptor. The warmup immediately before this call may use shared
        # communication scratch. Starting capture before that work completes
        # lets the captured kernels race the warmup on the same storage.
        # We drive capture_begin/end directly and bypass torch.cuda.graph(),
        # so its synchronize + gc + empty_cache sequence never runs. Run it
        # here once per _capture call -- NOT inside _begin_segment, since this
        # capture session may issue many begin/end pairs (one per layer's
        # break), and repeated cleanup would dominate capture time.
        torch.accelerator.synchronize()
        gc.collect()
        torch.accelerator.empty_cache()
        # Sync the offloader's copy stream before capture so any in-flight
        # pre-capture prefetches are complete and don't leak into the graph.
        get_offloader().sync_prev_onload()

        capture = BreakableCUDAGraphCapture(pool=self.graph_pool)
        with collect_cuda_graph_capture_resources() as resources, capture:
            output = self.runnable(*args, **kwargs)
            # Join the offloader's copy stream while we still hold the last
            # segment open, so the join is captured into the graph (otherwise
            # we get an "unjoined stream" error on subsequent forwards).
            get_offloader().join_after_forward()
            # Convert output to a weak ref *inside* the capture context so the
            # strong ref is dropped before the last segment closes, letting
            # the cudagraph pool reclaim/reuse that memory immediately for
            # the next batch descriptor's capture.
            output = weak_ref_tensors(output)

        previous = entry.capture
        previous_resources = entry.resources or []
        if previous is not None:
            retained = capture.adopt_updated_execs(previous)
            if retained:
                resources = previous_resources + resources
            else:
                previous.reset()
            for key in ("updated", "fallback"):
                self.afd_graph_stats[key] += capture.update_stats[key]
        self.afd_graph_stats["capture"] += 1
        entry.capture = capture
        entry.input_signature = _afd_signature((args, kwargs))
        entry.resources = resources
        entry.output = weak_ref_tensors(output)

        logger.debug(
            "Captured breakable cudagraph for %s: %r",
            entry.batch_descriptor,
            capture,
        )
        # Return the (already-weak) output from the captured run so the
        # caller of model(...) gets a tensor pointing at the cudagraph pool's memory
        return output

    def _replay(
        self,
        entry: _BreakableEntry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if self.is_debugging_mode and entry.input_addresses is not None:
            new_addresses = self._collect_tensor_addresses(args, kwargs)
            assert new_addresses == entry.input_addresses, (
                "Input tensor addresses changed between capture and replay "
                f"for {entry.batch_descriptor}. Expected "
                f"{entry.input_addresses}, got {new_addresses}."
            )
        # Sync the offloader's copy stream before replay so any external
        # dependencies from pre-capture prefetches are satisfied.
        get_offloader().sync_prev_onload()
        assert entry.capture is not None
        entry.capture.replay()
        return entry.output
