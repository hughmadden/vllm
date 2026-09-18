# Disabled native executor cache command adapter

`vllm/v1/core/native_cache_adapter.py` is the scheduler side of the DS41RT
schema-1 command seam. It emits serializable dictionaries and consumes explicit
worker acknowledgements. It holds no endpoint, device pointer, tensor, physical
block pool or native allocator. Nothing selects it in the serving launcher.

The companion Rust schema is pinned at
`aa12dda907d755064cd6eca94de6bad65f2a09e8`, in
`rust/crates/ds41rt-daemon/src/native_executor.rs`. Its source-page ledger is
metadata only until attached to the retained native storage owner.

## Authority and lifecycle

The worker supplies a fresh snapshot with a nonzero, never-reused incarnation
owner and epoch zero. Commands are `{owner, epoch, command_id, command}` with
the Rust `op` tags. Request handles `{owner, slot, generation}`, transaction
handles and prefix IDs originate in native acknowledgements. Python allocates
only bounded logical names/IDs; it never derives physical page IDs from vLLM
block tables or converts free pages into a speculative token capacity promise.

One unresolved command is retained. `retry()` returns a detached copy of that
exact envelope, including its original ID and epoch. Native exact-last-command
retry makes a lost ACK safe. Transport errors/timeouts must leave it unresolved;
`reject(owner, epoch, command_id)` is only for a correlated native no-mutation
`Result::Err`, not a guessed transport failure. An identical duplicate of the
last ACK is ignored without consuming a later pending command. Changed/stale
ACKs fail with local state and retry bytes intact.

Successful ACK validation precedes every local update. Snapshot owner/epoch,
physical capacities, request generations/extents and transaction/prefix counts
must agree. Free credits come directly from the native snapshot, including
copy-on-write reservations and delayed free. Reader counts may progress
independently; only native completion controls publication and release.

`begin_batch` represents proposed work. `reserve_publish(accepted)` specifies
the accepted portion before physical source publication; `publish()` may return
pending. Only its successful ACK advances the visible source extent. Rejected
speculative suffixes never become committed and are not later truncated from
an already published cache. A failed start with `draining:true` keeps its
transaction/credits until abort or release drains. A drained write failure
discards the transaction without advancing committed extent.

Two different requests may occupy the two native lanes and complete in either
order. The same request cannot begin overlapping transactions in this initial
schema. Internal encoder chunk streaming and complete target execution are
separate native integration work; row reservations are not input-token delivery.

`revoke()` hides a request immediately, even with an ACK outstanding. `free()`
uses the same barrier for cancellation or preemption and emits native release
when the earlier envelope is resolved. It may raise `CommandPending` after
setting the barrier; retry/acknowledge that envelope, then retry `free()`.
Native `closing` keeps the slot/credits owned until `drain()` reports released.
A confirmed worker restart invalidates all local request and prefix handles;
`restart()` returns request names requiring recomputation. It does not claim
the previous process's physical allocations were freed by the new snapshot.

## Source references are not model cache hits

`retain_source_prefix`, `lookup_source_prefix`, `restore_source_prefix`,
`drop_source_prefix` and `reset_prefix_cache` concern **source-page references
only**. Keys are caller-supplied canonical 32-byte content digests. Native owns
references and shared-tail copy-on-write; the adapter mirrors only the returned
opaque identity/extent. Acknowledged reset drops the lookup index without
inventing free credits while requests/readers retain pages.

These operations lack window-ring snapshots, compressor carry, Engram history,
dSpark state and payload completion for a full checkpoint.
`get_computed_blocks()` and `require_serving_ready()` therefore always raise
`IntegrationDisabled`. `source_committed_tokens()` explicitly reports a source
extent, not `Request.num_computed_tokens`. No ordinary KV blocks or synthetic
empty tensors are returned to make the current scheduler accept this backend.

The adapter intentionally does not subclass `KVCacheManager`: its constructor
creates the ordinary `BlockPool`, violating the single physical owner contract.
Its free/reset/publication lifecycle follows the existing manager's scheduler
hooks, but integrating those hooks requires a separate backend choice before
`get_kv_cache_spec`/`initialize_from_config` can allocate ordinary KV.

## CPU verification and remaining integration

From the repository root, using the retained uv-created environment:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-project \
  --python .venv/bin/python .venv/bin/python -m pytest --noconftest \
  tests/v1/core/test_native_cache_adapter.py -q
```

The test imports this stdlib-only module directly. `--noconftest` avoids the
root fixtures that load model/GPU dependencies; no such fixtures are needed for
the command contract. A scripted endpoint controls ACK loss, completion order,
write failures, readers and reported copy-on-write credits. It does not
reimplement the native physical allocator or qualify kernel/cache mathematics.
Native ownership/refcount behavior remains covered by the Rust packet's tests.

No packages were installed or downloaded. Ruff 0.15.10 from the retained uv
cache checks/formats the two Python files. The repository pins Ruff 0.14.0 and
requires pre-commit, but pre-commit is absent locally and unavailable in the
offline cache; full pre-commit hooks remain a recorded tooling gap.

Before enabling serving, connect commands/ACKs through scheduler output and
worker results, attach the ledger to the actual native storage owner, report
weight/context/snapshot bytes before memory admission, and supply complete
checkpoint semantics. Connect real token/phase/selected-logit inputs and output
leases to the retained target executor and vLLM sampler. Confirm worker exit
before incarnation replacement, and provide recompute or complete-checkpoint
resume for preempted requests. This packet changes no model execution and makes
no model-quality, throughput or GPU-memory qualification claim.
