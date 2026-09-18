# Native target selection and worker lifecycle

19 September 2026 AEST. This is a selectable engine/worker seam with CPU fake
backend tests. It is off by default. A production target C ABI consumer,
vLLM sampler adapter and native scheduler must register together before serving
can start. This packet does not claim a vLLM GPU run, numerical parity or speed.

The retained Rust API at `ce1e1cd83b57da24b5f18abd352002ef4fc53bcb`
(`native_executor::target::with_target`) supplies the actual `Requests` /
`BackboneCache` bank and two `TargetPass` contexts. The independent schema-1
`CacheCommands` prototype must **never** be instantiated beside that bank.
Its earlier cross-language tests qualify the metadata contract only.

## Actual selection points

`additional_config={"afd_native_target": {"enabled": true}}` selects the path.
A general plugin calls `register_native_target(binding)` in the engine and
worker environments. The binding provides configuration validation, a sampler
factory, one combined native constructor and a scheduler factory. Configuration
validation must reject unsupported checkpoint/device/budget combinations before
allocation. Without that registered binding, EngineCore rejects selection before
constructing an executor; there is no fallback that loads ordinary weights.

The initial supported integration scope is one `uni` worker with synchronous V2
generation. Prefix caching, vLLM TP/PP/DP, async scheduling, speculation, LoRA,
sleep, KV/EC connectors and weight transfer reject at selection. The model may
have vision modules, but scheduled image inputs reject at execution. Only the
`full_target` phase with prefill/decode source kinds is currently accepted.

| Entry point | Selected native behavior |
| --- | --- |
| `Worker.init_device` | Constructs only `NativeTargetRunner` and a device descriptor before the ordinary distributed/workspace/V2 constructors. The binding owns CUDA initialization. |
| `Worker.load_model` | Creates the sampler first, then calls the combined native constructor once. The ordinary PyTorch backbone loader is bypassed. |
| Native constructor | Loads shared weights, both target/transport contexts and retained staging; runs native memory planning; allocates one actual bank; returns actual capacity. These are internal ordered stages of one call, not three separately callable APIs. Partial construction must unwind before returning an error. |
| `EngineCore._initialize_cache_and_scheduler` | Reads `native_cache_info` via executor RPC, validates one native receipt, then calls the native scheduler factory. Ordinary KV specs, memory profiling, KV tensors and the scheduler `BlockPool` are bypassed. A failed receipt/scheduler initialization closes the worker. |
| Ordinary worker cache methods | Explicitly reject when native is selected. No empty/fake KV tensor or fake physical block count is supplied. |
| `Worker.execute_model` | Updates the sampler's vLLM request metadata, submits up to two disjoint grants before driving native execution, then retains both logits leases. |
| `Worker.sample_tokens` | Invokes the supplied vLLM sampler, drains its GPU consumers, releases logits borrows, then commits only accepted rows through the actual native contexts. Returns a real `ModelRunnerOutput`. |
| `native_cancel_step`, `native_release`, `shutdown` | Drain logits consumers before cancel/discard; release requests only when the step no longer owns them. Close does not reclaim a timed-out consumer lease. |

Sampler allocations precede native planning so they are visible to its memory
calculation. `NativeCacheInfo.cache_bytes` is the native-reported actual cache
allocation, including window storage. It is **not** interchangeable with the
requested global source-pool budget. Weight/context/vision/transport/staging and
sampling workspaces must also remain in the binding's complete device budget.
This receipt does not report total model memory or establish that a larger
configuration fits. The engine does not convert four independent native source
pool credits into one ordinary vLLM `num_gpu_blocks` count.

## Scheduler, request and sampling contract

The worker exposes `native_admit(slot, numeric_request_id)`, returning the actual
`{owner, slot, generation}` lease, and `native_release(lease)` returning refreshed
native credits. The scheduler binding must implement `SchedulerInterface` using
these operations and the native receipts. It can track logical admission and
reservations; it must not create a second physical allocator. It must maintain
the usual request completion, stop conditions, fairness and output handling.

`SchedulerOutput.native_target` contains a `NativeSchedule(owner, step_id,
grants)`; each immutable grant carries:

- vLLM request ID and native request generation;
- lane 0 or 1, expected native committed end;
- exactly the granted owned token IDs and strictly increasing selected-logit rows;
- prefill/decode source kind, expert placement identity and `full_target` phase.

The runner checks the token counts against `num_scheduled_tokens`, actual native
capacity and actual `committed_end`. It rejects two lanes for the same request,
finished/preempted requests, stale owners/steps, unsupported phases and more than
48 selected rows per context (the current retained compact-head limit). Empty
steps may update/retire sampler metadata and return an empty output receipt.
The native backend must additionally enforce its checkpoint vocabulary, context
length, generation and device extents; those are authoritative native checks.

`NativeSampler.update_requests(SchedulerOutput)` receives the existing new/cached
request metadata and finished/preempted IDs. Its adapter must retain vLLM's
sampling parameters, token histories, RNG, penalties, grammar and logprob
semantics. `sample(grants, leases, grammar_output)` returns `NativeSample` with a
real `ModelRunnerOutput` and an accepted input-row count per grant. No greedy
native sampler is substituted. The sampler must register every GPU consumer on
the lease even when it later raises; all consumers must drain before native
publication or reuse. The initial C ABI consumer may constrain this to one
explicit CUDA stream; a second stream requires its own retained event.

`ModelRunnerOutput.native_target` carries the executor owner, step ID, actual
committed ends and refreshed source credits, after all commits succeed. It is
not the schema-1 ACK and does not introduce a second epoch ledger. A native
publication error fails the scope and produces no model output, even if another
request's earlier commit succeeded. The scheduler must retire that scope and
recompute rather than infer rollback of a commit already acknowledged by native.

The C ABI consumer is responsible for correlated command IDs, exact retry of a
lost native reply and bounded completion polling. This runner does not retry an
entire sampled step: it accepts strictly increasing step IDs and retains only
the active step. A worker restart mints a new nonzero owner; the scheduler must
discard every old request lease/receipt and re-admit from uncached tokens.

Cancellation currently applies to the entire active worker step. After successful
drain/discard, a scheduler may issue a new step for an unaffected request. Python
calls serialize under an `RLock`; a concurrent cancel cannot free a sampler's
borrow. Responsive cancellation during an in-flight native wait still belongs
in the C ABI's owner-thread command loop. This is not proof of asynchronous
engine scheduling or same-prompt two-chunk streaming.

## Remaining integration and qualification

The following are still required before registering a production binding:

1. Connect the actual native C ABI and its CUDA result leases to the backend
   protocol, preserving pending command/lease ownership across timeouts.
2. Implement the scheduler with actual native credits and acknowledgements,
   including request admission/release and preemption/recompute, and adapt the
   existing vLLM sampler without loading a PyTorch backbone. A factory object
   that merely satisfies Python method names does not qualify serving.
3. Bind full checkpoint lifecycle before advertising prefix hits: native source
   pages alone omit windows, compressor/Engram history and replay state. Native
   source prefix lookup must remain excluded from vLLM cache-hit reporting.
4. Add explicit encoder-chunk token grants, predecessor publication fences,
   retained suffix and final decoder replay before enabling streaming prefill.
   A full-target call cannot stand in for encoder/replay execution. dSpark taps
   and accepted-prefix transactions need their own binding afterward.
5. Run the strict numerical gates on the completed vLLM path, including mixed
   batch/request histories and C2 logprob/top-k invariance, then measure memory
   and throughput. Existing expert repeatability or a standalone native target
   smoke does not waive these gates.

## CPU validation

`tests/v1/worker/test_native_target.py` uses fake native completion/storage and
the real output dataclass definitions. Hook tests execute the actual Worker and
EngineCore entry method bodies with injected CPU dependencies; decorators are
removed, and the full GPU modules are not imported. Ordinary initialization
functions are deliberately absent in native hook tests, so reaching one fails.
This catches selection ordering without pretending a torch-less host ran the
GPU worker.

Tests cover constructor counts/order, authoritative bank receipts, failed cache
or scheduler initialization, both-lane submission, active-result reuse,
concurrent cancellation, sampling failure after launch, consumer-drain timeout
retention, zero/partial accepted publication, commit failure and invalid grants.
The ordinary EngineCore initialization order is tested with native selection off.

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-project \
  --python .venv/bin/python .venv/bin/python -m pytest --noconftest \
  tests/v1/worker/test_native_target.py -q
```

The native runner suite passes 29 cases; combined with the prior Python adapter
and actual Rust command-contract suites, 56 cases pass with no skips.

No dependencies were installed. The retained Ruff executable provides scoped
lint/format checks. Full pre-commit is unavailable in the retained offline
environment; torch and msgspec are also absent, so full worker imports and IPC
serialization are not qualified here. The selected initial `uni` path does not
serialize scheduler/worker grants through a multiprocessing transport.
