# Native target selection and worker lifecycle

19 September 2026 AEST. The default-disabled retained binding connects the native
target C ABI, actual native cache bank, text scheduler and existing vLLM sampler.
CPU tests exercise real vLLM request/output types and sampler behavior. This
packet does not claim a vLLM GPU run, numerical parity or speed; those remain
live qualification gates.

## Selection and dependencies

`additional_config.afd_native_target.enabled=true` selects this path.
`implementation="retained"` lazily registers the built-in production binding in
each process. No general plugin metadata or legacy `VLLM_AFD` transport hook is
required. Alternate bindings can explicitly register before selection. An enabled
selection without a binding rejects before executor construction.

Use package client `2174292f705743dda54b20f177dd1f3206d4ad2e`, including the
Torch 2.14 CUDA array-interface correction and encoder-stream support, with Rust
target C ABI `4ec58286ea357fd306b0780861b79d502ebdca84`.
Compatible descendants may extend this interface. This binding
uses the actual `NativeBank` / `TargetContext` ownership, never the independent
schema-1 `CacheCommands` prototype. Its earlier cross-language tests concern
metadata only and do not qualify the live physical bank.

The API/tokenizer checkpoint and native checkpoint must represent the same
DS4.1 Flash model. Dimensions are checked; checkpoint equivalence still requires
the deployment manifest and token-ID comparison. Example `--additional-config`
JSON, with example paths and addresses that must be replaced for deployment:

```json
{"afd_native_target": {
  "enabled": true,
  "implementation": "retained",
  "prefill_mode": "full_target",
  "abi_library": "/artifacts/libds41rt_daemon.so",
  "snapshot": "/weights/native-ds41",
  "native_lib": "/artifacts/libds41rt_native.so",
  "peers": ["127.0.0.2:9200", "127.0.0.3:9200", "127.0.0.4:9200", "127.0.0.5:9200"],
  "batch_tokens": 256,
  "slots": 2,
  "source_pool_budget_bytes": 536870912,
  "timeout_s": 60,
  "poll_interval_s": 0.00005
}}
```

Select `VLLM_USE_V2_MODEL_RUNNER=1`,
`--distributed-executor-backend uni`, `--no-async-scheduling`,
`--no-enable-prefix-caching`, `--max-num-seqs 2`,
`--max-num-batched-tokens 512`, a measured `--max-model-len`, and
`--enable-prompt-tokens-details`. Keep the intended model/tokenizer metadata,
served model name and chat template options. The 512 MiB example budgets native
source payload, not all native memory. The legacy AFD plugin should be disabled.

## Construction and physical ownership

| Entry point | Selected native behavior |
| --- | --- |
| `Worker.init_device` | Creates `NativeTargetRunner` before ordinary distributed/workspace/V2 construction. |
| `Worker.load_model` | Allocates the sampler first, retains the asynchronous native owner, then waits for its one combined initialization ACK. No PyTorch backbone loader runs. |
| Native initialization | Loads shared weights and both contexts/transports/staging; plans and allocates one real `Requests` / `BackboneCache` bank. These are internal stages of one call. |
| `EngineCore._initialize_cache_and_scheduler` | Reads one authoritative worker capacity receipt, creates `NativeScheduler`; bypasses ordinary KV specs, profiling, KV tensors and `BlockPool`. Failed initialization closes the worker. |
| Ordinary worker KV methods | Reject selection; no fake KV tensors or physical block counts are supplied. |
| `execute_model` | Updates sampling metadata, submits both disjoint grants before execution, retains result leases. |
| `sample_tokens` | Runs vLLM sampling, drains CUDA consumers, releases native logits borrows, commits input rows and returns a real `ModelRunnerOutput`. |
| Cancel/release/shutdown | Drain consumers before discard/reuse; retained commands and leases survive timeouts. |

`NativeCacheInfo.cache_bytes` is actual native cache allocation including windows.
It is not the requested `source_pool_budget_bytes` or total model memory. Shared
weights, contexts, retained vision workspace, transport and sampler scratch also
consume memory. The scheduler never converts four source pools into an ordinary
vLLM `num_gpu_blocks` value.

## Scheduler and sampling

`NativeScheduler` queues requests, issues bounded full-target prefill/decode chunks
to two disjoint contexts, uses upstream stop/max-token rules and returns ordinary
`EngineCoreOutputs`. Existing tokenization, detokenization, text stops, API output
processing, usage and logprob handling remain in vLLM. Initial prefill explicitly
reports zero cached tokens. Reset/preemption releases the actual native request,
re-admits a new generation and recomputes prompt plus emitted history without
emitting the old history again.

Admission asks `native_can_prepare` about the aggregate remaining maximum token
demand of all admitted requests plus the candidate. Before dispatch, both selected
grants are checked together again. These operations delegate to the actual native
append-capacity planner while both contexts are idle. One synchronous engine owns
check/dispatch; these queries are not independently retained reservations. Python
has no source-page formula or physical free list. Worst-case logical admission
avoids exhausting a running request midway, but can reduce concurrency; measure
that cost before changing the policy.

Each immutable grant carries request ID/generation, lane, expected committed end,
owned token IDs, selected rows, prefill/decode kind and `full_target` phase. The
runner checks step/owner, extents, disjointness and native committed end. All
publication receipts and sampled row counts validate before scheduler mutation.
The C ABI client owns correlated retry and timeout state. A lost reply resumes its
original command; neither sampled steps nor native commits are blindly replayed.
A native publication error poisons the scope and emits no model output, even if
another request's earlier native commit already succeeded.

The sampler uses vLLM's existing `Sampler` and built-in processors, preserving
parameters, output histories, per-request RNG, penalties, min tokens, logit bias,
min-p, top-k/top-p and output logprobs. Native FP32 logits are semantically read-only;
upstream sampling mutates logits, so the zero-copy CUDA view is copied into two
reusable GPU scratch rows. No full-vocabulary D2H copy is added. Every request is
sampled as a one-row batch independent of its lane partner. The client binds each
lease to the current PyTorch CUDA stream. Native records/drains its completion
event before commit or context reuse. Consumer timeout retains the borrow.

Request preprocessing rejects unsupported modes before the scheduler's serving
loop: images/embeddings, structured output, priority, prompt/full-vocabulary or
specific-token logprobs, trace replay, recurrent checkpoints, custom extra args,
thinking-token budgets and streaming input. Configuration rejects vLLM parallel
workers, async scheduling, prefixes, speculation, LoRA, connectors, sleep, custom
processors, routed-expert/sampling-mask outputs and ordinary KV memory budgets.

Cancellation arriving during a synchronous step is processed after GPU completion
and before output publication, following EngineCore's existing abort boundary.
Worker calls serialize under a lock, preventing cancellation from freeing sampler
storage. Responsive mid-kernel abort and asynchronous engine scheduling remain
outside this first binding.

## Native encoder streaming opt-in

Set `prefill_mode="encoder_stream"` explicitly to send each fresh text prompt
through the retained native encoder-stream/final-window replay path. Omission
keeps `full_target` for the comparison. Decode continues using full-target grants.
Recompute after tokens have already been emitted also uses bounded full-target
chunks; native prefix restoration and continuation streaming are not implied.

The scheduler issues one lane-0 grant containing **all N prompt tokens**, expected
committed end zero, absolute selected row `(N-1,)`, and `chunk_rows` equal to the
smaller of native `batch_tokens` and the vLLM scheduler token budget. The resulting
chunk size must be at least 80. Both contexts are exclusively owned by this grant
until final replay, sampler completion and native publication. A pending decode
step can run before the stream; it never shares the stream's contexts. Rotation
puts the next waiting fresh prefill ahead of the prior request's decode, then
ordinary two-context decode resumes. This first mode can delay unrelated decode
for the duration of a long prefill; it does not claim fine-grained fairness.

`num_scheduled_tokens` and `total_num_scheduled_tokens` report N, even when N is
larger than `max_num_batched_tokens`. That field is the native command's whole
token grant; `chunk_rows` separately bounds each internal GPU chunk. The Python
scheduler does not pretend N tokens were already computed. Native owns internal
encoder publication fences, suffix capture and layers20–39 replay over the final
128-token window. The selected last-row logits produce exactly one next token.
Only the completed full-N commit advances the request's computed end and emits
that token; partial acceptance is rejected before any native commit.

Native cancellation at any stream stage revokes the whole admission. The client
validates `revoked:true`; the runner retains this receipt so scheduler retirement
does not release the stale request a second time. A new admission gets a new
generation. A consumer-event timeout retains the borrow and both contexts until
drain succeeds. A failed execution/publication poisons the owner rather than
silently re-entering with partially published encoder state. Cached-token usage
remains zero: internal source publication is not a complete-model prefix hit.

The stream retains up to approximately 5 MiB of additional suffix CUDA storage.
Include this in the measured memory plan beyond the source payload budget.

## Remaining qualification

- Run GPU startup and strict C1/C2 numerical gates, including mixed request
  histories and top-k/logprob invariance. Expert repeatability and standalone
  native smoke results do not waive the complete vLLM gate.
- Run the complete vLLM encoder-stream GPU path and compare token/logprob quality
  and long-prefill throughput with the retained recipe and full-target path.
  dSpark requires its own accepted-prefix transaction integration afterward.
- Complete model checkpoint lifecycle before enabling prefix hits. Source pages
  alone omit windows, compressor/Engram history and replay state.
- Measure native memory, throughput and conservative admission's concurrency cost.

## CPU checks

`tests/v1/worker/test_native_target.py` runs 43 fake-backend cases. It executes the
actual Worker/EngineCore method bodies with CPU dependencies, tests constructor
ordering/counts, cache failure, result ownership, cancellation, retained timeouts,
publication errors and default-off behavior. Full GPU modules are not imported.
Together with the two startup reset checks and prior cache adapter/Rust metadata
contract suites, 72 tests pass
without skips. Use the retained uv environment and explicit pytest paths:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-project \
  --python .venv/bin/python .venv/bin/python -m pytest --noconftest \
  tests/v1/worker/test_native_target.py -q
```

`tests/v1/worker/test_native_target_serving.py` passes 30 cases using stdlib
`unittest` with real
Torch and vLLM request/output/sampling modules in retained image
`sha256:8041c897278b8372c15784d3a651c1cba689c24ccf6c12842ad3a7bf5b85abbe`.
The CPU container has no GPU/network, a two-CPU/four-GiB limit and only source/uv
mounts. Tests disable CUDA page locking and unwrap one compiled counting helper;
sampler logic is upstream. They cover the integrated scheduler→runner→sampler
loop, queue/credits, cached tokens zero, reset, cancellation, immutable logits,
sampling parameters, seeded RNG and top-logprob invariance.
Streaming cases also exercise whole-prompt versus chunk-budget accounting,
exclusive-prefill/two-lane-decode queue progression, final replay publication,
fresh generations, native cancellation receipts and lost-commit-ACK recovery.

No dependencies were installed. Retained Ruff supplies scoped lint/format checks;
full pre-commit is unavailable offline. The local environment lacks Torch/msgspec,
while the retained CPU container supplies both. These checks do not qualify live
CUDA leases or full API/EngineCore IPC; the selected uni worker itself does not
serialize grants through a second worker process.
