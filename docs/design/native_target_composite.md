# Native target composite-v1 candidate (21 September 2026 AEST)

This is an opt-in CPU-qualified packaging candidate, **not a serving release**.
Set `additional_config["afd_native_target"]["composite_drain_commit"] = true`
only in an explicitly selected candidate configuration. The optional key defaults
to false and rejects non-bools. An available native capability does not enable it.

`RetainedNativeBinding` reads each configuration at backend creation. Off selects
the original canonical `NativeTargetClient`; on selects the installed
`vllm_afd.native_target.composite.CompositeNativeTargetClient`. Initialization
negotiates optional composite-v1 support only when enabled. Missing/unsupported
capability retains the ordinary drain/commit path; missing Python adapter is an
explicit installation error, never a fixture fallback.

The reviewed R69 runner bytes are unchanged: one active ordinary grouped ticket
(any bounded valid member count), sampler validation before consumption,
all-lane drain barrier for multiple tickets, and original-command-handle recovery.
Single-family prefill/speculative paths and fresh bank/future queries remain
unchanged. Consumer memory proof never implies native owner health: quarantined
or cancelled poisoned owners remain failed/unusable. H1 is absent.

## Provenance

Base fork HEAD: `d222bc42da3105ef7aff7f5cae9a8efac6235079`.
Inspected working runner SHA `47d8b714d044cc40d039063db11b6e0ee7862452e1b7976ee6bed2fdbc40e25b`;
working binding SHA `ed2913b192e4a5672be2949c460d34ce071450dffafb2f172cb2c94cff539e97`.
The known-runtime binding overlay already included execution-poll interval support:
R69 original SHA `9373759c3a10f82e2d6d8e4eff4f987551dc9a008f99a44af0ca7e48d8c34331`;
R69 H2 input binding SHA `b491d93377a4dd0ed9ead64536075b6f41e29e651a48504f8fd6482ad53e0b56`.
Final runner SHA `eb5760506d45752370e855f2064f6b92eb93ad325381a70f18d2c52859ee2b45`;
final binding SHA `cd81f50e65533786e2affad382f02a99c310ec8c94812764d1f4d2e0f2237090`.
This explicitly attests the working-byte overlay, not HEAD alone.

CPU evidence lives under `/home/turq/.cache/afd-dsh-workers-20260919/serving-python-r70`:
290 tests and40 source/AST checks, including original25 fallback invariants.
Native CPU artifact SHA `d4158e2f35082c2d62d4646f10b7a3717aecbd3d893fc17fa75ae5e949dcfe77`
is from corrected source `dc781110d2ee008c9f980930338342d720267dd6` on actual a20db lineage.
It is distinct from earlier R69 CPU artifact94a9.

Tests import genuine candidate packages and real `ModelRunnerOutput`. The actual
getter/config validator/binding factory/runner `load_model` run with explicit
CPU-only client-constructor and sampler injections. Model snapshot/GPU creation,
full vLLM wheel building, numerical/model coherence, live performance and fleet
qualification are **not** tested. VLLM_TARGET_DEVICE=cpu and CPU Torch are mandatory
for this local suite. Source-checkout `_version` warning is expected.

Runtime attestation needs five untraced Python sources (sampler, runner, binding,
client, composite), or six when the quality observation module is enabled.
The quality overlay/observation pin is owned separately by the parent gate.
