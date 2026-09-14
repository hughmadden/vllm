# AFD branch — afd/deepseek-v41-dflash2

Base: upstream `main` @ `4be3dcf0fc7a9086d978eaea3c5a4d78fb98e44a` (2026-09-14).
Purpose: attention–FFN disaggregation for DeepSeek-V4.1-Flash (GLM-5.3 lane
later), with DSpark/DFlash2-family speculative decoding on the coordinator —
porting the design concepts of `tpurtell/ds41rt` onto this tree's seams
(`PluggableLayer.register_oot` / `RoutedExperts.forward_modular`,
`vllm.models.deepseek_v41`, `--additional-config`).

The implementation package lives in its own repo (kept out of this tree until
P6 in-tree binding): **https://github.com/hughmadden/vllm-afd** (private;
274 CPU-safe tests; protocol v2 byte-exact vs ds41rt golden vectors).

Campaign knowledge repo: `pg:/home/turq/dev/recipes/vllm-afd-port/`
(DESIGN.md, protocol-v2-spec.md, HANDOFF.md with the PO-gated S0–S4 fleet plan).

This branch currently carries NO changes to vllm proper — it pins the base and
records the plan. In-tree patches land in phase P6 (loader-filter binding,
breakable-graph eager breaks, spec-decode proposer binding).
