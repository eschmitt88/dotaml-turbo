# v7-unified-masked-multitask-740 log

## 2026-05-26 scaffold

- Forked v4 codebase (`experiments/2026-05-25-v4-iso-teambias-extended-740/`).
- Added 8 per-slot maskable input groups + 2 per-match maskable groups
  with learned mask embeddings.
- Separated K/D/A into three heads (10 task tokens each, 30 new task tokens).
- Switched duration head to scalar regression with `Linear(1, d_model)` input projection.
- Implemented `ScenarioSampler` (9 scenarios) and `ProbeSuite` (9 probes).
- Items pool: SUM(item_input_embed[id_in_bag]) / sqrt(K), implemented as
  matmul of multi-hot [B,10,V] by embedding weight [V,d_model] for efficiency.
- Per-scenario loss weights modulate per-head losses pre-sum; all 8 heads
  always computed regardless of scenario (multi-task anchor).
- Adaptive sampling re-normalizes after every probe-suite pass; cap at
  2x initial, floor at 0.5x initial.
- Smoke + profile pending.
