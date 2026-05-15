---
kind: concept
name: "fake-match-filtering"
status: seedling
added: "2026-05-15"
sources:
  - literature/repos/eschmitt88-DotaML.md
related_concepts:
  - draft-only-win-prediction
related_experiments: []
tags: [data-quality, label-noise, dota2]
---

# fake-match-filtering

## Definition

Two heuristics used by the prior-art DotaML repo to remove matches whose
outcomes were pre-arranged (boosting services, behavior-score recovery,
quest farming) from the training corpus:

1. **Forfeit filter.** A match where the losing team's both Tier-4 towers
   are still standing (bits 9 and 10 of `tower_status_radiant` or
   `tower_status_dire`, depending on `radiant_win`). Real matches almost
   never end with both T4s of the loser intact; "gg" surrender after 30
   minutes does.
2. **Empty-inventory filter.** More than two players in the match have
   zero items across all six inventory slots. Real players, even on the
   losing team, buy items.

## Why it matters here

About 10,000 matches with identical hero compositions where one team
always won were found in the early DotaML dataset and traced to a
boosting service. These act as adversarial label noise: identical
inputs map to deterministic outputs, but the determinism reflects
the service's revenue model, not the draft.

For `dotaml-turbo`, applying both filters before training is the cost-
free win. The filter is computable from fields already present in
`raw_json` and removes a known confound that would otherwise overfit
any sufficiently flexible model.

## Connections

- [[draft-only-win-prediction]] — the task whose label noise this
  concept addresses.
- Implementation recipes (Python) are reproduced verbatim in the
  source literature note.
- Open: are there other pre-arrangement signatures? Account-id
  clustering, suspiciously fast durations, abnormal first-blood times.
  Worth a follow-up data-audit experiment.
