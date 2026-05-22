# Project: dotaml-turbo

Short orientation only. User-level `~/.claude/CLAUDE.md` holds the durable
principles; this file refines them for this project.

## What this project is about

Pre-game ML modeling of **Dota 2 Turbo** (game_mode=23) matches on a
patch-7.40 frozen snapshot under HCE. Primary metric is win prediction
(`P(radiant_win | pre-game info)`), but the project also produces and
explores auxiliary models — duration curves, item recommendation,
learned player representations, anonymity-aware variants — and tests
which feature / architecture / supervision axes move the ceiling.

Live state pointers (read these first, not this file):

- [NOTES.md](NOTES.md) — day-by-day Did / Findings / Next
- [concepts/draft-prediction-plateau.md](concepts/draft-prediction-plateau.md) — running scoreboard + current ceiling
- [_meta/index.md](_meta/index.md) — completed experiments

## HCE

This project opts into the user-global HCE rule (`splits.yaml` exists).
Test window **[2026-03-10, 2026-03-23]** is sealed. `metrics.json` is the
val signal; `final_metrics.json` is only written by an explicit
final-scoring pass. Search-phase skills MUST NOT read `test/` paths or
any row whose `start_time_date` falls in that window.

## Turbo-specific facts (don't repeat past mistakes)

- **Drafts are hidden.** Turbo doesn't show enemy picks during the
  draft, so draft-order / pick-sequence / response-pick features are
  invalid — the model has only the final 10-hero composition pre-game.
- **~66% of player slots are anonymous** (`account_id` ∈ {0, 4294967295}).
  This is the binding constraint on player-identity-based features.
- Match payloads in `data/snapshots/.../raw/turbo/` carry full Steam Web
  API match-details `raw_json` fields: duration, picks_bans (post-game
  order), per-player items / KDA / GPM / XPM / hero_damage,
  ability_upgrades with timestamps. 175 GB across patch-7.40 +
  pre-patch history.

## Layout (see user CLAUDE.md for the full rationale)

- `raw/`, `literature/`, `concepts/`, `mocs/`
- `experiments/YYYY-MM-DD-<slug>/`, `docs/decisions/`
- `journal/` (hook-written), `_meta/` (index, log, templates)

## Scoped rules

@.claude/rules/experiments.md
@.claude/rules/notebooks.md
@.claude/rules/data.md

## Budget & compute

@budget.yaml

Multi-hour runs need to fit under `budget.yaml` ceilings. If not, flag in
proposal `risks:` and either scope down or raise the ceiling.

## Environment & hardware

- Python 3.12 + `uv`-managed `.venv/` at project root. torch 2.9.1+cu128,
  pyarrow 24+, LightGBM.
- Box: Ryzen 9950X + RTX 5080 + 96 GB **non-ECC** DDR5 at **JEDEC 4800
  MT/s** — DO NOT re-enable EXPO (see memory note
  `aiserver2026-ram-bitflips-root-cause`).
- DVC tracks data + model checkpoints. Large artifacts on SN850X via
  `/mnt/projects/`. Never put them under `~/`.

## Gotchas learned the hard way

- **Use `python -u`** for nohup-detached training runs. Default stdout
  is block-buffered (~8 KB); a 30-epoch run produces <6 KB of progress
  output, so silent training looks like a hang.
- **Don't re-read a multi-GB parquet** inside a process already holding
  heavy aggregator state — OOM-kills the process and can cascade to a
  hard system reboot. Use pyarrow row-group statistics for validation.

## Housekeeping

- End sessions with `/wrap`. SessionEnd hook backstops.
- `/new-experiment <slug>` — don't hand-roll the layout.
- `/propose` → `/implement` is the canonical experiment cycle.
- `/lint` weekly.
