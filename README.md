# dotaml-turbo

**Pre-game ML modeling of Dota 2 Turbo matches — win prediction and a foundation-model arc, under hidden consistent evaluation.**

📊 **[Browse experiments & results →](https://eschmitt88.github.io/dotaml-turbo/)** —
an interactive, always-live view of every dated experiment, its metrics, and the
concept/literature graph.

## What this is

Predicting `P(radiant_win | pre-game info)` for Dota 2 Turbo (game_mode=23) on a
frozen patch-7.40 snapshot — plus a family of auxiliary models (duration curves,
item recommendation, learned player representations, anonymity-aware variants)
probing which feature / architecture / supervision axes move the ceiling.

The experiment line traces a foundation-model arc: from supervised baselines to
masked-multitask pretraining that serves many downstream queries from a single
encoder.

### The interesting constraints

- **Hidden drafts** — enemy picks aren't visible pre-game, so no draft-order features.
- **~66% anonymous players** (`account_id ∈ {0, 4294967295}`) — a hard limit on player-identity features.
- **Held-out discipline (HCE)** — a chronologically sealed test window is off-limits
  during the search loop; the validation split is the only search signal
  (`splits.yaml` is the authority).

## How it's organized

- `experiments/YYYY-MM-DD-<slug>/` — self-contained runs: hypothesis → result,
  `config.yaml`, `metrics.json`, notebook, log.
- `concepts/` / `mocs/` — the knowledge graph behind the modeling choices.
- `literature/` — processed notes on the papers that informed the architecture.
- `raw/` — immutable sources · `docs/decisions/` — ADRs · `_meta/` — index, log, templates.

Large match data (~175 GB) is DVC-tracked off-repo; git holds code, configs, notes,
and metrics. The [browsable site](https://eschmitt88.github.io/dotaml-turbo/) renders
the experiment table and graph live — no build step.

## Local use

```sh
make env    # uv sync
make lint   # knowledge-graph + experiment health check
```

Part of a personal research framework
([claude-system](https://github.com/eschmitt88/claude-system)). See `CLAUDE.md` for
the agent-facing orientation and `~/.claude/CLAUDE.md` for the framework's durable
principles.
