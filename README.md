# dotaml-research

Personal research project. See `CLAUDE.md` for the agent-facing orientation
and `~/.claude/CLAUDE.md` for the framework's durable principles.

## Quick start

```sh
make env       # uv sync
make lint      # orphan/dead-link check
```

## Layout

- `raw/` — immutable sources (papers, repos, web captures).
- `literature/` — processed notes.
- `concepts/` / `mocs/` — knowledge graph.
- `experiments/` — runs, each dated and slugged.
- `docs/decisions/` — ADRs.
- `journal/` — per-session log.
- `_meta/` — index, log, templates.
