---
name: experiments-rules
description: Applies when editing files under experiments/.
applies_to: ["experiments/**"]
---

# Rules for `experiments/**`

- Use `/new-experiment <slug>` to create a new experiment folder. Do not
  hand-roll the layout.
- Every experiment folder has, at minimum: `README.md`, `config.yaml`,
  `notes.qmd`, `results/`, `log.md`, `metrics.json`.
- `README.md` frontmatter must set `status` to one of:
  `running | done | abandoned`. Update it when the state changes.
- Destructive or long-running runs happen in a Git worktree under
  `.worktrees/`, never in the primary checkout.
- Metrics go to `metrics.json` so DVC can track them. Plots go under
  `results/`.
- When an experiment finishes, write the `result:` frontmatter field
  (short sentence) and update `_meta/index.md` to remove it from
  "Active experiments".
