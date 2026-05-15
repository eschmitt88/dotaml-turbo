---
name: notebooks-rules
description: Applies when editing notebooks (*.qmd, *.ipynb).
applies_to: ["**/*.qmd", "**/*.ipynb", "notebooks/**"]
---

# Rules for notebooks

- Prefer Quarto (`.qmd`) over `.ipynb`. Diffs are readable; cells are
  plain text.
- Notebooks live inside their experiment folder as `notes.qmd`, not in a
  top-level `notebooks/` dump.
- Keep notebook state reproducible from `config.yaml` + the tracked data.
  Don't hide mutations inside a notebook.
- Outputs (plots, tables) go under the experiment's `results/` — not
  embedded in the notebook as base64 blobs when they're load-bearing.
- `.ipynb_checkpoints/` and `_files/` directories are gitignored; don't
  unignore them.
