.PHONY: env lint repro clean

env:
	uv sync

lint:
	@echo "Run the /lint skill in Claude Code for full checks."
	@grep -rIl --include='*.md' '\[\[' . | head -20 || true

repro:
	dvc repro

clean:
	rm -rf .venv __pycache__ .pytest_cache
