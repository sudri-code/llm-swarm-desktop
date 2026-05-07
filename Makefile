.PHONY: help install dev lint test tokens vendor-sync sync-tokens sync-all

help:
	@echo "Targets: install dev lint test tokens vendor-sync sync-tokens sync-all"

install:
	uv sync --all-extras
	$(MAKE) tokens

dev:
	uv run python -m app.main

lint:
	uv run ruff check .
	uv run pyright

test:
	uv run pytest; ec=$$?; [ $$ec -eq 0 ] || [ $$ec -eq 5 ]

tokens:
	uv run python tools/build_qss.py
	@echo "tokens target complete (pyside6-rcc invoked by build_qss.py if fonts present)"

# Sync node/+shared/ from ../llm-swarm (requires local checkout of ../llm-swarm)
vendor-sync:
	uv run python tools/sync_vendor.py --target swarm

# Sync vendor/tokens.css from ../llm-swarm-webclient (requires local checkout)
sync-tokens:
	uv run python tools/sync_vendor.py --target webclient

# Sync both swarm code and webclient tokens in one step
sync-all:
	uv run python tools/sync_vendor.py --target all
