.PHONY: help install dev lint test tokens vendor-sync

help:
	@echo "Targets: install dev lint test tokens vendor-sync"

install:
	uv sync --all-extras

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

vendor-sync:
	uv run python tools/sync_vendor.py
