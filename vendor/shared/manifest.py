"""Утилиты для manifest chunk registry (Stage 5).

Вынесено из tracker/chunk_registry.py как shared helper (followup m1, wave 2 code-review).
Используется:
  - tracker/chunk_registry.py: верификация manifest_sha256 при публикации.
  - scripts/build_manifest.py: вычисление sha256 перед POST /api/v1/models.
"""

from __future__ import annotations

import hashlib
import json

from shared.protocol import ChunkSpec


def compute_manifest_sha256(chunks: list[ChunkSpec]) -> str:
    """SHA256 от canonical JSON списка ChunkSpec[] (ADR-0030).

    Canonical JSON: sort_keys=True, сепараторы (",", ":"), UTF-8.
    Chunks сортируются по ord перед сериализацией для детерминизма.

    Args:
        chunks: список ChunkSpec в любом порядке.

    Returns:
        hex-строка SHA256 (64 символа).
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.ord)
    data = [c.model_dump(mode="json") for c in sorted_chunks]
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
