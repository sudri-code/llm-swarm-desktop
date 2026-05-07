"""tests/unit/test_build_qss_source_resolution.py

Tests for _resolve_tokens_source() fallback logic in tools/build_qss.py:
  1. webclient-live path is chosen when ../llm-swarm-webclient exists.
  2. vendor-snapshot is chosen when webclient-live is absent.
  3. RuntimeError is raised when both are absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS_DIR = REPO_ROOT / "tools"


def _load_build_qss() -> Any:
    """Load tools/build_qss.py as a fresh module instance (isolated from cache)."""
    spec_path = TOOLS_DIR / "build_qss.py"
    spec = importlib.util.spec_from_file_location("build_qss_src_resolution", spec_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass resolution works.
    sys.modules["build_qss_src_resolution"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def build_qss() -> Any:
    return _load_build_qss()


# ---------------------------------------------------------------------------
# _resolve_tokens_source() — fallback chain
# ---------------------------------------------------------------------------


def test_resolve_prefers_webclient_live(
    build_qss: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When both webclient-live and vendor-snapshot exist, webclient-live is chosen."""
    live = tmp_path / "live" / "tokens.css"
    live.parent.mkdir(parents=True)
    live.write_text(":root {}", encoding="utf-8")

    vendor = tmp_path / "vendor" / "tokens.css"
    vendor.parent.mkdir(parents=True)
    vendor.write_text(":root {}", encoding="utf-8")

    monkeypatch.setattr(build_qss, "_TOKENS_CSS_WEBCLIENT_LIVE", live)
    monkeypatch.setattr(build_qss, "_TOKENS_CSS_VENDOR_SNAPSHOT", vendor)

    path, label = build_qss._resolve_tokens_source()
    assert path == live
    assert label == "webclient-live"


def test_resolve_falls_back_to_vendor_snapshot(
    build_qss: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When webclient-live is absent, vendor-snapshot is chosen."""
    live = tmp_path / "nonexistent" / "tokens.css"  # does NOT exist

    vendor = tmp_path / "vendor" / "tokens.css"
    vendor.parent.mkdir(parents=True)
    vendor.write_text(":root {}", encoding="utf-8")

    monkeypatch.setattr(build_qss, "_TOKENS_CSS_WEBCLIENT_LIVE", live)
    monkeypatch.setattr(build_qss, "_TOKENS_CSS_VENDOR_SNAPSHOT", vendor)

    path, label = build_qss._resolve_tokens_source()
    assert path == vendor
    assert label == "vendor-snapshot"


def test_resolve_raises_when_both_absent(
    build_qss: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RuntimeError is raised with a helpful message when neither source exists."""
    live = tmp_path / "no_live" / "tokens.css"
    vendor = tmp_path / "no_vendor" / "tokens.css"

    monkeypatch.setattr(build_qss, "_TOKENS_CSS_WEBCLIENT_LIVE", live)
    monkeypatch.setattr(build_qss, "_TOKENS_CSS_VENDOR_SNAPSHOT", vendor)

    with pytest.raises(RuntimeError, match="make sync-tokens"):
        build_qss._resolve_tokens_source()


def test_resolve_error_message_mentions_webclient(
    build_qss: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RuntimeError message mentions llm-swarm-webclient as a hint."""
    live = tmp_path / "no_live" / "tokens.css"
    vendor = tmp_path / "no_vendor" / "tokens.css"

    monkeypatch.setattr(build_qss, "_TOKENS_CSS_WEBCLIENT_LIVE", live)
    monkeypatch.setattr(build_qss, "_TOKENS_CSS_VENDOR_SNAPSHOT", vendor)

    with pytest.raises(RuntimeError, match="llm-swarm-webclient"):
        build_qss._resolve_tokens_source()
