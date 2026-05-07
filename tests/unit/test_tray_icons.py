"""tests/unit/test_tray_icons.py

Тесты для app/tray/icons.py:
- make_status_icon возвращает непустой QIcon для каждого статуса.
- LRU-кэш работает (hits >= 1 после второго вызова).
"""

from __future__ import annotations

import pytest

# NodeStatus берём из tray-модуля (fallback-логика там же)
from app.tray.icons import NodeStatus, make_status_icon, status_hex

# ---------------------------------------------------------------------------
# Skip-условие: системный трей недоступен
# ---------------------------------------------------------------------------

def _check_tray_available() -> None:
    from PySide6.QtWidgets import QSystemTrayIcon

    if not QSystemTrayIcon.isSystemTrayAvailable():
        pytest.skip("System tray not available in this environment")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _require_tray() -> None:
    _check_tray_available()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", list(NodeStatus))
def test_make_status_icon_returns_non_null(qapp: object, status: NodeStatus) -> None:
    """make_status_icon должна возвращать непустой QIcon для каждого статуса."""
    icon = make_status_icon(status)
    assert not icon.isNull(), f"Icon for status={status!r} is null"


def test_make_status_icon_cached(qapp: object) -> None:
    """Второй вызов с теми же аргументами должен давать cache hit."""
    # Сбрасываем кэш чтобы тест был изолирован
    make_status_icon.cache_clear()

    make_status_icon(NodeStatus.ONLINE, 22)
    make_status_icon(NodeStatus.ONLINE, 22)

    info = make_status_icon.cache_info()
    assert info.hits >= 1, f"Expected at least 1 cache hit, got {info.hits}"


@pytest.mark.parametrize("status", list(NodeStatus))
def test_status_hex_returns_non_empty_string(status: NodeStatus) -> None:
    """status_hex должна возвращать непустую строку hex-цвета."""
    hex_val = status_hex(status)
    assert hex_val.startswith("#"), f"Expected hex string, got {hex_val!r}"
    assert len(hex_val) in (7, 9), f"Unexpected hex length: {hex_val!r}"


def test_make_status_icon_custom_size(qapp: object) -> None:
    """make_status_icon с кастомным size_px не должна падать."""
    icon = make_status_icon(NodeStatus.ONLINE, 32)
    assert not icon.isNull()


def test_make_status_icon_different_statuses_cached_separately(qapp: object) -> None:
    """Кэш должен хранить разные иконки для разных статусов."""
    make_status_icon.cache_clear()

    icon_online = make_status_icon(NodeStatus.ONLINE, 22)
    icon_error = make_status_icon(NodeStatus.ERROR, 22)

    # Иконки должны быть разными объектами
    assert icon_online is not icon_error
    # После двух разных статусов — 0 hits (первые вызовы)
    info = make_status_icon.cache_info()
    assert info.misses >= 2
