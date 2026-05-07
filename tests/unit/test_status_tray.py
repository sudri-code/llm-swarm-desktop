"""tests/unit/test_status_tray.py

Тесты для app/tray/status_tray.py:
- Начальное состояние после __init__.
- set_status обновляет tooltip.
- pause_requested(PauseMode.GRACEFUL) эмитируется при trigger graceful action.
- hard action эмитит hard_pause_confirmation_requested, НЕ pause_requested.
- resume_requested эмитируется после set_can_pause(False).
- quit_requested эмитируется.
- Запрещённые термины отсутствуют в тексте actions.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QSystemTrayIcon

# NodeStatus и PauseMode берём из state
from app.state import PauseMode
from app.tray.status_tray import NodeStatus, StatusTray

# ---------------------------------------------------------------------------
# Skip-условие
# ---------------------------------------------------------------------------

def _check_tray_available() -> None:
    if not QSystemTrayIcon.isSystemTrayAvailable():
        pytest.skip("System tray not available in this environment")


@pytest.fixture(autouse=True)
def _require_tray() -> None:
    _check_tray_available()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tray(qapp: object) -> StatusTray:
    return StatusTray()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_status_tray_initial_state(tray: StatusTray) -> None:
    """После создания: tooltip непустой, иконка установлена.

    Pause submenu enabled, resume action disabled (начальный статус OFFLINE).
    """
    icon = tray.tray_icon.icon()
    assert not icon.isNull(), "Tray icon should be set after __init__"

    tooltip = tray.tray_icon.toolTip()
    assert tooltip, "Tooltip should be non-empty after __init__"

    # Начальный статус OFFLINE → pause submenu enabled, resume disabled
    assert tray.pause_menu.isEnabled(), "Pause submenu should be enabled initially"
    assert not tray.action_resume.isEnabled(), "Resume should be disabled initially"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# set_status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("status", "expected_word"),
    [
        (NodeStatus.ONLINE, "online"),
        (NodeStatus.DEGRADED, "degraded"),
        (NodeStatus.OFFLINE, "offline"),
        (NodeStatus.ERROR, "error"),
    ],
)
def test_set_status_updates_icon_and_tooltip(
    tray: StatusTray,
    status: NodeStatus,
    expected_word: str,
) -> None:
    """set_status должен обновлять иконку и tooltip с ожидаемым словом."""
    tray.set_status(status)

    icon = tray.tray_icon.icon()
    assert not icon.isNull(), f"Icon should not be null after set_status({status!r})"

    tooltip = tray.tray_icon.toolTip()
    assert expected_word in tooltip, (
        f"Tooltip {tooltip!r} should contain {expected_word!r} for status {status!r}"
    )


# ---------------------------------------------------------------------------
# Pause signals
# ---------------------------------------------------------------------------

def test_pause_emits_signal_graceful(tray: StatusTray, qtbot: object) -> None:
    """Graceful action должен эмитить pause_requested(PauseMode.GRACEFUL)."""
    received: list[PauseMode] = []
    tray.pause_requested.connect(received.append)

    tray._action_graceful.trigger()

    assert received == [PauseMode.GRACEFUL], (
        f"Expected [PauseMode.GRACEFUL], got {received}"
    )


def test_hard_action_emits_confirmation_requested_not_pause(
    tray: StatusTray, qtbot: object
) -> None:
    """Hard action должен эмитить hard_pause_confirmation_requested, НЕ pause_requested.

    Предупреждение о strikes — в диалоге MainWindow, не в прямом emit pause_requested.
    """
    pause_received: list[PauseMode] = []
    confirm_received: list[None] = []

    tray.pause_requested.connect(pause_received.append)
    tray.hard_pause_confirmation_requested.connect(lambda: confirm_received.append(None))

    tray._action_hard.trigger()

    assert pause_received == [], (
        "Hard action must NOT emit pause_requested directly; "
        f"got {pause_received}"
    )
    assert len(confirm_received) == 1, (
        "Hard action must emit hard_pause_confirmation_requested once"
    )


def test_hard_action_text_warns_strikes(tray: StatusTray) -> None:
    """Текст hard-action обязан содержать предупреждение о strikes (blocker review)."""
    text = tray._action_hard.text()
    assert "strikes" in text.lower(), (
        f"Hard action text must warn about strikes, got: {text!r}"
    )


# ---------------------------------------------------------------------------
# Resume signal
# ---------------------------------------------------------------------------

def test_resume_emits_signal(tray: StatusTray, qtbot: object) -> None:
    """После set_can_pause(False) resume action enabled и эмитит resume_requested."""
    tray.set_can_pause(False)

    assert tray.action_resume.isEnabled(), "Resume should be enabled after set_can_pause(False)"  # type: ignore[union-attr]

    received: list[None] = []
    tray.resume_requested.connect(lambda: received.append(None))

    tray._action_resume.trigger()

    assert len(received) == 1, "resume_requested should have been emitted once"


# ---------------------------------------------------------------------------
# set_can_pause semantics
# ---------------------------------------------------------------------------

def test_set_can_pause_true_enables_pause_disables_resume(tray: StatusTray) -> None:
    """set_can_pause(True) → pause enabled, resume disabled."""
    tray.set_can_pause(True)
    assert tray.pause_menu.isEnabled()
    assert not tray.action_resume.isEnabled()  # type: ignore[union-attr]


def test_set_can_pause_false_disables_pause_enables_resume(tray: StatusTray) -> None:
    """set_can_pause(False) → pause disabled, resume enabled."""
    tray.set_can_pause(False)
    assert not tray.pause_menu.isEnabled()
    assert tray.action_resume.isEnabled()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Quit signal
# ---------------------------------------------------------------------------

def test_quit_emits_signal(tray: StatusTray, qtbot: object) -> None:
    """Quit action должен эмитить quit_requested."""
    received: list[None] = []
    tray.quit_requested.connect(lambda: received.append(None))

    tray._action_quit.trigger()

    assert len(received) == 1, "quit_requested should have been emitted once"


# ---------------------------------------------------------------------------
# No forbidden terms
# ---------------------------------------------------------------------------

_FORBIDDEN_TERMS = [
    "Billing",
    "billing",
    "top-up",
    "credits",
    "кредиты",
    "subscription",
    "тариф",
    "оплата",
    "Stripe",
    "invoice",
    "refund",
]


def _collect_all_action_texts(tray: StatusTray) -> list[str]:
    """Собрать тексты всех actions из контекстного меню рекурсивно."""
    texts: list[str] = []
    menu = tray.tray_icon.contextMenu()
    if menu is None:
        return texts

    def _walk(m: object) -> None:
        from PySide6.QtWidgets import QMenu
        assert isinstance(m, QMenu)
        for action in m.actions():
            texts.append(action.text())
            sub = action.menu()
            if sub is not None:
                _walk(sub)

    _walk(menu)
    return texts


@pytest.mark.parametrize("term", _FORBIDDEN_TERMS)
def test_no_forbidden_terms(tray: StatusTray, term: str) -> None:
    """Тексты actions не должны содержать запрещённых money-терминов."""
    texts = _collect_all_action_texts(tray)
    for text in texts:
        assert term not in text, (
            f"Forbidden term {term!r} found in action text: {text!r}"
        )


# ---------------------------------------------------------------------------
# show_window_requested via show_window action
# ---------------------------------------------------------------------------

def test_show_window_requested_via_open_action(tray: StatusTray) -> None:
    """'Open llm-swarm' action эмитит show_window_requested."""
    received: list[None] = []
    tray.show_window_requested.connect(lambda: received.append(None))

    tray._action_open.trigger()

    assert len(received) == 1, "show_window_requested should have been emitted once"
