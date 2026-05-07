"""tests/unit/test_main_window.py — тесты главного окна и enum NodeStatus."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QStackedWidget, QWidget

from app.state import NodeStatus, status_color  # noqa: F401 — used in tests
from app.styles.tokens import COLORS

# ---------------------------------------------------------------------------
# Headless guard
# ---------------------------------------------------------------------------

def _platform_name() -> str:
    app = QGuiApplication.instance()
    if app is None:
        return ""
    return app.platformName()  # type: ignore[attr-defined]


def _skip_if_headless() -> None:
    name = _platform_name()
    if name and name not in ("xcb", "cocoa", "windows", "offscreen", "wayland"):
        pytest.skip(f"Unsupported platform for GUI tests: {name!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_widget_texts(root: QWidget) -> list[str]:
    """Рекурсивно собирает все текстовые строки из дерева виджетов."""
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel

    texts: list[str] = []

    def _walk(w: QWidget) -> None:
        if isinstance(w, (QLabel, QAbstractButton)):
            texts.append(w.text())
        elif isinstance(w, QGroupBox):
            texts.append(w.title())
        title = w.windowTitle()
        if title:
            texts.append(title)
        for child in w.children():
            if isinstance(child, QWidget):
                _walk(child)

    _walk(root)
    return texts


# ---------------------------------------------------------------------------
# Tests: NodeStatus и status_color
# ---------------------------------------------------------------------------

def test_status_color_mapping() -> None:
    """status_color возвращает правильный hex для каждого статуса."""
    assert status_color(NodeStatus.ONLINE) == COLORS.OK
    assert status_color(NodeStatus.DEGRADED) == COLORS.WARN
    assert status_color(NodeStatus.OFFLINE) == COLORS.FG_3
    assert status_color(NodeStatus.ERROR) == COLORS.ERR


def test_node_status_values() -> None:
    """NodeStatus содержит ожидаемые строковые значения."""
    assert NodeStatus.ONLINE == "online"
    assert NodeStatus.DEGRADED == "degraded"
    assert NodeStatus.OFFLINE == "offline"
    assert NodeStatus.ERROR == "error"


# ---------------------------------------------------------------------------
# Tests: MainWindow
# ---------------------------------------------------------------------------

def test_main_window_creates_without_error(qtbot: pytest.FixtureRequest) -> None:
    """Главное окно создаётся без исключений."""
    _skip_if_headless()

    from app.windows.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    assert window.windowTitle() == "llm-swarm"

    stack = window.findChild(QStackedWidget)
    assert stack is not None
    assert stack.count() == 5


def test_status_banner_set_status(qtbot: pytest.FixtureRequest) -> None:
    """StatusBanner корректно обновляет property и цвет точки для всех статусов."""
    _skip_if_headless()

    from app.windows.main_window import MainWindow, StatusBanner

    window = MainWindow()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    banner: StatusBanner = window.status_banner

    status_color_pairs = [
        (NodeStatus.ONLINE, COLORS.OK),
        (NodeStatus.DEGRADED, COLORS.WARN),
        (NodeStatus.OFFLINE, COLORS.FG_3),
        (NodeStatus.ERROR, COLORS.ERR),
    ]

    for status, expected_color in status_color_pairs:
        banner.set_status(status, f"Test: {status.value}")

        assert banner.property("status") == status.value, (
            f"status property should be {status.value!r}"
        )

        dot_style = banner._dot.styleSheet()
        assert expected_color.lower() in dot_style.lower(), (
            f"Dot styleSheet should contain {expected_color} for status={status.value}, "
            f"got: {dot_style!r}"
        )


def test_sidebar_navigation(qtbot: pytest.FixtureRequest) -> None:
    """Переключение сайдбара меняет активный экран и эмитит сигнал."""
    _skip_if_headless()

    from app.windows.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    emitted: list[str] = []
    window.current_screen_changed.connect(emitted.append)

    # Переключаемся на индекс 2 — "balance"
    window.sidebar.setCurrentRow(2)

    assert window.stack.currentIndex() == 2
    assert len(emitted) >= 1
    assert emitted[-1] == "balance"


def test_sidebar_balance_label(qtbot: pytest.FixtureRequest) -> None:
    """Сайдбар на индексе 2 должен показывать 'Balance', а не 'Earnings'."""
    _skip_if_headless()

    from app.windows.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    item = window.sidebar.item(2)
    assert item is not None
    assert item.text() == "Balance", (
        f"Expected sidebar[2]='Balance', got {item.text()!r}"
    )


def test_confirm_hard_pause_returns_true_on_accept(
    qtbot: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """confirm_hard_pause() возвращает True когда пользователь нажал «Отключить»."""
    _skip_if_headless()

    from unittest.mock import MagicMock

    from app.windows.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    # Используем MagicMock: автоматически создаёт атрибуты (Icon.Warning, ButtonRole и т.п.)
    confirm_btn = object()
    mock_box = MagicMock()
    mock_box.clickedButton.return_value = confirm_btn

    # addButton возвращает confirm_btn для label с "Отключить", иначе другой объект
    def _add_button(label: str, role: object) -> object:
        if "Отключить" in label:
            return confirm_btn
        return object()

    mock_box.addButton.side_effect = _add_button

    MockQMessageBox = MagicMock(return_value=mock_box)
    # Копируем реальные enum-атрибуты чтобы `QMessageBox.Icon.Warning` работало
    from PySide6.QtWidgets import QMessageBox as _RealQMB
    MockQMessageBox.Icon = _RealQMB.Icon
    MockQMessageBox.ButtonRole = _RealQMB.ButtonRole

    monkeypatch.setattr("app.windows.main_window.QMessageBox", MockQMessageBox)

    result = window.confirm_hard_pause()
    assert result is True


def test_confirm_hard_pause_returns_false_on_cancel(
    qtbot: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """confirm_hard_pause() возвращает False когда пользователь нажал «Отмена»."""
    _skip_if_headless()

    from unittest.mock import MagicMock

    from PySide6.QtWidgets import QMessageBox as _RealQMB

    from app.windows.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    cancel_btn = object()
    mock_box = MagicMock()
    mock_box.clickedButton.return_value = cancel_btn  # вернули кнопку «Отмена»

    def _add_button(label: str, role: object) -> object:
        if "Отмена" in label:
            return cancel_btn
        return object()

    mock_box.addButton.side_effect = _add_button

    MockQMessageBox = MagicMock(return_value=mock_box)
    MockQMessageBox.Icon = _RealQMB.Icon
    MockQMessageBox.ButtonRole = _RealQMB.ButtonRole

    monkeypatch.setattr("app.windows.main_window.QMessageBox", MockQMessageBox)

    result = window.confirm_hard_pause()
    assert result is False


def test_no_forbidden_terms(qtbot: pytest.FixtureRequest) -> None:
    """Ни один виджет в дереве не содержит запрещённых терминов."""
    _skip_if_headless()

    from app.windows.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    forbidden = [
        "Billing",
        "top-up",
        "credits",
        "subscription",
        "тариф",
        "оплата",
    ]

    all_texts = _collect_widget_texts(window)

    for text in all_texts:
        for term in forbidden:
            assert term.lower() not in text.lower(), (
                f"Forbidden term {term!r} found in widget text: {text!r}"
            )
