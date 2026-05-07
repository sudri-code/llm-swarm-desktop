"""tests/integration/test_app_bootstrap.py — smoke-тест bootstrap'а приложения.

Проверяет, что:
- build_app() создаёт MainWindow и StatusTray без ошибок.
- Сигналы трея корректно подключены к окну.
- set_status() меняет иконку (иконка ненулевая после изменения статуса).
- Нет дублирующих event loop'ов — qasync не запускается, только объектный граф.

Запуск в headless-режиме: loop.run_forever() не вызывается.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from app.main import build_app
from app.state import NodeStatus, PauseMode
from app.tray.status_tray import StatusTray
from app.windows.main_window import MainWindow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_components(qapp: QApplication) -> tuple[MainWindow, StatusTray]:
    """Создать MainWindow + StatusTray через build_app().

    qapp уже предоставлен pytest-qt; не создаём второй QApplication.
    build_app() принимает QApplication — передаём qapp напрямую.
    """
    window, tray = build_app(qapp)
    return window, tray


# ---------------------------------------------------------------------------
# Smoke: объекты создаются
# ---------------------------------------------------------------------------


def test_build_app_returns_main_window(app_components: tuple[MainWindow, StatusTray]) -> None:
    """build_app() должен вернуть MainWindow."""
    window, _ = app_components
    assert isinstance(window, MainWindow), f"Expected MainWindow, got {type(window)}"


def test_build_app_returns_status_tray(app_components: tuple[MainWindow, StatusTray]) -> None:
    """build_app() должен вернуть StatusTray."""
    _, tray = app_components
    assert isinstance(tray, StatusTray), f"Expected StatusTray, got {type(tray)}"


def test_main_window_has_correct_title(app_components: tuple[MainWindow, StatusTray]) -> None:
    """Заголовок главного окна — 'llm-swarm'."""
    window, _ = app_components
    assert window.windowTitle() == "llm-swarm"


def test_main_window_minimum_size(app_components: tuple[MainWindow, StatusTray]) -> None:
    """Минимальный размер окна не меньше 880×600."""
    window, _ = app_components
    min_size = window.minimumSize()
    assert min_size.width() >= 880
    assert min_size.height() >= 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_no_tray() -> None:
    """Пропустить тест если системный трей недоступен в текущей среде."""
    if not QSystemTrayIcon.isSystemTrayAvailable():
        pytest.skip("System tray not available in this environment")


# ---------------------------------------------------------------------------
# Tray: иконка и видимость
# ---------------------------------------------------------------------------


def test_tray_icon_not_null_after_build(app_components: tuple[MainWindow, StatusTray]) -> None:
    """После build_app() иконка трея установлена и ненулевая."""
    _skip_if_no_tray()
    _, tray = app_components
    icon = tray.tray_icon.icon()
    assert not icon.isNull(), "Tray icon should be non-null after build_app()"


def test_tray_set_status_degraded_icon_nonnull(
    app_components: tuple[MainWindow, StatusTray],
) -> None:
    """set_status(DEGRADED) устанавливает ненулевую иконку."""
    _skip_if_no_tray()
    _, tray = app_components
    tray.set_status(NodeStatus.DEGRADED)
    icon = tray.tray_icon.icon()
    assert not icon.isNull(), "Tray icon should be non-null after set_status(DEGRADED)"


def test_tray_set_status_online_icon_nonnull(
    app_components: tuple[MainWindow, StatusTray],
) -> None:
    """set_status(ONLINE) устанавливает ненулевую иконку."""
    _skip_if_no_tray()
    _, tray = app_components
    tray.set_status(NodeStatus.ONLINE)
    icon = tray.tray_icon.icon()
    assert not icon.isNull(), "Tray icon should be non-null after set_status(ONLINE)"


def test_tray_initial_status_is_offline(app_components: tuple[MainWindow, StatusTray]) -> None:
    """После build_app() начальный статус трея — OFFLINE."""
    _skip_if_no_tray()
    _, tray = app_components
    assert tray._current_status == NodeStatus.OFFLINE


# ---------------------------------------------------------------------------
# Сигналы трея подключены
# ---------------------------------------------------------------------------


def test_quit_signal_connected_to_app(
    qapp: QApplication,
    app_components: tuple[MainWindow, StatusTray],
) -> None:
    """quit_requested трея подключён к qt_app.quit().

    Проверяем косвенно: сигнал имеет подключения (receivers > 0).
    """
    _, tray = app_components
    # Signal.receivers() — количество подключённых слотов
    # В Qt/PySide6 нет прямого receivers(), но можно проверить через connect
    # Триггерим quit_requested и проверяем что QApplication получил about-to-quit
    # Вместо этого — проверим, что сигнал не raises при emit (функциональный smoke)
    received_quit: list[None] = []
    # Подключаем дополнительный slot чтобы перехватить emit до app.quit()
    # Для этого disconnectим app.quit и подключаем заглушку
    tray.quit_requested.disconnect(qapp.quit)
    tray.quit_requested.connect(lambda: received_quit.append(None))
    tray._action_quit.trigger()
    assert len(received_quit) == 1, "quit_requested should have been emitted"
    # Восстанавливаем подключение
    tray.quit_requested.connect(qapp.quit)


def test_pause_signal_emits_graceful(app_components: tuple[MainWindow, StatusTray]) -> None:
    """pause_requested(PauseMode.GRACEFUL) эмитируется при trigger graceful action."""
    _, tray = app_components
    received: list[PauseMode] = []
    tray.pause_requested.connect(received.append)
    tray._action_graceful.trigger()
    assert received == [PauseMode.GRACEFUL]


def test_hard_action_emits_confirmation_not_pause(
    app_components: tuple[MainWindow, StatusTray],
) -> None:
    """Hard action эмитит hard_pause_confirmation_requested, НЕ pause_requested напрямую.

    Диалог подтверждения (strikes warning) показывает MainWindow.confirm_hard_pause().
    notes.md §6, ADR-0039 swarm.

    В integration-контексте build_app() уже подключил hard_pause_confirmation_requested
    к window.confirm_hard_pause() (реальный QMessageBox). Мы отключаем этот слот перед
    тестом, чтобы не блокировать выполнение на GUI-диалоге.
    """
    window, tray = app_components

    # Отключаем реальный обработчик из build_app, чтобы тест не открывал QMessageBox.
    # Использование disconnect() без аргументов отключает ВСЕ слоты — это нормально
    # для данного smoke-теста.
    tray.hard_pause_confirmation_requested.disconnect()

    pause_received: list[PauseMode] = []
    confirm_received: list[None] = []
    tray.pause_requested.connect(pause_received.append)
    tray.hard_pause_confirmation_requested.connect(lambda: confirm_received.append(None))
    tray._action_hard.trigger()
    assert pause_received == [], (
        "Hard action must NOT emit pause_requested directly"
    )
    assert len(confirm_received) == 1, (
        "Hard action must emit hard_pause_confirmation_requested once"
    )


# ---------------------------------------------------------------------------
# Проверка: нет swarm-внутренностей в заголовке окна
# ---------------------------------------------------------------------------


def test_window_title_no_swarm_internals(app_components: tuple[MainWindow, StatusTray]) -> None:
    """Заголовок окна не раскрывает swarm-внутренности (peer_id, session_id и т.п.)."""
    window, _ = app_components
    title = window.windowTitle()
    forbidden_in_title = ["peer_id", "session_id", "relay_token", "hop_index"]
    for term in forbidden_in_title:
        assert term not in title.lower(), (
            f"Swarm internal term {term!r} must not appear in window title: {title!r}"
        )
