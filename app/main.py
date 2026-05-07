"""app/main.py — точка входа llm-swarm-desktop.

Запускает один Python-процесс с qasync event loop:
  - QApplication + QSS стили из tokens.qss
  - MainWindow (основной GUI)
  - StatusTray (системный трей)
  - asyncio event loop через qasync (один loop, не два)

Ресурсы из resources_rc загружаются через try/except, потому что
файл gitignored и генерируется через ``make tokens``.  Если ресурсы
не загрузились — GUI поднимается без bundled-шрифтов.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import platformdirs
import qasync  # type: ignore[import-untyped]
from PySide6.QtWidgets import QApplication

from agent.auth.pairing import PairingBridge, PairingController
from app.logging_setup import setup_logging
from app.state import NodeStatus, PauseMode
from app.tray.status_tray import StatusTray
from app.windows.main_window import MainWindow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ресурсы Qt (генерируются make tokens; gitignored)
# ---------------------------------------------------------------------------


def _load_resources() -> None:
    """Попытаться загрузить сгенерированный resources_rc.

    Если файл ещё не сгенерирован (fresh checkout без ``make tokens``) —
    логируем warning и продолжаем.  Шрифты упадут на системные, QSS
    загрузится без font-embedding, но GUI поднимется.
    """
    try:
        from app.resources import resources_rc  # type: ignore[import] # noqa: F401
    except ImportError:
        logger.warning(
            "app/resources/resources_rc.py not found — bundled fonts unavailable. "
            "Run `make tokens` to generate Qt resources."
        )


# ---------------------------------------------------------------------------
# Bootstrap (вынесен в отдельную функцию для testability)
# ---------------------------------------------------------------------------


def build_app(qt_app: QApplication) -> tuple[MainWindow, StatusTray]:
    """Создать и связать все компоненты GUI.

    Не запускает event loop — только строит объектный граф и подключает сигналы.
    Вызывается как из ``main()``, так и из integration-тестов.

    Args:
        qt_app: Уже созданный экземпляр QApplication.

    Returns:
        Кортеж (MainWindow, StatusTray).
    """
    from app.styles import apply_to_app

    # Применяем QSS-стили из design tokens
    try:
        apply_to_app(qt_app)
    except FileNotFoundError as exc:
        logger.warning("QSS not loaded: %s", exc)

    # Создаём pairing bridge и controller
    bridge = PairingBridge()
    controller = PairingController(bridge=bridge)

    # Создаём главное окно и подключаем pairing controller
    window = MainWindow()
    window.set_pairing_controller(controller)

    # Создаём трей, передаём ссылку на окно как parent_window
    tray = StatusTray(parent_window=window)

    # --- Подключение сигналов трея ---

    tray.show_window_requested.connect(window.showNormal)
    tray.show_window_requested.connect(window.activateWindow)
    tray.show_window_requested.connect(window.raise_)

    tray.quit_requested.connect(qt_app.quit)

    def _on_pause_requested(mode: PauseMode) -> None:
        if mode is PauseMode.GRACEFUL:
            logger.info("graceful pause requested")
        else:
            logger.info("hard pause requested (mode=%s)", mode)

    def _on_resume_requested() -> None:
        logger.info("resume requested")

    def _on_hard_pause() -> None:
        """Трей запросил подтверждение hard pause — показываем диалог в MainWindow."""
        if window.confirm_hard_pause():
            logger.info("hard pause confirmed by user")
            # W3+: здесь будет вызов контроллера ноды
            # node_controller.hard_pause()

    tray.pause_requested.connect(_on_pause_requested)
    tray.hard_pause_confirmation_requested.connect(_on_hard_pause)
    tray.resume_requested.connect(_on_resume_requested)

    # Начальный статус — OFFLINE до получения данных от local agent
    tray.set_status(NodeStatus.OFFLINE)

    # При выходе прячем трей чтобы иконка не зависала в трее ОС
    qt_app.aboutToQuit.connect(tray.tray_icon.hide)

    # При выходе отменяем активный pairing flow (если был запущен)
    qt_app.aboutToQuit.connect(controller.cancel_pairing)

    return window, tray


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Запустить llm-swarm-desktop.

    Инициализирует logging, создаёт QApplication, запускает qasync event loop.
    Один loop для Qt + asyncio — критично, второй loop не создаётся.
    """
    # Logging инициализируется до QApplication — важно для early errors
    log_dir = Path(platformdirs.user_log_dir("llm-swarm-desktop"))
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=log_dir / "app.log")

    logger.info("Starting llm-swarm-desktop")

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("llm-swarm-desktop")
    qt_app.setOrganizationName("sudri")
    # Не выходить при закрытии последнего окна — живём в трее
    qt_app.setQuitOnLastWindowClosed(False)

    _load_resources()

    window, tray = build_app(qt_app)

    window.show()
    tray.tray_icon.show()

    # --- qasync event loop ---
    # Один loop: Qt event loop совмещён с asyncio через qasync.
    # asyncio.set_event_loop вызывается внутри QEventLoop.__init__.
    loop = qasync.QEventLoop(qt_app)
    asyncio.set_event_loop(loop)

    logger.info("Event loop starting")
    with loop:
        loop.run_forever()

    logger.info("llm-swarm-desktop stopped")


if __name__ == "__main__":
    main()
