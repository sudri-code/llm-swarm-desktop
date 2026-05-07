"""app/tray/status_tray.py — системный трей llm-swarm-desktop.

StatusTray управляет QSystemTrayIcon с цветовым индикатором статуса,
контекстным меню и сигналами для главного окна.

Инварианты (notes.md §3.5, §6):
- Иконка всегда отражает актуальный статус ноды.
- Hard disconnect предупреждает о возможных strikes.
- Запрещённые money-термины не используются.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMainWindow, QMenu, QSystemTrayIcon

from app.state import NodeStatus, PauseMode
from app.tray.icons import make_status_icon

# Tooltip-строки — без swarm-внутренностей и money-терминов
_TOOLTIPS: dict[str, str] = {
    NodeStatus.ONLINE: "llm-swarm — online, обслуживает запросы",
    NodeStatus.DEGRADED: "llm-swarm — degraded, временно исключён из routing",
    NodeStatus.OFFLINE: "llm-swarm — offline, не подключён",
    NodeStatus.ERROR: "llm-swarm — error",
}


class StatusTray(QObject):
    """Системный трей с цветовым индикатором статуса ноды.

    Сигналы:
        show_window_requested: пользователь хочет открыть главное окно.
        pause_requested(PauseMode): пользователь хочет поставить на паузу.
            PauseMode.GRACEFUL → дождаться завершения сессий.
            PauseMode.HARD     → эмитится только после подтверждения через
                                 MainWindow.confirm_hard_pause() (не напрямую).
        hard_pause_confirmation_requested: трей запрашивает диалог hard pause
            у MainWindow. MainWindow показывает QMessageBox и при согласии
            эмитит hard_pause_confirmed, который ловит app/main.py.
        resume_requested: пользователь хочет возобновить работу.
        quit_requested: пользователь хочет выйти из приложения.
    """

    show_window_requested = Signal()
    pause_requested = Signal(PauseMode)
    hard_pause_confirmation_requested = Signal()
    resume_requested = Signal()
    quit_requested = Signal()

    def __init__(
        self,
        parent: QObject | None = None,
        parent_window: QMainWindow | None = None,
    ) -> None:
        super().__init__(parent)

        self._current_status: NodeStatus = NodeStatus.OFFLINE

        # --- Контекстное меню ---
        self._menu = QMenu()

        # "Open"
        self._action_open = self._menu.addAction("Open llm-swarm")
        self._action_open.triggered.connect(self.show_window_requested)

        self._menu.addSeparator()

        # "Pause…" submenu
        self._pause_menu = QMenu("Pause…")
        self._action_graceful = self._pause_menu.addAction(
            "Graceful (дождаться сессий)"
        )
        self._action_hard = self._pause_menu.addAction(
            "Hard disconnect (возможны strikes)"
        )
        self._action_graceful.triggered.connect(
            lambda: self.pause_requested.emit(PauseMode.GRACEFUL)
        )
        self._action_hard.triggered.connect(self.hard_pause_confirmation_requested)
        self._menu.addMenu(self._pause_menu)

        # "Resume"
        self._action_resume = self._menu.addAction("Resume")
        self._action_resume.triggered.connect(self.resume_requested)
        self._action_resume.setEnabled(False)

        self._menu.addSeparator()

        # "Quit"
        self._action_quit = self._menu.addAction("Quit")
        self._action_quit.triggered.connect(self.quit_requested)

        # --- QSystemTrayIcon ---
        self._tray = QSystemTrayIcon(parent_window)
        self._tray.setContextMenu(self._menu)

        # Устанавливаем начальный статус (offline)
        self._apply_status(NodeStatus.OFFLINE)
        self._tray.show()

        # Double-click → show window
        self._tray.activated.connect(self._on_activated)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(self, status: NodeStatus) -> None:
        """Обновить иконку трея и tooltip под новый статус.

        Args:
            status: Новый статус ноды.
        """
        self._current_status = status
        self._apply_status(status)

    def set_can_pause(self, can_pause: bool) -> None:
        """Управлять доступностью Pause / Resume (взаимоисключающе).

        Args:
            can_pause: True — пауза доступна, Resume заблокирован.
                       False — пауза заблокирована, Resume доступен.
        """
        self._pause_menu.setEnabled(can_pause)
        self._action_resume.setEnabled(not can_pause)

    @property
    def tray_icon(self) -> QSystemTrayIcon:
        """Доступ к нативному QSystemTrayIcon (для тестов)."""
        return self._tray

    @property
    def pause_menu(self) -> QMenu:
        """Доступ к submenu Pause (для тестов)."""
        return self._pause_menu

    @property
    def action_resume(self) -> object:
        """Доступ к action Resume (для тестов)."""
        return self._action_resume

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _apply_status(self, status: NodeStatus) -> None:
        icon = make_status_icon(status)
        self._tray.setIcon(icon)
        tooltip = _TOOLTIPS.get(status, f"llm-swarm — {status}")
        self._tray.setToolTip(tooltip)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_window_requested.emit()
