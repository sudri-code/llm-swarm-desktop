"""app/windows/main_window.py — главное окно приложения llm-swarm-desktop."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.screens.balance import BalanceScreen
from app.screens.hardware import HardwareScreen
from app.screens.logs import LogsScreen
from app.screens.models import ModelsScreen
from app.screens.settings import SettingsScreen
from app.state import NodeStatus, status_color


class StatusBanner(QFrame):
    """Статус-баннер: цветовая точка + текстовое описание состояния ноды.

    Используется как single source of truth видимого статуса — синхронно
    с иконкой трея (поток 3). QSS может адресовать через:
        StatusBanner[status="online"] { ... }
        StatusBanner[status="degraded"] { ... }
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusBanner")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(16, 8, 16, 8)
        self._layout.setSpacing(10)

        self._dot = QLabel()
        self._dot.setObjectName("StatusDot")
        self._dot.setFixedSize(10, 10)

        self._label = QLabel()
        self._label.setObjectName("StatusLabel")

        self._layout.addWidget(self._dot)
        self._layout.addWidget(self._label)
        self._layout.addStretch()

        # Начальное состояние
        self.set_status(NodeStatus.OFFLINE, "Нода не запущена")

    def set_status(self, status: NodeStatus, message: str) -> None:
        """Обновить статус-баннер.

        Args:
            status: текущий статус ноды из NodeStatus.
            message: человекочитаемый текст статуса.
        """
        color = status_color(status)
        self._dot.setStyleSheet(
            f"background-color: {color};"
            f"border-radius: 5px;"
        )
        self._label.setText(message)
        # Устанавливаем Qt-property для QSS-селекторов вида [status="online"]
        self.setProperty("status", status.value)
        # Принудительно обновить QSS после смены dynamic property
        self.style().unpolish(self)
        self.style().polish(self)


# Маппинг индекс → имя экрана (используется в сигнале current_screen_changed)
_SCREEN_NAMES: list[str] = [
    "hardware",
    "models",
    "balance",
    "logs",
    "settings",
]

# Маппинг индекс → заголовок в сайдбаре
_SIDEBAR_LABELS: list[str] = [
    "Hardware",
    "Models",
    "Balance",
    "Logs",
    "Settings",
]


class MainWindow(QMainWindow):
    """Главное окно llm-swarm-desktop.

    Layout:
        QVBoxLayout (central widget)
        ├── StatusBanner          ← статус ноды, всегда виден
        └── QSplitter(Horizontal)
            ├── QListWidget       ← sidebar навигация
            └── QStackedWidget    ← контент активного экрана

    Не создаёт QSystemTrayIcon — это поток 3.
    Не запускает backend-логику — это W3+.
    """

    current_screen_changed = Signal(str)
    hard_pause_confirmed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MainWindow")
        self.setWindowTitle("llm-swarm")
        self.resize(1100, 720)
        self.setMinimumSize(880, 600)

        self._pairing_controller: object | None = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Статус-баннер поверх сплиттера
        self._status_banner = StatusBanner()
        root_layout.addWidget(self._status_banner)

        # Разделитель: sidebar слева, контент справа
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("MainSplitter")
        root_layout.addWidget(self._splitter, stretch=1)

        # --- Sidebar ---
        self._sidebar = QListWidget()
        self._sidebar.setObjectName("Sidebar")
        self._sidebar.setFixedWidth(180)
        for label in _SIDEBAR_LABELS:
            self._sidebar.addItem(label)
        self._sidebar.setCurrentRow(0)
        self._splitter.addWidget(self._sidebar)

        # --- Stacked widget с экранами ---
        self._stack = QStackedWidget()
        self._stack.setObjectName("ContentStack")
        self._stack.addWidget(HardwareScreen())
        self._stack.addWidget(ModelsScreen())
        self._stack.addWidget(BalanceScreen())
        self._stack.addWidget(LogsScreen())
        self._settings_screen = SettingsScreen()
        self._stack.addWidget(self._settings_screen)
        self._splitter.addWidget(self._stack)

        # Пропорции сплиттера: sidebar фиксированный, контент растягивается
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)

    def _connect_signals(self) -> None:
        self._sidebar.currentRowChanged.connect(self._on_sidebar_changed)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_sidebar_changed(self, index: int) -> None:
        if 0 <= index < len(_SCREEN_NAMES):
            self._stack.setCurrentIndex(index)
            self.current_screen_changed.emit(_SCREEN_NAMES[index])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def confirm_hard_pause(self) -> bool:
        """Показать диалог подтверждения жёсткого отключения.

        Используется при запросе hard pause из трея: трей эмитит
        ``hard_pause_confirmation_requested``, app/main.py вызывает этот метод,
        и при True эмитит сигнал дальше на контроллер ноды.

        Returns:
            True, если пользователь подтвердил жёсткое отключение.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Жёсткое отключение")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            "Это прервёт активные сессии.\n"
            "Нода потеряет репутацию в сети — вплоть до временного исключения из routing."
        )
        box.setInformativeText(
            "Технически это называется «strike» (ADR-0039). "
            "Несколько подряд снижают позицию ноды при выборе участника сессии."
        )
        cancel_btn = box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        confirm_btn = box.addButton("Отключить (риск)", QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        return box.clickedButton() is confirm_btn

    def _on_hard_pause_confirmation_requested(self) -> None:
        """Слот: трей запросил подтверждение hard pause.

        Показывает диалог; при согласии пользователя эмитит ``hard_pause_confirmed``.
        """
        if self.confirm_hard_pause():
            self.hard_pause_confirmed.emit()

    def set_pairing_controller(self, controller: object) -> None:
        """Подключить pairing controller к Settings-экрану.

        Вызывается из build_app() после создания PairingController.
        SettingsScreen пересоздаёт блок «Привязка к sudri.ru» с реальным controller.

        Args:
            controller: объект, реализующий PairingControllerProtocol.
                        Тип `object` здесь чтобы принимать реальный PairingController
                        без pyright-ошибки на несовместимость property/attribute в Protocol.
        """
        self._pairing_controller = controller
        # Пересоздаём SettingsScreen с controller и заменяем в стеке
        settings_index = _SCREEN_NAMES.index("settings")
        old_settings = self._stack.widget(settings_index)
        self._settings_screen = SettingsScreen(controller=controller)
        self._stack.insertWidget(settings_index, self._settings_screen)
        if old_settings is not None:
            self._stack.removeWidget(old_settings)
            old_settings.deleteLater()

    def set_status(self, status: NodeStatus, message: str) -> None:
        """Обновить статус-баннер (проксируется из local agent через IPC)."""
        self._status_banner.set_status(status, message)

    @property
    def status_banner(self) -> StatusBanner:
        """Доступ к баннеру статуса (для тестов)."""
        return self._status_banner

    @property
    def sidebar(self) -> QListWidget:
        """Доступ к сайдбару (для тестов)."""
        return self._sidebar

    @property
    def stack(self) -> QStackedWidget:
        """Доступ к стеку экранов (для тестов)."""
        return self._stack

    @property
    def pairing_controller(self) -> object | None:
        """Доступ к pairing controller (для тестов и интеграции)."""
        return self._pairing_controller

    @property
    def settings_screen(self) -> SettingsScreen:
        """Доступ к экрану настроек (для тестов)."""
        return self._settings_screen
