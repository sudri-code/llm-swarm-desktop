"""app/screens/settings.py — заглушка экрана Settings."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class SettingsScreen(QWidget):
    """Экран настроек: автостарт, throttle, папка весов, привязка устройства.

    Stage 1 W2: заглушка. Реальная логика — в Stage 1 W3+.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SettingsScreen")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Settings")
        title.setObjectName("screenTitle")
        title.setProperty("role", "h1")

        description = QLabel(
            "Автостарт, throttle, выбор папки для весов, привязка устройства к sudri.ru."
        )
        description.setObjectName("screenDescription")
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch()
