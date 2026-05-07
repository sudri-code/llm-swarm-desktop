"""app/screens/logs.py — заглушка экрана Logs."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class LogsScreen(QWidget):
    """Экран логов ноды с фильтрацией и отправкой в поддержку.

    Stage 1 W2: заглушка. Реальная логика — в Stage 1 W3+.
    Инвариант: при «Send to support» фильтровать sensitive-поля
    (device_token, Ed25519 private key) — notes.md §6.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LogsScreen")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Logs")
        title.setObjectName("screenTitle")
        title.setProperty("role", "h1")

        description = QLabel(
            "Локальные логи ноды; кнопка «Send to support» с явным согласием."
        )
        description.setObjectName("screenDescription")
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch()
