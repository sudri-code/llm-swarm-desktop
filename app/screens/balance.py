"""app/screens/balance.py — заглушка экрана Balance."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class BalanceScreen(QWidget):
    """Экран баланса и истории начислений токенов.

    Stage 1 W2: заглушка. Реальная логика — в Stage 1 W3+.
    URL-путь: /balance (не /billing — инвариант из notes.md §6).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("BalanceScreen")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Balance")
        title.setObjectName("screenTitle")
        title.setProperty("role", "h1")

        description = QLabel(
            "Заработанные токены за сессию, сегодня, всего."
        )
        description.setObjectName("screenDescription")
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch()
