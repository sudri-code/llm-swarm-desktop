"""app/screens/models.py — заглушка экрана Models."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ModelsScreen(QWidget):
    """Экран выбора моделей и диапазона слоёв.

    Stage 1 W2: заглушка. Реальная логика — в Stage 1 W3+.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ModelsScreen")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Models")
        title.setObjectName("screenTitle")
        title.setProperty("role", "h1")

        description = QLabel(
            "Список моделей на диске и доли весов, которые обслуживает ваше устройство."
        )
        description.setObjectName("screenDescription")
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch()
