"""app/screens/hardware.py — заглушка экрана Hardware probe."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class HardwareScreen(QWidget):
    """Экран отображения характеристик железа и подбора слоёв модели.

    Stage 1 W2: заглушка. Реальная логика — в Stage 1 W3+.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("HardwareScreen")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Hardware probe")
        title.setObjectName("screenTitle")
        title.setProperty("role", "h1")

        description = QLabel(
            "Здесь будут показатели GPU/CPU/RAM и подбор слоёв модели."
        )
        description.setObjectName("screenDescription")
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch()
