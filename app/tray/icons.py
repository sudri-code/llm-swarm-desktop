"""app/tray/icons.py — генератор QIcon для каждого статуса ноды.

Рисует цветной круг через QPainter в цветах из app.styles.tokens.
Антиалиасинг включён. LRU-кэш по (status, size_px).
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

from app.state import NodeStatus
from app.styles.tokens import COLORS

# Маппинг статус → sRGB hex (из design tokens)
_STATUS_HEX: dict[str, str] = {
    NodeStatus.ONLINE: COLORS.OK,      # #4ebe7d
    NodeStatus.DEGRADED: COLORS.WARN,  # #eba941
    NodeStatus.OFFLINE: COLORS.FG_3,   # #82807a
    NodeStatus.ERROR: COLORS.ERR,      # #f3625d
}


@lru_cache(maxsize=32)
def make_status_icon(status: NodeStatus, size_px: int = 22) -> QIcon:
    """Создать QIcon с цветным кружком для данного статуса.

    Args:
        status: Статус ноды.
        size_px: Размер иконки в пикселях (default 22 для системного трея).

    Returns:
        QIcon с нарисованным кружком.
    """
    hex_color = _STATUS_HEX.get(status, COLORS.FG_3)

    pixmap = QPixmap(size_px, size_px)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, on=True)

    color = QColor(hex_color)
    painter.setBrush(color)
    painter.setPen(Qt.PenStyle.NoPen)

    # Отступ 1px чтобы круг не обрезался по краям
    margin = 1
    painter.drawEllipse(margin, margin, size_px - 2 * margin, size_px - 2 * margin)
    painter.end()

    return QIcon(pixmap)


def status_hex(status: NodeStatus) -> str:
    """Вернуть sRGB hex-цвет для данного статуса (для тестов и диагностики)."""
    return _STATUS_HEX.get(status, COLORS.FG_3)
