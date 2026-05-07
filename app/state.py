"""app/state.py — общие enum'ы и утилиты состояния приложения."""

from __future__ import annotations

from enum import StrEnum

from app.styles.tokens import COLORS


class PauseMode(StrEnum):
    """Режим паузы ноды.

    GRACEFUL — дождаться завершения активных сессий перед остановкой.
    HARD     — немедленное отключение; может привести к strikes (ADR-0039 swarm).
    """

    GRACEFUL = "graceful"
    HARD = "hard"


class NodeStatus(StrEnum):
    """Статус ноды swarm.

    Значение используется как CSS-property для QSS-селекторов,
    а также передаётся в системный трей (поток 3) и StatusBanner.
    """

    ONLINE = "online"  # COLORS.OK — зелёный
    DEGRADED = "degraded"  # COLORS.WARN — жёлтый (ADR-0038)
    OFFLINE = "offline"  # COLORS.FG_3 — серый
    ERROR = "error"  # COLORS.ERR — красный


_STATUS_COLOR_MAP: dict[NodeStatus, str] = {
    NodeStatus.ONLINE: COLORS.OK,
    NodeStatus.DEGRADED: COLORS.WARN,
    NodeStatus.OFFLINE: COLORS.FG_3,
    NodeStatus.ERROR: COLORS.ERR,
}


def status_color(status: NodeStatus) -> str:
    """Возвращает sRGB hex-цвет для данного статуса ноды."""
    return _STATUS_COLOR_MAP[status]
