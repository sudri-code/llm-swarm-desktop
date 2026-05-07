"""app/windows/pairing — pairing flow UI package.

Экспортирует:
  PairingPanel — встроенный виджет для Settings (QWidget, QStackedWidget внутри).
  PairingControllerProtocol — runtime_checkable Protocol для controller.

Все типы событий (PairingStarted, PairingPolling, PairingSucceeded, PairingFailed,
PairingFailureReason, PairingState) и HardwareSummary доступны через:

    from app.windows.pairing.controller_protocol import (
        PairingStarted, PairingPolling, PairingSucceeded, PairingFailed,
        PairingFailureReason, PairingState, HardwareSummary, PairingBridge,
    )

Волна 3: после слияния с agent.auth.pairing зеркальные определения в
controller_protocol.py заменяются прямыми импортами из agent.auth.pairing.
"""

from __future__ import annotations

from app.windows.pairing.controller_protocol import PairingControllerProtocol
from app.windows.pairing.pairing_panel import PairingPanel

__all__ = [
    "PairingPanel",
    "PairingControllerProtocol",
]
