"""app/windows/pairing/controller_protocol.py — Protocol-контракт для PairingController.

Волна 3 (sewing): типы событий и bridge импортируются напрямую из agent.auth.pairing.
Mirror-dataclass'ы удалены. Источник истины — agent.auth.pairing (24 зелёных теста).

Re-export через __all__ — от app.windows.pairing.controller_protocol импортировать
все нужные типы можно как раньше.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Прямые импорты из agent.auth.pairing (source of truth)
# ---------------------------------------------------------------------------
from agent.auth.models import HardwareSummary  # noqa: F401 — re-exported
from agent.auth.pairing import (  # noqa: F401 — re-exported
    PairingBridge,
    PairingEvent,
    PairingFailed,
    PairingFailureReason,
    PairingPolling,
    PairingStarted,
    PairingState,
    PairingSucceeded,
)

# ---------------------------------------------------------------------------
# Re-export list — все старые клиенты (pairing_panel, тесты) продолжают работать
# ---------------------------------------------------------------------------

__all__ = [
    "PairingState",
    "PairingFailureReason",
    "PairingStarted",
    "PairingPolling",
    "PairingSucceeded",
    "PairingFailed",
    "PairingEvent",
    "PairingBridge",
    "HardwareSummary",
    "PairingControllerProtocol",
]

# ---------------------------------------------------------------------------
# Protocol — публичная поверхность контроллера для GUI
# ---------------------------------------------------------------------------


@runtime_checkable
class PairingControllerProtocol(Protocol):
    """Публичный контракт PairingController для GUI-слоя.

    GUI использует этот Protocol — не импортирует agent.auth.pairing напрямую.
    Структурно совместим с реальным PairingController (duck typing).

    Синхронизирован с docs/components/pairing.md §3 (commands) и §4 (bridge).
    """

    bridge: PairingBridge
    """Qt-bridge для подписки GUI на события привязки."""

    def start_pairing(
        self,
        label: str,
        hardware: HardwareSummary | None = None,
    ) -> None:
        """Запустить flow привязки устройства.

        Idempotent в STARTING/AWAITING_USER/POLLING (no-op).
        В SUCCESS — игнорируется (нужен unlink для повторной привязки).
        В ERROR_*/CANCELLED — эквивалент retry_pairing.

        Args:
            label: метка устройства (socket.gethostname() по умолчанию).
                   Санитизация выполняется контроллером перед отправкой в BFF.
            hardware: опциональные диагностические данные оборудования.
        """
        ...

    def cancel_pairing(self) -> None:
        """Отменить текущий flow привязки.

        Переход CANCELLED. Не удаляет keypair устройства из keychain.
        docs/components/pairing.md §6.
        """
        ...

    def retry_pairing(self) -> None:
        """Повторить попытку привязки.

        Эквивалент start_pairing(last_label, last_hardware).
        Допустим только из ERROR_*/CANCELLED.
        """
        ...
