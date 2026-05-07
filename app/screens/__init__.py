"""app/screens — экраны основного окна приложения."""

from __future__ import annotations

from app.screens.balance import BalanceScreen
from app.screens.hardware import HardwareScreen
from app.screens.logs import LogsScreen
from app.screens.models import ModelsScreen
from app.screens.settings import SettingsScreen

__all__ = [
    "BalanceScreen",
    "HardwareScreen",
    "LogsScreen",
    "ModelsScreen",
    "SettingsScreen",
]
