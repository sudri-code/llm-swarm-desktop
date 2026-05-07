"""Stage 6: resource monitor — VRAM, RAM, bandwidth, GPU temp; авто-throttle (ADR-0038).

Реализует:
  - ResourceMonitor: фоновый опрос ресурсов каждые MONITOR_INTERVAL_SEC секунд.
  - pynvml для GPU temp / VRAM free (опционально — graceful fallback при ImportError).
  - psutil для RAM free и bandwidth (обязательные).
  - Пороги degraded из §4.6: GPU >85°C, VRAM <1 ГиБ, RAM <2 ГиБ, bandwidth >80%.
  - allow_new_session() → throttle hook для inference и network.
  - Immutable snapshot (frozen dataclass) — атомарная замена каждые 5 сек.
  - get_kv_stats() — интерфейс для получения KV-метрик из ModelShard (ADR-0040).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import psutil

from shared.constants import (
    BANDWIDTH_USAGE_LIMIT,
    GPU_TEMP_LIMIT_C,
    MONITOR_INTERVAL_SEC,
    RAM_FREE_MIN_GIB,
    VRAM_FREE_MIN_GIB,
)

if TYPE_CHECKING:
    from node.inference import ModelShard

logger = logging.getLogger(__name__)

# Попытка импорта pynvml (опциональная, ADR-0038)
try:
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    logger.debug("pynvml initialized successfully — GPU metrics enabled")
except Exception:  # ImportError, pynvml.NVMLError, любая ошибка nvmlInit
    pynvml = None  # type: ignore[assignment]
    _NVML_AVAILABLE = False
    logger.debug("pynvml not available — GPU metrics disabled (CPU-only mode)")


# ---------------------------------------------------------------------------
# Snapshot (immutable, frozen dataclass)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSnapshot:
    """Immutable снапшот состояния ресурсов ноды (ADR-0038).

    Атомарно заменяется ResourceMonitor каждые MONITOR_INTERVAL_SEC секунд.
    Читается из любого потока без блокировок.
    """

    # GPU метрики (None если pynvml недоступен или GPU отсутствует)
    gpu_temp_c: float | None = None
    vram_free_bytes: int | None = None
    vram_used_bytes: int | None = None
    vram_total_bytes: int | None = None

    # RAM метрики (psutil, всегда доступны)
    ram_free_bytes: int = 0
    ram_used_bytes: int = 0
    ram_total_bytes: int = 0

    # Bandwidth (байты/сек за последний интервал)
    bw_bytes_per_sec: float = 0.0

    # Флаги деградации
    degraded: bool = False
    degraded_reasons: list[str] = field(default_factory=list)

    # KV-cache метрики (заполняются из ModelShard)
    kv_active_sessions: int = 0
    kv_used_bytes: int = 0

    # Timestamp снапшота
    ts: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """Фоновый монитор ресурсов ноды (ADR-0038, §4.6).

    Запускается как asyncio.Task после успешной регистрации ноды.
    Опрашивает ресурсы каждые MONITOR_INTERVAL_SEC секунд.
    Публикует immutable ResourceSnapshot через атомарную замену _snapshot.

    Throttle hook:
        allow_new_session() → False если degraded=True (любая причина).

    При degraded ram → также тригерит eviction в ModelShard.

    Использование:
        monitor = ResourceMonitor(bandwidth_mbps=100.0, shard=my_shard)
        await monitor.start()
        # ... в heartbeat loop:
        snapshot = monitor.snapshot
        # ... в inference/network:
        if not monitor.allow_new_session():
            reject(...)
        await monitor.stop()
    """

    def __init__(
        self,
        bandwidth_mbps: float = 0.0,
        shard: ModelShard | None = None,
        interval_sec: int = MONITOR_INTERVAL_SEC,
    ) -> None:
        """
        Args:
            bandwidth_mbps: Заявленная полоса ноды (из конфига). 0 → проверка bandwidth отключена.
            shard: ModelShard для получения KV-метрик (ADR-0040). None → метрики нулевые.
            interval_sec: Интервал опроса в секундах. По умолчанию MONITOR_INTERVAL_SEC=5.
        """
        self._bandwidth_mbps = bandwidth_mbps
        self._shard = shard
        self._interval_sec = interval_sec

        # Инициализируем пустым снапшотом
        self._snapshot: ResourceSnapshot = ResourceSnapshot()

        # Для расчёта bandwidth delta
        self._prev_net_bytes: int | None = None
        self._prev_poll_ts: float | None = None

        # NVML device handle (None если GPU недоступен)
        self._nvml_handle: object | None = None
        if _NVML_AVAILABLE:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.debug("NVML device handle obtained for GPU 0")
            except Exception as exc:
                logger.debug("NVML: failed to get device handle: %s", exc)
                self._nvml_handle = None

        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> ResourceSnapshot:
        """Текущий immutable снапшот (атомарное чтение)."""
        return self._snapshot

    def allow_new_session(self) -> bool:
        """Разрешить ли принятие новой инференс-сессии.

        Returns False если текущий снапшот degraded=True (ADR-0038).
        Существующие сессии продолжают работу.
        """
        return not self._snapshot.degraded

    async def start(self) -> None:
        """Запустить фоновую задачу опроса ресурсов."""
        # Первый poll синхронно для немедленной инициализации снапшота
        self._poll()
        self._task = asyncio.create_task(self._loop(), name="resource_monitor")
        logger.info(
            "ResourceMonitor started: interval=%ds, gpu=%s, bandwidth_mbps=%.1f",
            self._interval_sec,
            "enabled" if self._nvml_handle is not None else "disabled",
            self._bandwidth_mbps,
        )

    async def stop(self) -> None:
        """Остановить фоновую задачу."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ResourceMonitor stopped")

    def set_shard(self, shard: ModelShard) -> None:
        """Привязать ModelShard для получения KV-метрик (вызывается после load())."""
        self._shard = shard

    # ------------------------------------------------------------------
    # Internal poll
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Бесконечный цикл опроса ресурсов."""
        while True:
            try:
                await asyncio.sleep(self._interval_sec)
            except asyncio.CancelledError:
                return
            try:
                self._poll()
            except Exception as exc:
                logger.warning("ResourceMonitor poll error: %s", exc)

    def _poll(self) -> None:
        """Однократный опрос всех ресурсов и обновление снапшота."""
        now = time.monotonic()

        # --- GPU метрики (pynvml) ---
        gpu_temp_c: float | None = None
        vram_free_bytes: int | None = None
        vram_used_bytes: int | None = None
        vram_total_bytes: int | None = None

        if self._nvml_handle is not None:
            try:
                gpu_temp_c = float(
                    pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                )
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                vram_free_bytes = int(mem_info.free)
                vram_used_bytes = int(mem_info.used)
                vram_total_bytes = int(mem_info.total)
            except Exception as exc:
                logger.debug("NVML poll error: %s", exc)

        # --- RAM метрики (psutil) ---
        vm = psutil.virtual_memory()
        ram_free_bytes = int(vm.available)
        ram_used_bytes = int(vm.used)
        ram_total_bytes = int(vm.total)

        # --- Bandwidth метрики (psutil) ---
        bw_bytes_per_sec: float = 0.0
        try:
            net_io = psutil.net_io_counters()
            total_net_bytes = net_io.bytes_sent + net_io.bytes_recv
            if self._prev_net_bytes is not None and self._prev_poll_ts is not None:
                delta_bytes = total_net_bytes - self._prev_net_bytes
                delta_sec = now - self._prev_poll_ts
                if delta_sec > 0:
                    bw_bytes_per_sec = delta_bytes / delta_sec
            self._prev_net_bytes = total_net_bytes
            self._prev_poll_ts = now
        except Exception as exc:
            logger.debug("psutil net_io_counters error: %s", exc)

        # --- KV-метрики из ModelShard ---
        kv_active_sessions = 0
        kv_used_bytes = 0
        if self._shard is not None:
            try:
                kv_stats = self._shard.get_kv_stats()
                kv_active_sessions = kv_stats["active_sessions"]
                kv_used_bytes = kv_stats["used_bytes"]
            except Exception as exc:
                logger.debug("get_kv_stats error: %s", exc)

        # --- Вычисляем degraded флаги ---
        degraded_reasons: list[str] = []

        # GPU temp > 85°C
        if gpu_temp_c is not None and gpu_temp_c > GPU_TEMP_LIMIT_C:
            degraded_reasons.append("gpu_temp")
            logger.warning(
                "ResourceMonitor: GPU temp %.1f°C > %.1f°C threshold", gpu_temp_c, GPU_TEMP_LIMIT_C
            )

        # VRAM free < 1 ГиБ
        if vram_free_bytes is not None:
            vram_free_gib = vram_free_bytes / (1024**3)
            if vram_free_gib < VRAM_FREE_MIN_GIB:
                degraded_reasons.append("vram")
                logger.warning(
                    "ResourceMonitor: VRAM free %.2f GiB < %.1f GiB threshold",
                    vram_free_gib,
                    VRAM_FREE_MIN_GIB,
                )

        # RAM free < 2 ГиБ
        ram_free_gib = ram_free_bytes / (1024**3)
        if ram_free_gib < RAM_FREE_MIN_GIB:
            degraded_reasons.append("ram")
            logger.warning(
                "ResourceMonitor: RAM free %.2f GiB < %.1f GiB threshold",
                ram_free_gib,
                RAM_FREE_MIN_GIB,
            )
            # При degraded ram — тригерим eviction в shard (ADR-0040)
            if self._shard is not None:
                try:
                    self._shard.evict_half_kv_sessions()
                except Exception as exc:
                    logger.debug("evict_half_kv_sessions error: %s", exc)

        # Bandwidth > 80% от лимита
        if self._bandwidth_mbps > 0:
            bw_limit_bytes_per_sec = self._bandwidth_mbps * 1_000_000 / 8
            if bw_bytes_per_sec > bw_limit_bytes_per_sec * BANDWIDTH_USAGE_LIMIT:
                degraded_reasons.append("bandwidth")
                logger.warning(
                    "ResourceMonitor: bandwidth %.2f MB/s > %.0f%% of %.1f Mbps limit",
                    bw_bytes_per_sec / 1_000_000,
                    BANDWIDTH_USAGE_LIMIT * 100,
                    self._bandwidth_mbps,
                )

        degraded = len(degraded_reasons) > 0

        # Атомарная замена снапшота
        self._snapshot = ResourceSnapshot(
            gpu_temp_c=gpu_temp_c,
            vram_free_bytes=vram_free_bytes,
            vram_used_bytes=vram_used_bytes,
            vram_total_bytes=vram_total_bytes,
            ram_free_bytes=ram_free_bytes,
            ram_used_bytes=ram_used_bytes,
            ram_total_bytes=ram_total_bytes,
            bw_bytes_per_sec=bw_bytes_per_sec,
            degraded=degraded,
            degraded_reasons=degraded_reasons,
            kv_active_sessions=kv_active_sessions,
            kv_used_bytes=kv_used_bytes,
            ts=now,
        )

        if degraded:
            logger.debug(
                "ResourceMonitor snapshot: degraded=True reasons=%s",
                degraded_reasons,
            )
