"""Stage 2/5/6: загрузка шарда модели (ModelShard), forward pass (B2), KV-cache.

ModelShard отвечает за загрузку только нужного диапазона слоёв LlamaDecoderLayer
плюс опционально embed_tokens (первая нода) и lm_head/norm (последняя нода).
Поддерживаемые архитектуры: только Llama 2/3 (ADR-0008).
Embedding/lm_head размещение — ADR-0010.
OOM fail-fast на CUDA — ADR-0008.

KV-cache:
  Stage 2 (ADR-0008): single-session in-memory, LRU-1 вытеснение.
  Stage 6 (ADR-0040): multi-session dict с TTL 300с, VRAM-квотой, LRU-eviction.
  Метод begin_session: принимает новую сессию если есть место, иначе 503.

Stage 5 (E3):
  - E3.1: _materialize_layer_from_chunk — загрузка из ChunkStore вместо random-init.
  - E3.2: bitsandbytes Linear8bitLt для Llama-2/3 при weights_dtype='int8' (ADR-0034).
  - E3.3: Deterministic flags (torch.use_deterministic_algorithms + cudnn + tf32).
  - E3.4: CUDA capability check (min 8.0 для int8).
  - E3.5: Lazy materialize — слои грузятся при первом обращении (async _ensure_layer).

Stage 6 (ADR-0040):
  - Multi-session KV-cache: _kv_sessions dict[UUID, KVCacheEntry].
  - TTL eviction, LRU eviction при превышении квоты.
  - evict_half_kv_sessions() — принудительный eviction при degraded ram.
  - get_kv_stats() — метрики для ResourceMonitor и heartbeat.
  - Фоновый sweep-таск (kv_sweep_interval_s=60).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

if TYPE_CHECKING:
    from transformers import LlamaConfig

    from node.weights import WeightManager

logger = logging.getLogger(__name__)

# Sentinel — означает «лимит не задан», OOM-проверка пропускается.
_NO_VRAM_LIMIT: float = 0.0

# KV-cache дефолты (ADR-0040)
_KV_TTL_SECONDS_DEFAULT: int = 300
_KV_QUOTA_FRACTION_DEFAULT: float = 0.25
_KV_QUOTA_MAX_GB_DEFAULT: float = 8.0
_KV_MAX_SESSIONS_DEFAULT: int = 64
_KV_SWEEP_INTERVAL_DEFAULT: int = 60


@dataclass
class KVCacheEntry:
    """Запись в multi-session KV-cache (ADR-0040).

    Хранит DynamicCache + метаданные для TTL и LRU eviction.
    """

    cache: DynamicCache
    last_use_ts: float = field(default_factory=time.monotonic)
    created_ts: float = field(default_factory=time.monotonic)
    est_bytes: int = 0


# Минимальная CUDA compute capability для int8 (bitsandbytes требует Ampere+).
_MIN_CUDA_CAPABILITY_INT8: tuple[int, int] = (8, 0)

# Условный импорт bitsandbytes (не доступен на macOS CPU).
try:
    import bitsandbytes as bnb  # type: ignore[import-untyped]

    HAS_BNB = True
except ImportError:
    bnb = None  # type: ignore[assignment]
    HAS_BNB = False


class ModelShard(nn.Module):
    """Шард трансформера: диапазон слоёв [layer_start, layer_end) + embed/lm_head.

    Args:
        model_id: Идентификатор модели (путь или HuggingFace hub id).
        layer_start: Первый слой включительно.
        layer_end: Последний слой исключительно.
        device: Целевое устройство ('cpu', 'cuda', 'cuda:0' и т.д.).
        dtype: Тип весов (по умолчанию torch.float16).
        vram_gb: Лимит VRAM в ГБ для OOM fail-fast. 0.0 = без лимита.
        weight_manager: WeightManager для загрузки реальных весов (Stage 5, E3.1).
                        None → random-init (для тестов и tiny-llama-test).
        weights_dtype: 'fp16' | 'int8'. При 'int8' и HAS_BNB — Linear8bitLt (E3.2).
    """

    def __init__(
        self,
        model_id: str,
        layer_start: int,
        layer_end: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        vram_gb: float = _NO_VRAM_LIMIT,
        weight_manager: WeightManager | None = None,
        weights_dtype: str = "fp16",
        kv_ttl_seconds: int = _KV_TTL_SECONDS_DEFAULT,
        kv_quota_fraction: float = _KV_QUOTA_FRACTION_DEFAULT,
        kv_quota_max_gb: float = _KV_QUOTA_MAX_GB_DEFAULT,
        kv_max_sessions: int = _KV_MAX_SESSIONS_DEFAULT,
        kv_sweep_interval_s: int = _KV_SWEEP_INTERVAL_DEFAULT,
    ) -> None:
        super().__init__()
        if layer_start < 0:
            raise ValueError(f"layer_start must be >= 0, got {layer_start}")
        if layer_end <= layer_start:
            raise ValueError(
                f"layer_end must be > layer_start, got layer_end={layer_end}, "
                f"layer_start={layer_start}"
            )

        self.model_id = model_id
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.device = device
        self.dtype = dtype
        self.vram_gb = vram_gb
        self._weight_manager = weight_manager
        self._weights_dtype = weights_dtype

        # KV-cache параметры (ADR-0040)
        self._kv_ttl_seconds = kv_ttl_seconds
        self._kv_quota_fraction = kv_quota_fraction
        self._kv_quota_max_gb = kv_quota_max_gb
        self._kv_max_sessions = kv_max_sessions
        self._kv_sweep_interval_s = kv_sweep_interval_s

        self._config: LlamaConfig | None = None
        # Sparse dict: layer_idx → LlamaDecoderLayer (lazy loading E3.5)
        self._layers_dict: dict[int, nn.Module] = {}
        self._layers: nn.ModuleList | None = None
        self._embed_tokens: nn.Embedding | None = None
        self._norm: nn.Module | None = None
        self._lm_head: nn.Linear | None = None
        self._rotary_emb: LlamaRotaryEmbedding | None = None
        self._loaded: bool = False

        # KV-cache (ADR-0040): multi-session dict, заменяет single-session LRU-1 (ADR-0008).
        # Каждая сессия — независимый DynamicCache с TTL и VRAM-квотой.
        self._kv_sessions: dict[uuid.UUID, KVCacheEntry] = {}

        # Счётчики eviction для observability (ADR-0040 §6)
        self._kv_evictions_ttl: int = 0
        self._kv_evictions_lru: int = 0
        self._kv_evictions_degraded: int = 0

        # Фоновый sweep task (запускается при первом begin_session в async контексте)
        self._kv_sweep_task: asyncio.Task | None = None

        # Детерминизм (E3.3, ADR-0034)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        # allow_tf32 доступен только если CUDA скомпилирован в torch
        if torch.cuda.is_available() or hasattr(torch.backends, "cuda"):
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
            except (AttributeError, AssertionError):
                pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_embedding(self) -> bool:
        """True если нода отвечает за первый слой — несёт embed_tokens."""
        return self.layer_start == 0

    @property
    def has_lm_head(self) -> bool:
        """True если нода отвечает за последний слой — несёт lm_head + norm."""
        if self._config is None:
            raise RuntimeError("Call load() before accessing has_lm_head")
        return self.layer_end == self._config.num_hidden_layers

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Загрузить конфиг модели и нужные слои в память.

        Только Llama 2/3 (model_type == 'llama'). Слои вне [layer_start, layer_end)
        не материализуются — LlamaForCausalLM создаётся на meta-device,
        после чего нужные слои переносятся на реальный device.

        При CUDA — OOM fail-fast: если потребление VRAM превышает vram_gb, процесс
        завершается с sys.exit(1).

        При weight_manager is not None — используется lazy loading (E3.5):
        слои не загружаются здесь, а материализуются при первом вызове forward().

        При weight_manager is None — eagerly random-init все слои (как Stage 2).
        """
        # --- Config ---
        config: LlamaConfig = AutoConfig.from_pretrained(self.model_id)
        if config.model_type != "llama":
            raise ValueError(
                f"Stage 2 supports Llama 2/3 only, see ADR-0008. "
                f"Got model_type='{config.model_type}' for model_id='{self.model_id}'"
            )
        self._config = config

        num_layers = config.num_hidden_layers
        if self.layer_end > num_layers:
            raise ValueError(
                f"layer_end={self.layer_end} exceeds model num_hidden_layers={num_layers}"
            )

        logger.info(
            "Loading shard: model=%s layers=[%d,%d) device=%s dtype=%s weight_manager=%s",
            self.model_id,
            self.layer_start,
            self.layer_end,
            self.device,
            self.dtype,
            "real" if self._weight_manager is not None else "random-init",
        )

        if self._weight_manager is None:
            # --- Eager random-init (Stage 2 / тесты / tiny-llama-test) ---
            with torch.device("meta"):
                full_model = LlamaForCausalLM(config)

            target_layers = nn.ModuleList()
            for i in range(self.layer_start, self.layer_end):
                layer = _materialize_layer(full_model.model.layers[i], config, self.dtype)
                target_layers.append(layer)

            self._layers = target_layers
        else:
            # --- Lazy loading (E3.5): слои загружаются при первом forward() ---
            # _layers остаётся None; _layers_dict заполняется в _ensure_layer_sync()
            logger.info(
                "Lazy loading mode: layers will be materialized from ChunkStore on first access"
            )
            self._layers = None

        # --- embed_tokens для первой ноды ---
        if self.has_embedding:
            self._embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=getattr(config, "pad_token_id", None),
                dtype=self.dtype,
            )
            logger.info("Shard has embed_tokens (layer_start=0)")

        # --- lm_head + norm для последней ноды ---
        if self.layer_end == config.num_hidden_layers:
            self._norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(self.dtype)
            self._lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False, dtype=self.dtype
            )
            logger.info("Shard has lm_head + norm (layer_end=%d)", self.layer_end)

        # --- Rotary embeddings ---
        self._rotary_emb = LlamaRotaryEmbedding(config=config)

        # --- Перенос на устройство ---
        # Вызываем _move_to_device() в обоих случаях для backward-совместимости
        # (тесты могут патчить этот метод через patch.object).
        # В lazy path _layers == None, поэтому перенос слоёв — noop.
        self._move_to_device()

        # --- OOM fail-fast для CUDA (только eager path) ---
        if self._weight_manager is None and self.device.startswith("cuda"):
            self._check_cuda_oom()

        self._loaded = True
        logger.info(
            "Shard loaded: layers=[%d,%d) embed=%s lm_head=%s lazy=%s",
            self.layer_start,
            self.layer_end,
            self.has_embedding,
            self.layer_end == config.num_hidden_layers,
            self._weight_manager is not None,
        )

    def _move_to_device(self) -> None:
        """Перенести все параметры на целевое устройство.

        При lazy loading (_weight_manager is not None) _layers == None, поэтому
        перенос слоёв пропускается; embed_tokens/lm_head/norm переносятся всегда.
        """
        if self._layers is not None:
            self._layers = self._layers.to(self.device)
        if self._embed_tokens is not None:
            self._embed_tokens = self._embed_tokens.to(self.device)
        if self._norm is not None:
            self._norm = self._norm.to(self.device)
        if self._lm_head is not None:
            self._lm_head = self._lm_head.to(self.device)
        if self._rotary_emb is not None:
            self._rotary_emb = self._rotary_emb.to(self.device)

    def _check_cuda_oom(self) -> None:
        """OOM fail-fast: завершить процесс если потребление VRAM превышает лимит.

        Вызывается только при device='cuda*'. На CPU не вызывается (CI без VRAM).
        Если vram_gb == 0.0 (лимит не задан) — проверка пропускается.
        """
        if self.vram_gb == _NO_VRAM_LIMIT:
            return

        torch.cuda.synchronize()
        used_bytes = torch.cuda.memory_allocated()
        used_gb = used_bytes / (1024**3)

        if used_gb > self.vram_gb:
            msg = f"OOM: shard requires {used_gb:.2f}GB but vram_gb={self.vram_gb}GB"
            logger.error(msg)
            sys.exit(1)

    def _check_cuda_capability(self) -> None:
        """Проверить CUDA capability >= 8.0 для int8 режима (E3.4).

        Вызывается при weights_dtype='int8' перед загрузкой слоя с CUDA.
        На CPU проверка пропускается.
        Поднимает RuntimeError при несоответствии требований.
        """
        if not self.device.startswith("cuda"):
            return

        if not torch.cuda.is_available():
            return

        # Определяем номер устройства
        device_idx = 0
        if ":" in self.device:
            try:
                device_idx = int(self.device.split(":")[-1])
            except ValueError:
                device_idx = 0

        capability = torch.cuda.get_device_capability(device_idx)
        if capability < _MIN_CUDA_CAPABILITY_INT8:
            min_maj, min_min = _MIN_CUDA_CAPABILITY_INT8
            cap_maj, cap_min = capability
            raise RuntimeError(
                f"CUDA device {self.device} has compute capability "
                f"{cap_maj}.{cap_min}, but bitsandbytes int8 requires "
                f">= {min_maj}.{min_min} (Ampere+). "
                f"Use --weights-dtype fp16 for older GPUs."
            )

    # ------------------------------------------------------------------
    # Lazy layer materialization (E3.5)
    # ------------------------------------------------------------------

    async def _ensure_layer(self, layer_idx: int) -> nn.Module:
        """Гарантировать наличие слоя layer_idx, загрузив при необходимости (E3.5).

        При weight_manager is None — возвращает из eager-loaded _layers.
        При weight_manager is not None — lazy load из ChunkStore.
        """
        if self._weight_manager is None:
            # Eager path: берём из ModuleList
            if self._layers is None:
                raise RuntimeError("Shard not loaded; call load() first")
            local_idx = layer_idx - self.layer_start
            return self._layers[local_idx]

        # Lazy path
        if layer_idx not in self._layers_dict:
            await self._load_layer_from_chunk(layer_idx)

        return self._layers_dict[layer_idx]

    async def _load_layer_from_chunk(self, layer_idx: int) -> None:
        """Загрузить один слой из ChunkStore через WeightManager (E3.1).

        1. Вызвать weight_manager.ensure_chunks для этого слоя.
        2. Найти нужный чанк по layer_idx из манифеста.
        3. torch.load state_dict из файла чанка.
        4. Применить state_dict к свежесозданному LlamaDecoderLayer.
        5. При weights_dtype='int8' и HAS_BNB — конвертировать Linear в Linear8bitLt (E3.2).
        6. Перенести на device.
        """
        assert self._weight_manager is not None
        assert self._config is not None

        # Проверяем CUDA capability перед первой int8-загрузкой (E3.4)
        if self._weights_dtype == "int8" and self.device.startswith("cuda"):
            self._check_cuda_capability()

        # Обеспечиваем наличие чанков для этого слоя
        chunk_paths = await self._weight_manager.ensure_chunks(
            self.model_id, (layer_idx, layer_idx + 1)
        )

        if not chunk_paths:
            logger.warning(
                "No chunk paths returned for layer %d, falling back to random-init",
                layer_idx,
            )
            layer = _make_random_layer(self._config, layer_idx, self.dtype)
            layer = layer.to(self.device)
            self._layers_dict[layer_idx] = layer
            return

        # Ищем нужный чанк через манифест
        chunk_path = _find_chunk_for_layer(
            chunk_paths,
            self._weight_manager,
            self.model_id,
            layer_idx,
        )

        if chunk_path is None or not chunk_path.exists():
            logger.warning(
                "Chunk file not found for layer %d, falling back to random-init",
                layer_idx,
            )
            layer = _make_random_layer(self._config, layer_idx, self.dtype)
            layer = layer.to(self.device)
            self._layers_dict[layer_idx] = layer
            return

        # Создаём базовый слой и загружаем state_dict
        layer = _load_layer_from_path(chunk_path, self._config, layer_idx, self.dtype)

        # E3.2: конвертировать Linear в Linear8bitLt при int8
        if self._weights_dtype == "int8":
            if HAS_BNB:
                layer = _convert_to_int8(layer)
                logger.debug("Converted layer %d to bitsandbytes int8", layer_idx)
            else:
                logger.warning(
                    "weights_dtype=int8 requested but bitsandbytes not available "
                    "(macOS or missing install); loading layer %d in fp16",
                    layer_idx,
                )

        layer = layer.to(self.device)
        self._layers_dict[layer_idx] = layer

        logger.info(
            "Layer %d materialized from chunk %s (dtype=%s)",
            layer_idx,
            chunk_path.name,
            self._weights_dtype,
        )

    # ------------------------------------------------------------------
    # Accessors (для forward pass)
    # ------------------------------------------------------------------

    @property
    def layers(self) -> nn.ModuleList:
        """Список загруженных DecoderLayer (только для eager path)."""
        if self._weight_manager is not None:
            raise RuntimeError(
                "layers property is not available in lazy loading mode; "
                "use forward() which calls _ensure_layer() internally"
            )
        if self._layers is None:
            raise RuntimeError("Call load() first")
        return self._layers

    @property
    def embed_tokens(self) -> nn.Embedding:
        """embed_tokens (только если has_embedding)."""
        if self._embed_tokens is None:
            raise RuntimeError("This shard has no embed_tokens (layer_start != 0)")
        return self._embed_tokens

    @property
    def norm(self) -> nn.Module:
        """Финальный RMSNorm (только если has_lm_head)."""
        if self._norm is None:
            raise RuntimeError("This shard has no norm (layer_end != num_hidden_layers)")
        return self._norm

    @property
    def lm_head(self) -> nn.Linear:
        """lm_head (только если has_lm_head)."""
        if self._lm_head is None:
            raise RuntimeError("This shard has no lm_head (layer_end != num_hidden_layers)")
        return self._lm_head

    @property
    def config(self) -> LlamaConfig:
        """LlamaConfig модели (доступен после load())."""
        if self._config is None:
            raise RuntimeError("Call load() first")
        return self._config

    # ------------------------------------------------------------------
    # KV-cache API (ADR-0040, Stage 6)
    # ------------------------------------------------------------------

    def _kv_quota_bytes(self) -> int:
        """Вычислить VRAM-квоту под KV-cache в байтах (ADR-0040 §3).

        На CPU/dev (vram_gb == 0) — квота не ограничена (возвращает sys.maxsize).
        """
        if self.vram_gb == _NO_VRAM_LIMIT:
            return sys.maxsize
        fraction_bytes = self._kv_quota_fraction * self.vram_gb * (1024**3)
        max_bytes = self._kv_quota_max_gb * (1024**3)
        return int(min(fraction_bytes, max_bytes))

    def _kv_used_bytes(self) -> int:
        """Суммарный est_bytes всех активных KV-сессий."""
        return sum(e.est_bytes for e in self._kv_sessions.values())

    def _estimate_session_bytes(self, seq_len: int) -> int:
        """Оценить занимаемые байты KV-cache для сессии (ADR-0040 §1).

        Формула: seq_len × num_kv_heads × head_dim × num_layers_in_shard × 2 (K+V) × bytes_per_elem.
        При отсутствии конфига — возвращает 0 (квота не считается).
        """
        if self._config is None:
            return 0
        cfg = self._config
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        num_layers = self.layer_end - self.layer_start
        bytes_per_elem = 2  # fp16
        return seq_len * num_kv_heads * head_dim * num_layers * 2 * bytes_per_elem

    def _sweep_ttl(self) -> int:
        """Вычистить сессии с истёкшим TTL. Возвращает число evicted."""
        now = time.monotonic()
        evicted = 0
        expired = [
            sid
            for sid, e in self._kv_sessions.items()
            if now - e.last_use_ts > self._kv_ttl_seconds
        ]
        for sid in expired:
            del self._kv_sessions[sid]
            evicted += 1
            self._kv_evictions_ttl += 1
            logger.debug("KV-cache TTL eviction: session %s", sid)
        return evicted

    def _evict_lru_until_fits(self, new_est_bytes: int) -> bool:
        """Вытеснять LRU-сессии пока новая не поместится.

        Returns True если после eviction новая сессия влезает, False иначе.
        """
        quota = self._kv_quota_bytes()
        while self._kv_used_bytes() + new_est_bytes > quota and self._kv_sessions:
            # Находим LRU (минимальный last_use_ts)
            lru_id = min(self._kv_sessions, key=lambda s: self._kv_sessions[s].last_use_ts)
            del self._kv_sessions[lru_id]
            self._kv_evictions_lru += 1
            logger.debug("KV-cache LRU eviction: session %s", lru_id)
        return self._kv_used_bytes() + new_est_bytes <= quota

    def begin_session(self, session_id: uuid.UUID, seq_len: int = 1) -> None:
        """Начать новую сессию инференса (ADR-0040).

        Если сессия уже существует — обновляет last_use_ts (idempotent).
        Если новая — пытается выделить место:
          1. Sweep TTL.
          2. LRU eviction пока не влезет.
          3. Если всё равно не влезает (quota) → RuntimeError("kv_quota_exhausted").
          4. Если превышен kv_max_sessions → RuntimeError("kv_quota_exhausted").

        Raises:
            RuntimeError: если сессия не может быть принята (quota_exhausted).
        """
        # Idempotent: сессия уже открыта — просто обновляем last_use_ts
        if session_id in self._kv_sessions:
            self._kv_sessions[session_id].last_use_ts = time.monotonic()
            return

        # Новая сессия — sweep expired first
        self._sweep_ttl()

        # Max sessions hard cap: evict LRU до тех пор пока не влезем
        while len(self._kv_sessions) >= self._kv_max_sessions:
            if not self._kv_sessions:
                raise RuntimeError(
                    f"kv_quota_exhausted: max_sessions={self._kv_max_sessions} reached"
                )
            lru_id = min(self._kv_sessions, key=lambda s: self._kv_sessions[s].last_use_ts)
            del self._kv_sessions[lru_id]
            self._kv_evictions_lru += 1
            logger.debug("KV-cache LRU eviction (max_sessions): session %s", lru_id)

        est_bytes = self._estimate_session_bytes(seq_len)
        if not self._evict_lru_until_fits(est_bytes):
            raise RuntimeError(
                f"kv_quota_exhausted: cannot fit session "
                f"(est={est_bytes}, used={self._kv_used_bytes()}, quota={self._kv_quota_bytes()})"
            )

        self._kv_sessions[session_id] = KVCacheEntry(
            cache=DynamicCache(),
            est_bytes=est_bytes,
        )
        logger.debug(
            "KV-cache: new session %s (est=%d bytes, total_sessions=%d)",
            session_id,
            est_bytes,
            len(self._kv_sessions),
        )

    def clear_session(self, session_id: uuid.UUID) -> None:
        """Явная очистка сессии. Если session_id отсутствует — игнорируется."""
        if session_id in self._kv_sessions:
            del self._kv_sessions[session_id]
            logger.debug("KV-cache: session %s cleared", session_id)

    def evict_half_kv_sessions(self) -> int:
        """Принудительный eviction половины самых старых сессий (degraded ram, ADR-0040 §4).

        Вызывается из ResourceMonitor при ram < RAM_FREE_MIN_GIB.
        Возвращает число evicted сессий.
        """
        if not self._kv_sessions:
            return 0
        n_evict = max(1, len(self._kv_sessions) // 2)
        sorted_sessions = sorted(
            self._kv_sessions.keys(), key=lambda s: self._kv_sessions[s].last_use_ts
        )
        evicted = 0
        for sid in sorted_sessions[:n_evict]:
            del self._kv_sessions[sid]
            self._kv_evictions_degraded += 1
            evicted += 1
            logger.debug("KV-cache degraded eviction: session %s", sid)
        logger.info("KV-cache: evicted %d sessions due to degraded RAM", evicted)
        return evicted

    def get_kv_stats(self) -> dict:
        """Метрики KV-cache для ResourceMonitor и heartbeat (ADR-0040 §6).

        Returns dict с ключами:
            active_sessions: int — число активных сессий.
            used_bytes: int — суммарный est_bytes.
            evictions_ttl_total: int
            evictions_lru_total: int
            evictions_degraded_total: int
        """
        return {
            "active_sessions": len(self._kv_sessions),
            "used_bytes": self._kv_used_bytes(),
            "evictions_ttl_total": self._kv_evictions_ttl,
            "evictions_lru_total": self._kv_evictions_lru,
            "evictions_degraded_total": self._kv_evictions_degraded,
        }

    async def start_kv_sweep(self) -> None:
        """Запустить фоновый sweep TTL-expired сессий (ADR-0040 §2).

        Должен вызываться один раз после load() в async контексте.
        """
        if self._kv_sweep_task is not None:
            return
        self._kv_sweep_task = asyncio.create_task(self._kv_sweep_loop(), name="kv_sweep")
        logger.debug("KV-cache sweep task started (interval=%ds)", self._kv_sweep_interval_s)

    async def stop_kv_sweep(self) -> None:
        """Остановить фоновый sweep task."""
        if self._kv_sweep_task is not None and not self._kv_sweep_task.done():
            self._kv_sweep_task.cancel()
            try:
                await self._kv_sweep_task
            except asyncio.CancelledError:
                pass
        self._kv_sweep_task = None

    async def _kv_sweep_loop(self) -> None:
        """Фоновый цикл: вычищает TTL-expired сессии раз в kv_sweep_interval_s."""
        while True:
            try:
                await asyncio.sleep(self._kv_sweep_interval_s)
            except asyncio.CancelledError:
                return
            n = self._sweep_ttl()
            if n > 0:
                logger.debug("KV-cache sweep: evicted %d TTL-expired sessions", n)

    # ------------------------------------------------------------------
    # Forward pass (B2, ADR-0008/0009/0010)
    # ------------------------------------------------------------------

    @torch.no_grad()
    async def forward_async(
        self,
        *,
        token_ids: list[int] | None = None,
        hidden_states: torch.Tensor | None = None,
        position_offset: int = 0,
        session_id: uuid.UUID,
    ) -> dict:
        """Async-версия forward pass с lazy layer loading (E3.5).

        Используется когда weight_manager задан — вызывает _ensure_layer() для каждого слоя.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before forward()")
        if session_id not in self._kv_sessions:
            raise RuntimeError(
                f"session not started, call begin_session first (session_id={session_id})"
            )

        has_tok = token_ids is not None and len(token_ids) > 0
        has_hs = hidden_states is not None
        if has_tok == has_hs:
            raise ValueError(
                "Exactly one of token_ids or hidden_states must be provided "
                f"(token_ids={'set' if has_tok else 'None'}, "
                f"hidden_states={'set' if has_hs else 'None'})"
            )

        # Определяем device из embed_tokens или norm или первого доступного слоя
        dev = self._get_device()

        if has_tok:
            if not self.has_embedding:
                raise ValueError("token_ids provided but shard has no embed_tokens")
            ids_t = torch.tensor([token_ids], dtype=torch.long, device=dev)
            hs = self._embed_tokens(ids_t)  # type: ignore[union-attr]
        else:
            hs = hidden_states.to(device=dev, dtype=self.dtype)  # type: ignore[union-attr]

        # Обновляем last_use_ts и получаем KV-cache сессии
        entry = self._kv_sessions[session_id]
        entry.last_use_ts = time.monotonic()
        kv_cache = entry.cache

        seq_len = hs.shape[1]
        past_length = kv_cache.get_seq_length()
        position_ids = torch.arange(
            past_length + position_offset,
            past_length + position_offset + seq_len,
            dtype=torch.long,
            device=dev,
        ).unsqueeze(0)

        cos, sin = self._rotary_emb(hs, position_ids)  # type: ignore[misc]

        for layer_idx in range(self.layer_start, self.layer_end):
            layer = await self._ensure_layer(layer_idx)
            hs = layer(
                hs,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=kv_cache,
                use_cache=True,
                position_embeddings=(cos, sin),
            )

        if self.has_lm_head:
            last_hs = hs[:, -1:, :]
            normed = self._norm(last_hs)  # type: ignore[union-attr]
            logits = self._lm_head(normed)  # type: ignore[union-attr]
            return {"logits": logits.squeeze(0).squeeze(0).to(torch.float16)}

        return {"hidden_states": hs.to(torch.float16)}

    @torch.no_grad()
    def forward(
        self,
        *,
        token_ids: list[int] | None = None,
        hidden_states: torch.Tensor | None = None,
        position_offset: int = 0,
        session_id: uuid.UUID,
    ) -> dict:
        """Выполнить forward pass через слои шарда (sync, eager path).

        Args:
            token_ids: список токенов (только если has_embedding и первый хоп).
            hidden_states: тензор [1, seq, hidden] от предыдущей ноды.
            position_offset: смещение позиции для RoPE (инкрементальный декодинг).
            session_id: UUID активной сессии (должна быть открыта begin_session).

        Returns:
            Если has_lm_head: {"logits": Tensor[vocab_size]} — fp16, последний токен.
            Иначе: {"hidden_states": Tensor[1, seq, hidden]} — fp16.

        Raises:
            RuntimeError: если load() не был вызван, или session_id не совпадает
                          с активной сессией, или weight_manager задан (используй forward_async).
            ValueError: если ни token_ids, ни hidden_states не переданы,
                        или переданы оба.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before forward()")
        if session_id not in self._kv_sessions:
            raise RuntimeError(
                f"session not started, call begin_session first (session_id={session_id})"
            )

        # Ровно один из аргументов должен быть передан
        has_tok = token_ids is not None and len(token_ids) > 0
        has_hs = hidden_states is not None
        if has_tok == has_hs:
            raise ValueError(
                "Exactly one of token_ids or hidden_states must be provided "
                f"(token_ids={'set' if has_tok else 'None'}, "
                f"hidden_states={'set' if has_hs else 'None'})"
            )

        if self._weight_manager is not None:
            raise RuntimeError(
                "ModelShard has weight_manager; use forward_async() for lazy loading. "
                "Or call load() in eager mode (weight_manager=None) for sync forward()."
            )

        dev = self._get_device()

        # --- Embedding (первая нода) ---
        if has_tok:
            if not self.has_embedding:
                raise ValueError("token_ids provided but shard has no embed_tokens")
            ids_t = torch.tensor([token_ids], dtype=torch.long, device=dev)
            hs = self._embed_tokens(ids_t)  # type: ignore[union-attr]
        else:
            hs = hidden_states.to(device=dev, dtype=self.dtype)  # type: ignore[union-attr]

        # Обновляем last_use_ts и получаем KV-cache
        entry = self._kv_sessions[session_id]
        entry.last_use_ts = time.monotonic()
        kv_cache = entry.cache

        # --- Прогон через decoder layers ---
        seq_len = hs.shape[1]

        # Строим position_ids для RoPE с учётом кэша и offset
        past_length = kv_cache.get_seq_length()
        position_ids = torch.arange(
            past_length + position_offset,
            past_length + position_offset + seq_len,
            dtype=torch.long,
            device=dev,
        ).unsqueeze(0)  # [1, seq]

        # Вычисляем RoPE embeddings (cos, sin) один раз для всех слоёв
        cos, sin = self._rotary_emb(hs, position_ids)  # type: ignore[misc]

        for layer in self._layers:  # type: ignore[union-attr]
            hs = layer(
                hs,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=kv_cache,
                use_cache=True,
                position_embeddings=(cos, sin),
            )

        # --- lm_head (последняя нода) ---
        if self.has_lm_head:
            # Применяем norm к последнему токену, затем lm_head
            last_hs = hs[:, -1:, :]  # [1, 1, hidden]
            normed = self._norm(last_hs)  # [1, 1, hidden]  # type: ignore[union-attr]
            logits = self._lm_head(normed)  # [1, 1, vocab]  # type: ignore[union-attr]
            return {"logits": logits.squeeze(0).squeeze(0).to(torch.float16)}  # [vocab]

        return {"hidden_states": hs.to(torch.float16)}  # [1, seq, hidden]

    def _get_device(self) -> str | torch.device:
        """Определить фактическое устройство из загруженных параметров."""
        # Приоритет: embed_tokens → norm → lm_head → layers_dict → self.device
        if self._embed_tokens is not None:
            return next(self._embed_tokens.parameters()).device
        if self._norm is not None:
            return next(self._norm.parameters()).device
        if self._lm_head is not None:
            return next(self._lm_head.parameters()).device
        if self._layers_dict:
            first = next(iter(self._layers_dict.values()))
            try:
                return next(first.parameters()).device
            except StopIteration:
                pass
        if self._layers is not None:
            try:
                return next(self._layers.parameters()).device
            except StopIteration:
                pass
        return self.device


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------


def _materialize_layer(
    meta_layer: nn.Module,
    config: LlamaConfig,
    dtype: torch.dtype,
) -> nn.Module:
    """Создать реальный (не meta) LlamaDecoderLayer с random-init весами.

    Используется для eager random-init (тесты, tiny-llama-test).
    """
    self_attn = getattr(meta_layer, "self_attn", None)
    if self_attn is not None:
        layer_idx = int(getattr(self_attn, "layer_idx", 0) or 0)
    else:
        layer_idx = 0

    layer = LlamaDecoderLayer(config, layer_idx=layer_idx)
    return layer.to(dtype)


def _make_random_layer(config: LlamaConfig, layer_idx: int, dtype: torch.dtype) -> nn.Module:
    """Создать LlamaDecoderLayer с random-init весами для fallback при ошибках чанка."""
    layer = LlamaDecoderLayer(config, layer_idx=layer_idx)
    return layer.to(dtype)


def _find_chunk_for_layer(
    chunk_paths: list[Path],
    weight_manager: WeightManager,
    model_id: str,
    layer_idx: int,
) -> Path | None:
    """Найти path к файлу чанка для layer_idx через манифест.

    Возвращает Path или None если не найдено.
    """
    # Получаем манифест из кэша weight_manager
    cached = weight_manager._manifest_cache.get(model_id)  # type: ignore[attr-defined]
    if cached is None:
        # Манифест не в кэше — используем первый chunk_path
        return chunk_paths[0] if chunk_paths else None

    manifest, _ = cached
    for chunk in manifest.chunks:
        start, end = chunk.layer_range
        if start <= layer_idx < end:
            store = weight_manager._store  # type: ignore[attr-defined]
            path = store.path_for(model_id, chunk.chunk_id)
            return path

    return None


def _load_layer_from_path(
    chunk_path: Path,
    config: LlamaConfig,
    layer_idx: int,
    dtype: torch.dtype,
) -> nn.Module:
    """Загрузить LlamaDecoderLayer из state_dict в файле чанка (E3.1).

    Формат файла: torch.save(state_dict, path), где ключи имеют вид
    "model.layers.{N}.self_attn.q_proj.weight" и т.д.

    Создаёт LlamaDecoderLayer, затем применяет нужные параметры из state_dict.
    """
    logger.debug("Loading layer %d from chunk %s", layer_idx, chunk_path)

    layer = LlamaDecoderLayer(config, layer_idx=layer_idx)
    layer = layer.to(dtype)

    try:
        state_dict = torch.load(chunk_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning(
            "Failed to load state_dict from %s: %s; using random-init for layer %d",
            chunk_path,
            exc,
            layer_idx,
        )
        return layer

    # Фильтруем только ключи для нашего layer_idx
    prefix = f"model.layers.{layer_idx}."
    layer_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    if not layer_state:
        logger.warning(
            "No keys found for layer %d in chunk %s (prefix=%s); using random-init",
            layer_idx,
            chunk_path.name,
            prefix,
        )
        return layer

    # Применяем state_dict (strict=False чтобы не падать на мисматч)
    missing, unexpected = layer.load_state_dict(layer_state, strict=False)
    if missing:
        logger.warning("Layer %d: missing keys in state_dict: %s", layer_idx, missing[:5])
    if unexpected:
        logger.warning("Layer %d: unexpected keys in state_dict: %s", layer_idx, unexpected[:5])

    return layer


def _convert_to_int8(layer: nn.Module) -> nn.Module:
    """Конвертировать nn.Linear → bnb.nn.Linear8bitLt (E3.2, ADR-0034).

    Проходит по всем подмодулям; заменяет nn.Linear на Linear8bitLt.
    Не заменяет слои с bias (bias=True), так как они редки в Llama.

    Требует HAS_BNB=True (проверять снаружи).
    """
    assert bnb is not None, "bitsandbytes not available"

    def _replace_linear(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                new_linear = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                )
                # Копируем веса (из fp16 → int8 будет сделано при первом forward bnb)
                new_linear.weight = child.weight
                if child.bias is not None:
                    new_linear.bias = child.bias
                setattr(module, name, new_linear)
            else:
                _replace_linear(child)

    _replace_linear(layer)
    return layer
