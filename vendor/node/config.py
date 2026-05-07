"""Stage 3: загрузка TOML-конфига ноды.

Читает конфиг через tomllib (stdlib Python 3.11+), валидирует через Pydantic v2.
Отсутствие обязательного поля → ConfigError с указанием ключа.
data_dir расширяется (~, relative) и создаётся автоматически.

Stage 3 добавляет секцию [network]:
  quic_port = 9001          — UDP-порт QUIC-сервера
  stun_servers = [...]      — override STUN-серверов (ADR-0015)

Stage 4 (ADR-0020): tracker_url используется ОДНОВРЕМЕННО как HTTP base URL
  и как tracker_url для подписи challenge-response. canonicalize_tracker_url()
  из node/tracker_client.py приводит его к канонической форме совпадающей с
  трекером (lowercase scheme+host, без trailing slash).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shared.constants import STUN_SERVERS_DEFAULT


class ConfigError(ValueError):
    """Ошибка загрузки или валидации TOML-конфига ноды."""


class InferenceConfig(BaseModel):
    """Секция [inference] TOML-конфига ноды (Stage 6, ADR-0040).

    Параметры multi-session KV-cache и ресурс-лимиты inference.
    """

    model_config = ConfigDict(extra="forbid")

    kv_ttl_seconds: int = Field(
        default=300,
        ge=1,
        description="TTL KV-cache сессии (секунды неактивности → evict, ADR-0040)",
    )
    kv_quota_fraction: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Доля VRAM под KV-cache (от vram_gb конфига, ADR-0040)",
    )
    kv_quota_max_gb: float = Field(
        default=8.0,
        gt=0.0,
        description="Максимум VRAM под KV-cache в ГБ (верхняя крышка, ADR-0040)",
    )
    kv_max_sessions: int = Field(
        default=64,
        ge=1,
        le=1024,
        description="Максимальное число KV-cache сессий (safety cap, ADR-0040)",
    )
    kv_sweep_interval_s: int = Field(
        default=60,
        ge=5,
        description="Интервал фоновой чистки истёкших KV-cache сессий (секунды, ADR-0040)",
    )


class UploadConfig(BaseModel):
    """Секция [upload] TOML-конфига ноды (Stage 6, ADR-0043).

    Параметры rate-limit на peer-uploader.
    """

    model_config = ConfigDict(extra="forbid")

    max_concurrent_per_peer: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Максимум одновременных weight-chunk стримов от одного peer_id (ADR-0043)",
    )
    max_bytes_per_sec_per_peer: int = Field(
        default=17_000_000,
        ge=1,
        description="Token bucket refill rate в байтах/сек на peer (ADR-0043, ≈1 ГБ/мин)",
    )
    burst_bytes: int = Field(
        default=268_435_456,
        ge=1,
        description="Capacity token bucket'а в байтах (ADR-0043, 256 МБ = 1 чанк §5.2)",
    )
    max_concurrent_global: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Глобальный лимит параллельных uploads (ADR-0043)",
    )


class WeightsConfig(BaseModel):
    """Секция [weights] TOML-конфига ноды (Stage 5, ADR-0033, ADR-0035)."""

    model_config = ConfigDict(extra="forbid")

    cache_dir: Path = Field(
        default_factory=lambda: Path("~/.llm-swarm/weights").expanduser(),
        description="Каталог для кэша чанков весов (Stage 5, ADR-0033)",
    )
    fetch_concurrency: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Максимальное число параллельных загрузок чанков (E2.3)",
    )
    upload_concurrency: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Максимальное число параллельных раздач чанков (E2.3)",
    )
    manifest_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="TTL кэша манифеста модели в памяти (секунды, 0 = без кэша)",
    )
    weights_dtype: str = Field(
        default="int8",
        description=(
            "Dtype загружаемых весов: 'fp16' | 'int8'. "
            "'int8' включает bitsandbytes Linear8bitLt для Llama-2/3 (ADR-0034). "
            "Для tiny-llama-test используется random-init, поле игнорируется."
        ),
    )


class NetworkConfig(BaseModel):
    """Секция [network] TOML-конфига ноды (Stage 3, ADR-0015)."""

    model_config = ConfigDict(extra="forbid")

    quic_port: int = Field(
        default=9001,
        ge=1,
        le=65535,
        description="UDP-порт QUIC-сервера (ADR-0012, ADR-0013)",
    )
    stun_servers: list[str] = Field(
        default_factory=lambda: list(STUN_SERVERS_DEFAULT),
        description=(
            "Список STUN-серверов host:port для NAT probe (ADR-0015). "
            "По умолчанию: stun.l.google.com:19302, stun.cloudflare.com:3478."
        ),
    )


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracker_url: str = Field(description="URL трекера, например http://localhost:8000")
    model_id: str = Field(description="Идентификатор модели, например llama-2-70b")
    layers_start: int = Field(ge=0, description="Первый слой, обслуживаемый нодой (включительно)")
    layers_end: int = Field(gt=0, description="Последний слой, обслуживаемый нодой (исключительно)")
    data_dir: Path = Field(description="Каталог для хранения identity.key и прочих данных ноды")
    addr: str = Field(
        description=(
            "Публичный адрес ноды (декларативный). "
            "Stage 3: STUN определяет реальный публичный endpoint, "
            "это поле используется только как fallback если STUN недоступен."
        )
    )
    vram_gb: float = Field(gt=0, description="Объём VRAM в ГБ")
    bandwidth_mbps: float = Field(gt=0, description="Полоса пропускания в Мбит/с")
    network: NetworkConfig = Field(
        default_factory=NetworkConfig,
        description="Сетевые настройки (QUIC, STUN, relay — Stage 3, ADR-0012..0017)",
    )
    weights: WeightsConfig = Field(
        default_factory=WeightsConfig,
        description="Настройки весового слоя (Stage 5, ADR-0033, ADR-0035)",
    )
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Настройки KV-cache и inference (Stage 6, ADR-0040)",
    )
    upload: UploadConfig = Field(
        default_factory=UploadConfig,
        description="Настройки rate-limit на peer-uploader (Stage 6, ADR-0043)",
    )


def load_config(path: Path) -> NodeConfig:
    """Загрузить и провалидировать TOML-конфиг ноды.

    Args:
        path: Путь к .toml-файлу конфига.

    Returns:
        Провалидированный NodeConfig.

    Raises:
        ConfigError: Если файл не найден, TOML невалиден, или не хватает обязательных полей.
    """
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"TOML parse error in {path}: {exc}") from exc

    try:
        cfg = NodeConfig.model_validate(raw)
    except ValidationError as exc:
        # Собираем список недостающих/некорректных полей для понятного сообщения.
        missing = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            missing.append(f"{loc}: {err['msg']}")
        raise ConfigError("Invalid node config:\n" + "\n".join(missing)) from exc

    # Расширяем ~ и relative paths, создаём каталог.
    expanded = Path(cfg.data_dir).expanduser().resolve()
    expanded.mkdir(parents=True, exist_ok=True)

    # Pydantic frozen-модель — пересоздаём с расширенным data_dir.
    return cfg.model_copy(update={"data_dir": expanded})
