"""Pydantic v2 DTO для tracker API (§3.2 спеки) и envelope активаций (ADR-0009, §7.2).

Модели — immutable (frozen=True), strict (extra="forbid").
Все поля строго совпадают с §3.2; пояснения по non-obvious полям — inline.

Stage 1 ограничения (ADR-0005, ADR-0006):
  - RegisterResponse.token — opaque UUIDv4; challenge-response — Stage 4.
  - RegisterResponse.assigned_layers == RegisterRequest.layers — Stage 5.
  - ChainHop.score всегда 0.0 — Stage 4.
  - HeartbeatRequest.served_tokens принимается, но трекером игнорируется (§5.2, Stage 4).

Stage 2 (ADR-0009):
  - ForwardEnvelope — бинарный envelope активаций с цепочкой подписей.
  - serialize_envelope_for_signing / serialize_envelope / parse_envelope.

Stage 2 (ADR-0010, ADR-0011):
  - NodeRoute — описание одного хопа для клиентского запроса.
  - FirstHopRequest — первичный запрос клиента первой ноде (token_ids, маршрут).

Stage 3 (ADR-0012, ADR-0013, ADR-0017):
  - parse_multiaddr / format_multiaddr — узкое подмножество multiaddr (ADR-0013).
  - MultiAddr — dataclass результата парсинга.
  - SessionInit — первое сообщение QUIC-соединения (ADR-0017).
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Regex для валидации multiaddr (ADR-0013, §3.2).
# Допускается только /ip4/<addr>/udp/<port>/quic-v1/p2p/<base58_peer_id>
# или /ip6/<addr>/udp/<port>/quic-v1/p2p/<base58_peer_id>.
# base58 charset: 1-9 A-H J-N P-Z a-k m-z (исключены 0, I, O, l).
_MULTIADDR_RE = re.compile(r"^/ip[46]/[^/]+/udp/\d+/quic-v1/p2p/[1-9A-HJ-NP-Za-km-z]+$")

# Domain tag для отделения forward-envelope подписей от других протоколов (ADR-0009).
# Ровно 20 байт ASCII — assert проверяет это при импорте.
DOMAIN_TAG = b"llm-swarm/v1/forward"
assert len(DOMAIN_TAG) == 20, f"DOMAIN_TAG must be exactly 20 bytes, got {len(DOMAIN_TAG)}"


class LayerRange(BaseModel):
    """Диапазон слоёв [start, end), 0-based.

    Пример: start=0, end=32 означает слои 0..31 включительно.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0, description="Первый слой диапазона (включительно)")
    end: int = Field(gt=0, description="Последний слой диапазона (исключительно)")

    @field_validator("end")
    @classmethod
    def _end_gt_start(cls, v: int, info) -> int:
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"end ({v}) must be greater than start ({start})")
        return v


class RegisterRequest(BaseModel):
    """Тело POST /api/v1/nodes/register (§3.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    peer_id: str = Field(description="base58(sha256(public_key)) — выводится из ключа, §4.5")
    public_key: str = Field(description="base58-encoded 32-byte Ed25519 public key")
    layers: LayerRange = Field(description="Диапазон слоёв, которые нода берётся обслуживать")
    model_id: str = Field(description="Идентификатор модели, например 'llama-2-70b'")
    vram_gb: float = Field(gt=0, description="Объём VRAM в ГБ")
    bandwidth_mbps: float = Field(gt=0, description="Полоса пропускания в Мбит/с")
    addr: str = Field(
        description=(
            "Адрес ноды в формате multiaddr (ADR-0013, §3.2): "
            "/ip4/<host>/udp/<port>/quic-v1/p2p/<peer_id> или /ip6/..."
        )
    )

    @field_validator("addr")
    @classmethod
    def _addr_is_multiaddr(cls, v: str) -> str:
        if not _MULTIADDR_RE.match(v):
            raise ValueError(
                f"addr must be a multiaddr of the form "
                f"/ip4/<host>/udp/<port>/quic-v1/p2p/<peer_id> "
                f"(or /ip6/...), got: {v!r}"
            )
        return v


class RegisterResponse(BaseModel):
    """Ответ POST /api/v1/nodes/register (§3.2).

    Stage 4 (ADR-0020): register теперь возвращает challenge вместо session_token.
    Поле token оставлено для backward-compat с тестами Stage 1-3, но используется
    только при legacy-режиме (когда challenge_id is None).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Stage 1-3: opaque UUIDv4 (ADR-0005). Stage 4: challenge_id/challenge заменяют.
    token: str | None = Field(
        default=None,
        description=(
            "Opaque UUIDv4 сессионный токен (ADR-0005). "
            "Stage 4: None — используется challenge-response (ADR-0020)."
        ),
    )
    assigned_layers: LayerRange = Field(
        description=("Слои, подтверждённые трекером. Stage 1: == request.layers (ADR-0006).")
    )
    # Stage 4 (ADR-0020): challenge-response fields
    challenge_id: uuid.UUID | None = Field(
        default=None,
        description="UUID challenge для последующего /auth (ADR-0020). Stage 4+.",
    )
    challenge: str | None = Field(
        default=None,
        description="base64-encoded 32-byte challenge (ADR-0020). Stage 4+.",
    )
    ttl_seconds: int | None = Field(
        default=None,
        description="TTL challenge в секундах (ADR-0020, 60 сек). Stage 4+.",
    )


# ---------------------------------------------------------------------------
# Stage 4 (ADR-0020): Challenge-response auth DTOs
# ---------------------------------------------------------------------------


class AuthRequest(BaseModel):
    """Тело POST /api/v1/nodes/auth (ADR-0020)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    challenge_id: uuid.UUID = Field(description="UUID challenge из RegisterResponse")
    signature: str = Field(description="base64-encoded 64-byte Ed25519 подпись payload")
    issued_at_ms: int = Field(description="Unix timestamp в миллисекундах (момент подписи нодой)")


class AuthResponse(BaseModel):
    """Ответ POST /api/v1/nodes/auth (ADR-0020)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_token: str = Field(description="Opaque session token для heartbeat")
    expires_at: datetime = Field(description="Момент истечения session_token (ISO8601)")


class HeartbeatRequest(BaseModel):
    """Тело POST /api/v1/nodes/heartbeat (§3.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    peer_id: str
    token: str = Field(description="Сессионный токен из RegisterResponse")
    served_tokens: int = Field(
        ge=0,
        default=0,
        description=(
            "Токены, обслуженные нодой с последнего heartbeat. "
            "# TODO Stage 4: accounting (§5.2). Stage 1: трекер принимает, игнорирует."
        ),
    )
    gpu_temp: float | None = Field(default=None, description="Температура GPU (°C), опционально")
    vram_free_gb: float | None = Field(
        default=None,
        ge=0,
        description="Свободный VRAM в ГБ, опционально",
    )
    # Stage 6 (ADR-0038): resource monitor fields
    degraded: bool = Field(
        default=False,
        description=(
            "True если нода в деградированном состоянии (ADR-0038, §4.6). "
            "Трекер исключает деградированные ноды из маршрутизации."
        ),
    )
    degraded_reasons: list[str] = Field(
        default_factory=list,
        description=(
            "Список причин деградации: 'gpu_temp', 'vram', 'ram', 'bandwidth' (ADR-0038). "
            "Пустой список если degraded=False."
        ),
    )
    kv_active_sessions: int = Field(
        default=0,
        ge=0,
        description="Число активных KV-cache сессий (ADR-0040, §4.2)",
    )
    kv_used_bytes: int = Field(
        default=0,
        ge=0,
        description="Байты занятые KV-cache сессиями (ADR-0040, §4.2)",
    )
    vram_used_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Используемый VRAM в байтах (ADR-0038). None если GPU недоступен.",
    )
    ram_used_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Используемая RAM в байтах (ADR-0038). None если метрика недоступна.",
    )


class PunchPending(BaseModel):
    """Pending hole-punch record: нода B получает из heartbeat response (ADR-0027).

    Трекер кладёт этот объект в Redis ключ punch_pending:{B_peer_id} при обработке
    POST /api/v1/punch от ноды A.  Нода B забирает при следующем heartbeat.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_peer_id: str = Field(description="peer_id ноды-инициатора (A)")
    addr: str = Field(
        description=(
            "Observed multiaddr ноды A (IP:port, который трекер видит у инициатора). "
            "Формат: /ip4/<host>/udp/<port>/quic-v1/p2p/<peer_id>"
        )
    )
    rendezvous_at_unix_ms: int = Field(description="Unix timestamp (мс) момента simultaneous open")


class HeartbeatResponse(BaseModel):
    """Ответ POST /api/v1/nodes/heartbeat (§3.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool = True
    pending_punch: PunchPending | None = Field(
        default=None,
        description=(
            "Pending hole-punch от другой ноды (ADR-0027). "
            "None — ничего нет. Нода обязана выполнить SYN к addr в rendezvous_at_unix_ms."
        ),
    )


class PunchRequest(BaseModel):
    """Тело POST /api/v1/punch (ADR-0027).

    Инициатор (нода A) просит трекер уведомить ноду B и получить rendezvous_at.
    Bearer-токен в Authorization — session_token ноды A.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    self_peer_id: str = Field(description="peer_id инициатора (A)")
    peer_id: str = Field(description="peer_id цели hole punch (B)")
    session_token: str = Field(
        description="session_token ноды A (для auth; дублирует Bearer для удобства)"
    )


class PunchResponse(BaseModel):
    """Ответ POST /api/v1/punch (ADR-0027).

    Инициатор получает addr ноды B и момент simultaneous open.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    peer_addr: str = Field(description="multiaddr ноды B (цели)")
    self_addr_observed: str = Field(
        description=(
            "Observed multiaddr ноды A с точки зрения трекера "
            "(IP:port, который трекер видит у инициатора). "
            "Формат: /ip4/<host>/udp/<port>/quic-v1/p2p/<self_peer_id>"
        )
    )
    rendezvous_at_unix_ms: int = Field(
        description=(
            "Unix timestamp (мс) момента simultaneous open (NOW() + PUNCH_RENDEZVOUS_DELAY_MS)"
        )
    )


class ChainHop(BaseModel):
    """Один узел в маршруте инференса (§3.2 GET /route response)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    peer_id: str
    addr: str = Field(description="multiaddr ноды (ADR-0013): /ip4/.../udp/.../quic-v1/p2p/...")
    layers: LayerRange
    score: float = Field(
        default=0.0,
        description=("Репутационный score (§3.5). Stage 1-3: всегда 0.0 (ADR-0006)."),
    )
    pubkey_b58: str = Field(
        default="",
        description=(
            "base58-encoded 32-byte Ed25519 public key (ADR-0011, S4-A4). "
            "Stage 4+: трекер заполняет из nodes.public_key. "
            "Пустая строка — backward-compat для Stage 1-3 тестов."
        ),
    )


class RouteResponse(BaseModel):
    """Ответ GET /api/v1/route (§3.2).

    Stage 4 (ADR-0021): дополнен session_id, relay_token, expires_at.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chain: list[ChainHop] = Field(description="Цепочка нод, покрывающая все слои модели")
    # Stage 4 (ADR-0021): forward session fields
    session_id: uuid.UUID | None = Field(
        default=None,
        description="UUID forward-сессии (ADR-0021). Stage 4+.",
    )
    relay_token: str | None = Field(
        default=None,
        description="Relay token для WS relay binding (ADR-0021). Stage 4+.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Время истечения forward-сессии (ADR-0021, NOW()+1h). Stage 4+.",
    )


class NodeRoute(BaseModel):
    """Описание одного хопа в маршруте для клиентского запроса (ADR-0010).

    Клиент вкладывает список NodeRoute в FirstHopRequest, чтобы каждая нода
    знала, кому передавать активации дальше.

    Stage 2: использует поле url (HTTP).
    Stage 3: использует поле addr (multiaddr QUIC, ADR-0013). url остаётся
             для backward-compat с тестами Stage 2.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    peer_id: str = Field(description="base58(sha256(public_key)) ноды")
    url: str = Field(description="HTTP-адрес ноды в формате 'http://host:port' (Stage 2 legacy)")
    pubkey_b58: str = Field(description="base58-encoded 32-byte Ed25519 public key ноды")
    addr: str | None = Field(
        default=None,
        description=(
            "Multiaddr ноды (ADR-0013, Stage 3): "
            "/ip4/<host>/udp/<port>/quic-v1/p2p/<peer_id> или /ip6/... "
            "None — нода за relay (symmetric NAT), используется relay-fallback."
        ),
    )


class FirstHopRequest(BaseModel):
    """Первичный запрос клиента первой ноде цепочки (ADR-0010, §6.1).

    Клиент шлёт первой ноде token_ids вместо активаций (тонкий клиент).
    Первая нода выполняет embedding и передаёт активации дальше по route.

    client_peer_id — опциональный (ADR-0011): Ed25519-identity клиента
    появится на Stage 4 вместе с accounting и challenge-response.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: uuid.UUID = Field(description="UUID текущей сессии инференса")
    token_ids: list[int] = Field(description="Токенизированный промпт (u32 в практическом смысле)")
    route: list[NodeRoute] = Field(
        description="Цепочка нод с peer_id, URL и pubkey для верификации подписей"
    )
    client_peer_id: str | None = Field(
        default=None,
        description=(
            "Peer ID клиента (ADR-0011). "
            "Stage 2: опциональный, для ratio accounting. "
            "# TODO Stage 4: обязательный + Ed25519 challenge-response."
        ),
    )


class ForwardEnvelope(BaseModel):
    """Envelope активаций для передачи между нодами (ADR-0009, §7.2).

    Wire-формат (бинарный, детерминированный):
        session_id(16) || hop_index(u16 BE) || prev_sig(64)
        || layer_start(u16 BE) || layer_end(u16 BE)
        || blob_len(u32 BE) || activations_blob || sig(64)

    Для подписи используется serialize_envelope_for_signing().
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: uuid.UUID = Field(description="UUID сессии инференса")
    hop_index: int = Field(
        ge=0, le=65535, description="Порядковый номер хопа в цепочке (u16, 0-based)"
    )
    prev_sig: bytes = Field(
        description="Ed25519-подпись предыдущего хопа, 64 байта; для hop 0 = b'\\x00' * 64"
    )
    layer_start: int = Field(
        ge=0, le=65535, description="Начало диапазона слоёв (u16, включительно)"
    )
    layer_end: int = Field(ge=0, le=65535, description="Конец диапазона слоёв (u16, исключительно)")
    activations_blob: bytes = Field(
        description="Сериализованные активации (shared/quant.py wire-формат)"
    )

    @field_validator("prev_sig")
    @classmethod
    def _prev_sig_must_be_64_bytes(cls, v: bytes) -> bytes:
        if len(v) != 64:
            raise ValueError(f"prev_sig must be exactly 64 bytes, got {len(v)}")
        return v

    @field_validator("layer_end")
    @classmethod
    def _layer_end_gt_layer_start(cls, v: int, info) -> int:
        layer_start = info.data.get("layer_start")
        if layer_start is not None and v <= layer_start:
            raise ValueError(f"layer_end ({v}) must be greater than layer_start ({layer_start})")
        return v


def serialize_envelope_for_signing(env: ForwardEnvelope) -> bytes:
    """Сформировать байтовый буфер для подписи (sha256 этого буфера подписывается Ed25519).

    Конкатенация (big-endian для всех integer-полей вне activations_blob):
        domain_tag(20) || session_id(16) || hop_index(u16 BE)
        || prev_sig(64) || layer_start(u16 BE) || layer_end(u16 BE)
        || activations_blob
    """
    return (
        DOMAIN_TAG
        + env.session_id.bytes  # UUID в сетевом порядке (big-endian), 16 байт
        + struct.pack(">H", env.hop_index)
        + env.prev_sig
        + struct.pack(">H", env.layer_start)
        + struct.pack(">H", env.layer_end)
        + env.activations_blob
    )


def serialize_envelope(env: ForwardEnvelope, sig: bytes) -> bytes:
    """Сериализовать envelope + подпись в бинарный буфер для HTTP-тела.

    Формат:
        session_id(16) || hop_index(u16 BE) || prev_sig(64)
        || layer_start(u16 BE) || layer_end(u16 BE)
        || blob_len(u32 BE) || activations_blob || sig(64)

    Args:
        env: ForwardEnvelope.
        sig: 64-байтовая Ed25519-подпись (результат sign_envelope).

    Returns:
        Бинарный буфер.

    Raises:
        ValueError: если sig не ровно 64 байта.
    """
    if len(sig) != 64:
        raise ValueError(f"sig must be exactly 64 bytes, got {len(sig)}")
    blob_len = len(env.activations_blob)
    return (
        env.session_id.bytes
        + struct.pack(">H", env.hop_index)
        + env.prev_sig
        + struct.pack(">HH", env.layer_start, env.layer_end)
        + struct.pack(">I", blob_len)
        + env.activations_blob
        + sig
    )


def parse_envelope(blob: bytes) -> tuple[ForwardEnvelope, bytes]:
    """Десериализовать бинарный буфер в (ForwardEnvelope, sig).

    Обратная операция к serialize_envelope.

    Args:
        blob: бинарный буфер из HTTP-тела.

    Returns:
        Кортеж (ForwardEnvelope, sig_bytes).

    Raises:
        ValueError: если буфер слишком короткий или имеет некорректную структуру.
    """
    # Минимальный размер: session_id(16) + hop_index(2) + prev_sig(64)
    #   + layer_start(2) + layer_end(2) + blob_len(4) + activations_blob(>=0) + sig(64)
    _FIXED_HEADER = 16 + 2 + 64 + 2 + 2 + 4  # 90 bytes
    _SIG_LEN = 64
    min_size = _FIXED_HEADER + _SIG_LEN
    if len(blob) < min_size:
        raise ValueError(f"blob too short: expected at least {min_size} bytes, got {len(blob)}")

    offset = 0
    session_id = uuid.UUID(bytes=blob[offset : offset + 16])
    offset += 16

    (hop_index,) = struct.unpack_from(">H", blob, offset)
    offset += 2

    prev_sig = blob[offset : offset + 64]
    offset += 64

    (layer_start, layer_end) = struct.unpack_from(">HH", blob, offset)
    offset += 4

    (blob_len,) = struct.unpack_from(">I", blob, offset)
    offset += 4

    if len(blob) < offset + blob_len + _SIG_LEN:
        raise ValueError(
            f"blob too short for activations_blob ({blob_len} bytes) + sig: "
            f"remaining={len(blob) - offset}"
        )

    activations_blob = blob[offset : offset + blob_len]
    offset += blob_len

    sig = blob[offset : offset + _SIG_LEN]

    env = ForwardEnvelope(
        session_id=session_id,
        hop_index=hop_index,
        prev_sig=prev_sig,
        layer_start=layer_start,
        layer_end=layer_end,
        activations_blob=activations_blob,
    )
    return env, sig


def serialize_chain(envelopes: list[tuple[ForwardEnvelope, bytes]]) -> bytes:
    """Сериализовать цепочку envelope+sig в бинарный буфер.

    Формат:
        count(u16 BE) || for each: env_blob_len(u32 BE) || serialize_envelope(env, sig)

    Envelopes упорядочены по hop_index ASC (hop 0 первый).

    Args:
        envelopes: список кортежей (ForwardEnvelope, sig_bytes).

    Returns:
        Бинарный буфер.
    """
    count = len(envelopes)
    parts: list[bytes] = [struct.pack(">H", count)]
    for env, sig in envelopes:
        entry = serialize_envelope(env, sig)
        parts.append(struct.pack(">I", len(entry)))
        parts.append(entry)
    return b"".join(parts)


def parse_chain(blob: bytes) -> list[tuple[ForwardEnvelope, bytes]]:
    """Десериализовать бинарный буфер цепочки в список (ForwardEnvelope, sig).

    Обратная операция к serialize_chain.

    Args:
        blob: бинарный буфер.

    Returns:
        Список кортежей (ForwardEnvelope, sig_bytes), упорядоченный по hop_index ASC.

    Raises:
        ValueError: если буфер слишком короткий или структура некорректна.
    """
    if len(blob) < 2:
        raise ValueError(f"chain blob too short: expected at least 2 bytes, got {len(blob)}")

    offset = 0
    (count,) = struct.unpack_from(">H", blob, offset)
    offset += 2

    result: list[tuple[ForwardEnvelope, bytes]] = []
    for i in range(count):
        if offset + 4 > len(blob):
            raise ValueError(
                f"chain blob truncated at entry {i}: "
                f"cannot read env_blob_len, offset={offset}, len={len(blob)}"
            )
        (entry_len,) = struct.unpack_from(">I", blob, offset)
        offset += 4

        if offset + entry_len > len(blob):
            raise ValueError(
                f"chain blob truncated at entry {i}: "
                f"entry_len={entry_len} but only {len(blob) - offset} bytes remain"
            )
        entry_bytes = blob[offset : offset + entry_len]
        offset += entry_len

        env, sig = parse_envelope(entry_bytes)
        result.append((env, sig))

    return result


def canonical_json(model: BaseModel) -> bytes:
    """Стабильный байтовый JSON для будущих подписей (Stage 2).

    Инвариант 6 из плана: детерминированный порядок байтов в DTO.
    На Stage 2 этот байтовый blob будет входить в sha256 перед Ed25519-подписью.

    Гарантии:
      - sort_keys=True — ключи в алфавитном порядке, независимо от порядка полей в модели.
      - separators=(",", ":") — без пробелов, минимальный JSON.
      - UTF-8 encoding.
      - Два вызова с одинаковым объектом → идентичные байты.
    """
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Stage 3: Multiaddr (ADR-0013)
# ---------------------------------------------------------------------------


class MultiAddr:
    """Результат парсинга multiaddr ноды (ADR-0013).

    Поддерживается строго узкое подмножество:
        /ip4/<host>/udp/<port>/quic-v1/p2p/<peer_id>
        /ip6/<host>/udp/<port>/quic-v1/p2p/<peer_id>

    Атрибуты:
        proto: "ip4" или "ip6".
        host: строковый IP-адрес.
        port: UDP-порт (1–65535).
        peer_id: base58(sha256(public_key)).
    """

    __slots__ = ("host", "peer_id", "port", "proto")

    def __init__(self, proto: str, host: str, port: int, peer_id: str) -> None:
        self.proto = proto
        self.host = host
        self.port = port
        self.peer_id = peer_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiAddr):
            return NotImplemented
        return (
            self.proto == other.proto
            and self.host == other.host
            and self.port == other.port
            and self.peer_id == other.peer_id
        )

    def __repr__(self) -> str:
        return (
            f"MultiAddr({self.proto!r}, host={self.host!r}, "
            f"port={self.port}, peer_id={self.peer_id!r})"
        )


def parse_multiaddr(addr: str) -> MultiAddr:
    """Распарсить multiaddr-строку в MultiAddr.

    Допускается только: /ip4/<host>/udp/<port>/quic-v1/p2p/<peer_id>
                     или /ip6/<host>/udp/<port>/quic-v1/p2p/<peer_id>

    Args:
        addr: multiaddr-строка.

    Returns:
        MultiAddr с полями proto, host, port, peer_id.

    Raises:
        ValueError: если строка не соответствует ожидаемому формату.
    """
    if not _MULTIADDR_RE.match(addr):
        raise ValueError(
            f"Invalid multiaddr: expected /ip(4|6)/<host>/udp/<port>/quic-v1/p2p/<peer_id>, "
            f"got: {addr!r}"
        )
    # /ip4/1.2.3.4/udp/9001/quic-v1/p2p/Abc123
    parts = addr.split("/")
    # parts: ['', 'ip4', '1.2.3.4', 'udp', '9001', 'quic-v1', 'p2p', 'Abc123']
    if len(parts) != 8:
        raise ValueError(f"Unexpected multiaddr structure: {addr!r}")
    proto = parts[1]  # 'ip4' or 'ip6'
    host = parts[2]
    try:
        port = int(parts[4])
    except ValueError as exc:
        raise ValueError(f"Invalid port in multiaddr: {parts[4]!r}") from exc
    if not (1 <= port <= 65535):
        raise ValueError(f"Port out of range [1, 65535]: {port}")
    peer_id = parts[7]
    return MultiAddr(proto=proto, host=host, port=port, peer_id=peer_id)


def format_multiaddr(proto: str, host: str, port: int, peer_id: str) -> str:
    """Сформировать multiaddr-строку из компонентов.

    Args:
        proto: "ip4" или "ip6".
        host: IP-адрес.
        port: UDP-порт.
        peer_id: base58-encoded peer_id.

    Returns:
        Строка вида /ip4/<host>/udp/<port>/quic-v1/p2p/<peer_id>.

    Raises:
        ValueError: если proto неизвестен или полученная строка не проходит валидацию.
    """
    if proto not in ("ip4", "ip6"):
        raise ValueError(f"proto must be 'ip4' or 'ip6', got: {proto!r}")
    if not (1 <= port <= 65535):
        raise ValueError(f"Port out of range [1, 65535]: {port}")
    result = f"/{proto}/{host}/udp/{port}/quic-v1/p2p/{peer_id}"
    if not _MULTIADDR_RE.match(result):
        raise ValueError(f"Resulting multiaddr failed validation: {result!r}")
    return result


# ---------------------------------------------------------------------------
# Stage 3: SessionInit DTO (ADR-0017)
# ---------------------------------------------------------------------------


class SessionInit(BaseModel):
    """Первое сообщение QUIC-соединения (ADR-0017, §7.2).

    Клиент шлёт первой ноде как первый QUIC-фрейм сессии.
    Нода проверяет session_token и сверяет peer_id из маршрута с TLS SPKI.

    Формат сериализации — JSON (UTF-8), предваряется u32 BE длиной в wire-протоколе:
        length(u32 BE) || json_bytes

    session_token — opaque UUID до Stage 4 (ADR-0005).
    route — полная цепочка хопов как в FirstHopRequest.

    Stage 4: добавятся поля challenge/response для Ed25519 challenge-response.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: uuid.UUID = Field(description="UUID текущей сессии инференса")
    session_token: str = Field(
        description=(
            "Opaque сессионный токен (ADR-0005, ADR-0017). "
            "Stage 3: UUID, выданный трекером при /api/v1/route. "
            "Stage 4: заменяется Ed25519 challenge-response."
        )
    )
    route: list[NodeRoute] = Field(
        description="Полная цепочка нод в маршруте (все хопы, включая текущий)"
    )
    hop_index: int | None = Field(
        default=None,
        description=(
            "Индекс текущей ноды в цепочке (0-based). "
            "Устанавливается нодой при пересылке SessionInit следующей ноде. "
            "None = клиент отправляет первой ноде (hop 0)."
        ),
    )
    peer_id_self: str | None = Field(
        default=None,
        description=(
            "peer_id ноды, которая пересылает SessionInit (для relay-режима). "
            "None = клиент отправляет."
        ),
    )


# ---------------------------------------------------------------------------
# Stage 4 B2: Strike report DTOs (ADR-0024)
# ---------------------------------------------------------------------------


class StrikeEvidence(BaseModel):
    """Доказательство некорректного spot-check результата (ADR-0024).

    envelope_chain — список base64-encoded сериализованных envelope+sig
    (serialize_envelope output), ordered by hop_index ASC.
    Трекер верифицирует цепочку до hop_index включительно через verify_chain.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_hash: str = Field(description="SHA256 hex golden sample input")
    expected_blob_sha256: str = Field(description="SHA256 hex ожидаемого activations blob")
    got_blob_sha256: str = Field(description="SHA256 hex полученного activations blob")
    l2_distance: float = Field(ge=0.0, description="Относительная L2-норма отклонения")
    hop_index: int = Field(ge=0, description="Индекс виновного хопа в цепочке")
    envelope_chain: list[str] = Field(
        description="base64-encoded serialize_envelope(env,sig) для каждого хопа [0..hop_index]"
    )


class StrikeReportRequest(BaseModel):
    """Тело POST /api/v1/strikes/report (ADR-0024, §3.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: uuid.UUID = Field(description="UUID forward-сессии")
    reporter_peer_id: str = Field(description="peer_id репортёра (клиент сессии)")
    offender_peer_id: str = Field(description="peer_id виновной ноды")
    evidence: StrikeEvidence
    reporter_signature: str = Field(
        description=(
            "base64-encoded Ed25519 подпись над canonical evidence body "
            "(без поля reporter_signature), domain_tag='llm-swarm/v1/strike'"
        )
    )


class StrikeReportResponse(BaseModel):
    """Ответ POST /api/v1/strikes/report (ADR-0024, §3.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strike_applied: bool
    reason: str | None = None
    current_strike_count_24h: int = Field(default=0)
    quarantined: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Stage 4 B3: Accounting report DTOs (ADR-0022)
# ---------------------------------------------------------------------------


class SpotCheckResult(BaseModel):
    """Результат spot-check (§3.2, ADR-0022, опционален в accounting report)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checked_hop: int = Field(ge=0, description="Индекс проверенного хопа")
    passed: bool = Field(description="True если spot-check прошёл, False если нет")
    l2_distance: float = Field(ge=0.0, description="Относительная L2-норма отклонения")


class AccountingReportRequest(BaseModel):
    """Тело POST /api/v1/accounting/report (ADR-0022, §3.2).

    Stage 6 (ADR-0042): поле type позволяет различать два типа отчётов:
      - "forward"        — стандартный инференс-отчёт (Stage 4, по умолчанию).
      - "chunk_transfer" — batch ChunkReceipt'ов от uploader'а (Stage 6).

    При type="chunk_transfer":
      - chunk_receipts обязателен и содержит список ChunkReceipt v2.
      - chain_peer_ids / tokens_consumed / spot_check_result игнорируются трекером
        (должны быть пустыми/нулевыми, но не обязательны структурно).
      - client_peer_id = peer_id uploader'а, отправившего receipts.
      - client_signature = Ed25519-подпись над canonical body без client_signature,
        domain_tag='llm-swarm/v1/accounting' (та же функция accounting_canonical_bytes).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: uuid.UUID = Field(description="UUID forward-сессии")
    client_peer_id: str = Field(description="peer_id клиента (или uploader'а при chunk_transfer)")
    chain_peer_ids: list[str] = Field(description="peer_id нод цепочки [hop0, hop1, ...]")
    tokens_consumed: int = Field(ge=0, description="Токены, обслуженные цепочкой")
    spot_check_result: SpotCheckResult | None = Field(
        default=None,
        description="Результат spot-check (опционально)",
    )
    client_signature: str = Field(
        description=(
            "base64-encoded Ed25519 подпись над canonical body без поля client_signature, "
            "domain_tag='llm-swarm/v1/accounting'"
        )
    )
    # Stage 6 (ADR-0042): тип отчёта
    type: str = Field(
        default="forward",
        description=(
            "Тип accounting report: 'forward' (инференс, по умолчанию) или "
            "'chunk_transfer' (batch ChunkReceipt'ов, ADR-0042, Stage 6)."
        ),
    )
    # Stage 6 (ADR-0042): batch ChunkReceipt'ов для type=chunk_transfer.
    # Поле присутствует только при type=chunk_transfer; при type=forward игнорируется.
    chunk_receipts: list[ChunkReceipt] | None = Field(
        default=None,
        description=(
            "Список ChunkReceipt v2 (ADR-0042). "
            "Обязателен при type='chunk_transfer'; None при type='forward'."
        ),
    )


class AccountingReportResponse(BaseModel):
    """Ответ POST /api/v1/accounting/report (ADR-0022, §3.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    idempotent: bool = Field(default=False)


def accounting_canonical_bytes(body: AccountingReportRequest) -> bytes:
    """Канонические байты для подписи accounting report (без client_signature).

    Используется клиентом для sign_accounting и трекером для verify_accounting.

    Backward-compat (ADR-0042, Stage 6 R6 review):
      - type="forward" (default, Stage 4/5): оригинальные 5 полей без type и chunk_receipts.
        Stage 4/5 клиенты могут пройти верификацию без изменений.
      - type="chunk_transfer": включаем type-hash и receipts-hash в canonical bytes,
        чтобы атакующий не мог подменить receipts при валидной подписи клиента.

    Для type="chunk_transfer" canonical bytes = SHA256 от:
      json(5 оригинальных полей) + sha256(type.encode()) + sha256(json(receipts) или b"null")

    Безопасность: receipt подмена атакующего при chunk_transfer теперь ломает подпись.
    """
    data = body.model_dump(mode="json")
    data.pop("client_signature", None)

    # Определить тип (default = "forward" для backward-compat)
    report_type: str = data.pop("type", "forward") or "forward"
    receipts_data = data.pop("chunk_receipts", None)

    # Canonical base: оригинальные 5 полей (идентично Stage 4/5 клиентам)
    base_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    if report_type == "forward":
        # Backward-compat: Stage 4/5 формат без изменений
        return base_bytes

    # type="chunk_transfer" — включаем type-hash и receipts-hash
    type_hash = hashlib.sha256(report_type.encode("utf-8")).digest()
    if receipts_data is None:
        receipts_hash = hashlib.sha256(b"null").digest()
    else:
        # Детерминированная сериализация receipts: sort_keys чтобы порядок полей не влиял
        receipts_json = json.dumps(receipts_data, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        receipts_hash = hashlib.sha256(receipts_json).digest()

    # Итоговые canonical bytes: base || type_hash || receipts_hash
    return base_bytes + type_hash + receipts_hash


def strike_evidence_canonical_bytes(evidence: StrikeEvidence) -> bytes:
    """Канонические байты evidence для подписи strike report.

    Весь StrikeReportRequest без reporter_signature, то есть
    canonical JSON {session_id, reporter_peer_id, offender_peer_id, evidence}.
    Вызывается с полным реквестом (без поля reporter_signature).
    """
    data = evidence.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def strike_report_canonical_bytes(body: StrikeReportRequest) -> bytes:
    """Канонические байты для подписи strike report (без reporter_signature)."""
    data = body.model_dump(mode="json")
    data.pop("reporter_signature", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Stage 5 wave 2: Chunk registry DTOs (ADR-0030, ADR-0033, ADR-0035)
# ---------------------------------------------------------------------------


class ChunkSpec(BaseModel):
    """Спецификация одного чанка весов (ADR-0030).

    Входит в ManifestResponse и в тело POST /api/v1/models.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(description="Идентификатор чанка (уникален в рамках модели)")
    sha256: str = Field(min_length=64, max_length=64, description="SHA256 hex-дайджест чанка")
    byte_size: int = Field(gt=0, description="Размер чанка в байтах")
    layer_range: tuple[int, int] = Field(description="Диапазон слоёв [start, end) покрытых чанком")
    ord: int = Field(ge=0, description="Порядковый номер чанка внутри модели (0-based)")


class ModelManifest(BaseModel):
    """Манифест модели — тело POST /api/v1/models (ADR-0030).

    manifest_sha256 вычисляется клиентом как sha256(canonical JSON chunks[]).
    Трекер верифицирует manifest_sha256 при публикации.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(description="Уникальный идентификатор модели, например 'llama-2-70b'")
    num_layers: int = Field(gt=0, description="Общее число слоёв трансформера")
    hidden_size: int = Field(gt=0, description="Размер скрытого состояния (hidden_dim)")
    dtype: str = Field(description="Тип весов: 'fp16', 'int8', 'bf16'")
    cuda_capability_min: str = Field(
        description="Минимальная вычислительная способность CUDA, например '8.0'"
    )
    chunks: list[ChunkSpec] = Field(description="Список чанков модели")
    manifest_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="SHA256 hex от canonical JSON списка ChunkSpec[]",
    )


class ModelManifestResponse(BaseModel):
    """Ответ GET /api/v1/models/{model_id}/manifest (ADR-0030)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    num_layers: int
    hidden_size: int
    dtype: str
    cuda_capability_min: str
    total_chunks: int
    manifest_sha256: str
    chunks: list[ChunkSpec]


class ChunkPeer(BaseModel):
    """Один держатель чанка (ADR-0033, ADR-0035).

    Входит в ответ GET /api/v1/models/{model_id}/chunks/{chunk_id}/peers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    peer_id: str = Field(description="peer_id ноды-держателя")
    addr: str = Field(description="multiaddr ноды для p2p-скачивания")
    is_seed: bool = Field(description="True если нода — pinned weight seeder (ADR-0033)")


class ChunkPeersResponse(BaseModel):
    """Ответ GET /api/v1/models/{model_id}/chunks/{chunk_id}/peers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    peers: list[ChunkPeer]


class NodeChunkDeclaration(BaseModel):
    """Тело POST /api/v1/nodes/me/chunks — нода декларирует свои чанки (ADR-0035).

    chunk_ids — список chunk_id (строк), которые нода заявляет как имеющиеся.
    Трекер делает upsert в node_chunks с verified_at=now.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_ids: list[str] = Field(description="Список chunk_id, которые нода держит")


class NodeChunkDeclarationResponse(BaseModel):
    """Ответ POST /api/v1/nodes/me/chunks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: int = Field(description="Количество принятых (upsert) chunk_id")
    unknown: list[str] = Field(
        default_factory=list,
        description="chunk_id, не найденные ни в одной модели",
    )


# ---------------------------------------------------------------------------
# Stage 5 wave 3: Chunk transport DTOs (ADR-0035)
# ---------------------------------------------------------------------------


class WeightChunkRequest(BaseModel):
    """Запрос чанка весов по QUIC weight-chunk stream (ADR-0035, ADR-0043)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(description="Идентификатор модели")
    chunk_id: str = Field(description="Идентификатор чанка")
    # Stage 6 (ADR-0043): peer_id для per-peer rate-limit на uploader'е.
    requester_peer_id: str | None = Field(
        default=None,
        description=(
            "peer_id запрашивающей ноды (ADR-0043). "
            "Используется uploader'ом для per-peer rate-limit. "
            "None — backward-compat с Stage 5 нодами."
        ),
    )


class WeightChunkResponse(BaseModel):
    """Метаданные чанка перед стримингом байтов (ADR-0035).

    Тело чанка идёт сразу после этого фрейма — последовательными
    length-prefixed фреймами по 64 КБ.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(description="Идентификатор модели")
    chunk_id: str = Field(description="Идентификатор чанка")
    sha256: str = Field(min_length=64, max_length=64, description="SHA256 hex-дайджест чанка")
    size_bytes: int = Field(gt=0, description="Полный размер чанка в байтах")


class WeightChunkError(BaseModel):
    """Ошибка при запросе чанка — чанк недоступен (ADR-0035, ADR-0043).

    Stage 6 (ADR-0043): добавлен code (machine-readable) и retry_after_ms.
    Поле reason сохранено для backward-compat.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    reason: str
    # Stage 6 (ADR-0043): machine-readable error code.
    # "chunk_not_found" | "rate_limited" | "server_error"
    code: str = Field(
        default="chunk_not_found",
        description=(
            "Machine-readable error code (ADR-0043): "
            "'chunk_not_found' | 'rate_limited' | 'server_error'."
        ),
    )
    retry_after_ms: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Подсказка downloader'у: через сколько мс повторить запрос (ADR-0043). "
            "None если retry не имеет смысла (chunk_not_found)."
        ),
    )


class ChunkReceipt(BaseModel):
    """Квитанция о получении чанка — отправляется downloader'ом uploader'у (ADR-0035, ADR-0042).

    v2 (Stage 6, ADR-0042): добавлено обязательное поле downloader_pubkey (32 байта Ed25519).
    Receipt является self-contained: uploader проверяет подпись без обращения к трекеру.

    Подпись охватывает: domain_tag + chunk_id + sha256 + downloader_peer_id
    + uploader_peer_id + bytes_received + ts (детерминированный порядок, payload не изменён).
    downloader_pubkey не входит в payload подписи — только контейнер для верификации.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(description="Идентификатор чанка")
    sha256: str = Field(min_length=64, max_length=64, description="SHA256 hex полученного чанка")
    downloader_peer_id: str = Field(description="peer_id получателя")
    uploader_peer_id: str = Field(description="peer_id отправителя")
    bytes_received: int = Field(gt=0, description="Размер полученных данных в байтах")
    ts: int = Field(description="Unix timestamp момента получения (секунды)")
    signature: str = Field(description="base58-encoded Ed25519 подпись downloader'а (ADR-0035)")
    # Stage 6 (ADR-0042): inline pubkey для верификации без peer key cache.
    # base58-encoded 32-byte Ed25519 raw public key downloader'а.
    downloader_pubkey: str = Field(
        description=(
            "base58-encoded 32-byte Ed25519 public key downloader'а (ADR-0042). "
            "Используется uploader'ом для верификации подписи без обращения к трекеру. "
            "peer_id_from_public_key(b58decode(downloader_pubkey)) == downloader_peer_id."
        )
    )


# Resolve forward reference: AccountingReportRequest.chunk_receipts → ChunkReceipt
# (ChunkReceipt определён после AccountingReportRequest в этом файле).
AccountingReportRequest.model_rebuild()


def serialize_session_init(init: SessionInit) -> bytes:
    """Сериализовать SessionInit для отправки по QUIC-стриму.

    Wire-формат: length(u32 BE) || json_bytes(UTF-8).

    Args:
        init: SessionInit DTO.

    Returns:
        Байтовый буфер для отправки.
    """
    json_bytes = canonical_json(init)
    length = len(json_bytes)
    return struct.pack(">I", length) + json_bytes


def parse_session_init(data: bytes) -> SessionInit:
    """Десериализовать SessionInit из QUIC-фрейма.

    Wire-формат: length(u32 BE) || json_bytes(UTF-8).

    Args:
        data: полный фрейм (length prefix включён).

    Returns:
        SessionInit DTO.

    Raises:
        ValueError: если буфер слишком короткий или JSON невалидный.
    """
    if len(data) < 4:
        raise ValueError(f"SessionInit frame too short: {len(data)} bytes")
    (length,) = struct.unpack_from(">I", data, 0)
    if len(data) < 4 + length:
        raise ValueError(
            f"SessionInit frame truncated: expected {4 + length} bytes, got {len(data)}"
        )
    json_bytes = data[4 : 4 + length]
    return SessionInit.model_validate_json(json_bytes)


# --- Stage 6: mid-session failover (ADR-0039) ---


class RouteReplaceRequest(BaseModel):
    """Тело POST /api/v1/route/replace (ADR-0039, §6.2).

    Клиент запрашивает замену упавшего хопа dead_peer_id в активной сессии.
    Авторизация — relay_token из forward_sessions в заголовке Authorization: Bearer <token>.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: uuid.UUID = Field(description="UUID forward-сессии")
    dead_peer_id: str = Field(description="peer_id упавшей ноды, которую надо заменить")
    layer_range: tuple[int, int] = Field(
        description="Диапазон слоёв [start, end), назначенный dead_peer_id"
    )


class RouteReplaceResponse(BaseModel):
    """Ответ POST /api/v1/route/replace (ADR-0039, §6.2).

    replacement — NodeRoute замены; hop_index — позиция в цепочке.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    replacement: NodeRoute = Field(description="NodeRoute ноды-замены")
    hop_index: int = Field(ge=0, description="Позиция замены в цепочке (0-based)")
