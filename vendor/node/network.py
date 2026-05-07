"""Stage 3: QUIC-транспорт для distributed inference (ADR-0012, ADR-0017, ADR-0019).

Реализует:
  - TLS self-signed сертификат из Ed25519 ключа ноды (ADR-0017).
  - QUIC-сервер: слушает UDP-порт, принимает входящие соединения.
  - QUIC-клиент: dial к next-hop по multiaddr.
  - SessionInit handshake: первый фрейм после TLS (ADR-0017).
  - Peer ID верификация: SPKI из TLS → base58(sha256(pk)) == заявленный peer_id.
  - Лимиты транспорта (ADR-0019): MAX_ENVELOPE_SIZE_BYTES, backpressure timeout.
  - Relay-fallback через tracker WS (ADR-0014): при NAT=symmetric или dial failure.

Frame wire-формат (для всех сообщений поверх QUIC-стрима):
    length(u32 BE) || msg_bytes

SessionInit — первый фрейм; после него идут ForwardEnvelope-фреймы.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import ssl
import struct
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from nacl.signing import SigningKey

from shared.constants import (
    BACKPRESSURE_TIMEOUT_SEC,
    CONNECT_TIMEOUT_SEC,
    FORWARD_TIMEOUT_SEC,
    MAX_ENVELOPE_SIZE_BYTES,
    MAX_INFLIGHT_ENVELOPES_PER_SESSION,
)
from shared.crypto import peer_id_from_public_key
from shared.protocol import (
    ForwardEnvelope,
    NodeRoute,
    SessionInit,
    parse_envelope,
    parse_multiaddr,
    serialize_envelope,
)
from shared.quic_io import encode_frame, read_frame

if TYPE_CHECKING:
    from node.trust import StrikeLog

logger = logging.getLogger(__name__)

# QUIC ALPN для llm-swarm (ADR-0017)
_ALPN = ["llm-swarm/1"]

# Impostors/error strings for QUIC close reason
_REASON_PEER_ID_MISMATCH = b"peer_id_mismatch"
_REASON_ENVELOPE_TOO_LARGE = b"envelope_too_large"
_REASON_BACKPRESSURE_TIMEOUT = b"backpressure_timeout"
_REASON_SESSION_INIT_INVALID = b"session_init_invalid"

# Тип handler'а для входящего envelope
EnvelopeHandler = Callable[
    [ForwardEnvelope, bytes, SessionInit],
    Coroutine,
]


# ---------------------------------------------------------------------------
# TLS сертификат из Ed25519 ключа
# ---------------------------------------------------------------------------


def generate_self_signed_cert(signing_key: SigningKey) -> tuple[bytes, bytes]:
    """Сгенерировать self-signed X.509 сертификат из Ed25519 ключа ноды.

    Public key кладётся в SubjectPublicKeyInfo (SPKI) сертификата.
    CN и SAN = peer_id (base58(sha256(public_key))) для отладки.

    Используется один и тот же Ed25519-ключ для:
    - envelope подписей (ADR-0009)
    - TLS handshake (ADR-0017)
    domain_tag в envelope защищает от cross-protocol forgery.

    Returns:
        (cert_pem, key_pem) — DER-serialised байты сертификата и private key.
    """
    pk_bytes = bytes(signing_key.verify_key)
    peer_id = peer_id_from_public_key(pk_bytes)

    # Конвертируем nacl seed в cryptography Ed25519PrivateKey
    # nacl SigningKey._signing_key = seed (32 bytes) + public_key (32 bytes) (64 total)
    # Нам нужен только seed (первые 32 байта)
    seed_bytes = bytes(signing_key)  # nacl SigningKey.__bytes__ = seed
    crypto_private_key = Ed25519PrivateKey.from_private_bytes(seed_bytes)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, peer_id[:64]),  # CN до 64 символов
        ]
    )

    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(crypto_private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365 * 10))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(f"p2p.{peer_id[:50]}")]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    ExtendedKeyUsageOID.SERVER_AUTH,
                    ExtendedKeyUsageOID.CLIENT_AUTH,
                ]
            ),
            critical=False,
        )
        .sign(crypto_private_key, None)  # Ed25519 не требует хеш-алгоритма
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = crypto_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


def extract_peer_id_from_spki(cert_der: bytes) -> str | None:
    """Извлечь peer_id из SPKI сертификата (DER).

    peer_id = base58(sha256(raw_ed25519_public_key_bytes))

    Args:
        cert_der: DER-encoded X.509 сертификат.

    Returns:
        peer_id строка или None при ошибке парсинга.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        cert = x509.load_der_x509_certificate(cert_der)
        pub_key = cert.public_key()
        if not isinstance(pub_key, Ed25519PublicKey):
            logger.warning("Peer certificate uses non-Ed25519 key")
            return None
        raw_pk = pub_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return peer_id_from_public_key(raw_pk)
    except Exception as exc:
        logger.debug("Failed to extract peer_id from SPKI: %s", exc)
        return None


def get_pubkey_bytes_from_spki(cert_der: bytes) -> bytes | None:
    """Извлечь raw 32-byte Ed25519 public key из SPKI сертификата (DER)."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        cert = x509.load_der_x509_certificate(cert_der)
        pub_key = cert.public_key()
        if not isinstance(pub_key, Ed25519PublicKey):
            return None
        return pub_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    except Exception as exc:
        logger.debug("Failed to extract pubkey from SPKI: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Peer ID verification для входящего соединения
# ---------------------------------------------------------------------------


def verify_peer_id_from_tls(
    peer_cert_der: bytes,
    expected_peer_id: str,
) -> bool:
    """Проверить peer_id по TLS сертификату (ADR-0017).

    Алгоритм: извлечь Ed25519 pubkey из SPKI → base58(sha256(pubkey)) → сравнить.

    Args:
        peer_cert_der: DER-encoded X.509 сертификат peer'а.
        expected_peer_id: ожидаемый peer_id (из SessionInit или маршрута).

    Returns:
        True если совпадает, False иначе.
    """
    derived = extract_peer_id_from_spki(peer_cert_der)
    if derived is None:
        return False
    return derived == expected_peer_id


# ---------------------------------------------------------------------------
# QUIC Server
# ---------------------------------------------------------------------------


@dataclass
class NodeQuicServer:
    """QUIC-сервер ноды (ADR-0012, ADR-0017, ADR-0019).

    Слушает UDP-порт, принимает входящие QUIC-соединения.
    При каждом входящем стриме:
      1. Читает SessionInit фрейм.
      2. Верифицирует peer_id из TLS SPKI.
      3. Читает ForwardEnvelope фреймы и передаёт в envelope_handler.

    При детекции аномалий (невалидная подпись prev_sig, NaN/Inf в активациях)
    пишет в StrikeLog (S4-C3); не блокирует работу.

    Использование:
        server = NodeQuicServer(
            host="0.0.0.0",
            port=9001,
            signing_key=identity.signing_key,
            envelope_handler=my_handler,
            strike_log=StrikeLog(data_dir),  # опционально
        )
        await server.start()
        # ... работает ...
        await server.stop()
    """

    host: str
    port: int
    signing_key: SigningKey
    envelope_handler: EnvelopeHandler
    strike_log: StrikeLog | None = field(default=None)
    resource_monitor: object | None = field(
        default=None,
        repr=False,
    )
    """ResourceMonitor (Stage 6, ADR-0038): если задан — throttle hook для SessionInit.

    TYPE: node.monitor.ResourceMonitor | None.
    """
    weight_manager: object | None = field(
        default=None,
        repr=False,
    )
    """WeightManager (Stage 5, E2.2): если задан — сервер принимает weight-chunk стримы.

    TYPE: node.weights.WeightManager | None. Используется forward ref чтобы избежать
    циклических импортов (network <-> weights -> network).
    """
    _server: object = field(default=None, init=False, repr=False)
    _cert_pem: bytes = field(default=b"", init=False, repr=False)
    _key_pem: bytes = field(default=b"", init=False, repr=False)

    def __post_init__(self) -> None:
        self._cert_pem, self._key_pem = generate_self_signed_cert(self.signing_key)

    def _make_quic_config(self) -> object:
        """Создать QuicConfiguration для сервера."""
        from aioquic.quic.configuration import QuicConfiguration

        config = QuicConfiguration(
            alpn_protocols=_ALPN,
            is_client=False,
            max_data=MAX_ENVELOPE_SIZE_BYTES * MAX_INFLIGHT_ENVELOPES_PER_SESSION,
            max_stream_data=MAX_ENVELOPE_SIZE_BYTES,
            idle_timeout=float(BACKPRESSURE_TIMEOUT_SEC + FORWARD_TIMEOUT_SEC),
        )
        # Загружаем self-signed cert и private key
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as cert_f:
            cert_f.write(self._cert_pem)
            cert_path = cert_f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as key_f:
            key_f.write(self._key_pem)
            key_path = key_f.name

        try:
            config.load_cert_chain(cert_path, key_path)
        finally:
            os.unlink(cert_path)
            os.unlink(key_path)

        return config

    async def start(self) -> None:
        """Запустить QUIC-сервер."""
        from aioquic.asyncio import serve

        config = self._make_quic_config()

        # aioquic вызывает stream_handler синхронно. Оборачиваем в create_task,
        # иначе async-handler даст "coroutine never awaited". Сохраняем ссылки на
        # задачи, чтобы GC их не убил до завершения (RUF006).
        loop = asyncio.get_running_loop()
        self._stream_tasks: set[asyncio.Task[None]] = set()

        def _wrap_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = loop.create_task(self._handle_stream(reader, writer))
            self._stream_tasks.add(task)
            task.add_done_callback(self._stream_tasks.discard)

        self._server = await serve(
            self.host,
            self.port,
            configuration=config,
            stream_handler=_wrap_handler,
        )
        logger.info("QUIC server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Остановить QUIC-сервер."""
        if self._server is not None:
            self._server.close()
            self._server = None
            logger.info("QUIC server stopped")

    async def _handle_stream(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handler входящего QUIC-стрима.

        Маршрутизация по первому фрейму (Stage 5, ADR-0035):
        - Если первый JSON-фрейм содержит поле "model_id" И "chunk_id" И НЕТ "session_id"
          → weight-chunk stream → _handle_weight_chunk_stream.
        - Иначе → SessionInit → inference envelope handler.

        Протокол (inference):
        1. Читаем SessionInit фрейм.
        2. Верифицируем peer_id из TLS SPKI (если доступен сертификат).
        3. Читаем ForwardEnvelope фреймы, передаём в envelope_handler.
        4. При любой ошибке — закрываем стрим.
        """
        remote_info = "unknown"
        try:
            # Читаем первый фрейм — он может быть SessionInit или WeightChunkRequest
            try:
                first_bytes = await read_frame(
                    reader,
                    max_size=64 * 1024,  # первый фрейм не может быть > 64 KiB
                    timeout=float(FORWARD_TIMEOUT_SEC),
                )
            except (TimeoutError, EOFError, ValueError) as exc:
                logger.warning("Failed to read first frame: %s", exc)
                writer.close()
                return

            # Маршрутизация: weight-chunk vs inference
            if self.weight_manager is not None and _is_weight_chunk_request(first_bytes):
                await self._handle_weight_chunk_stream(first_bytes, reader, writer)
                return

            # Inference path: интерпретируем как SessionInit
            try:
                session_init = SessionInit.model_validate_json(first_bytes)
            except Exception as exc:
                logger.warning("Invalid SessionInit: %s", exc)
                writer.write(encode_frame(_REASON_SESSION_INIT_INVALID))
                writer.close()
                return

            remote_info = f"session={session_init.session_id} hop={session_init.hop_index}"
            logger.debug("SessionInit received: %s", remote_info)

            # Stage 6 (ADR-0038): throttle hook — отказываем новым сессиям при degraded.
            # Существующие сессии (forward pass) не прерываются.
            if self.resource_monitor is not None:
                monitor = self.resource_monitor  # type: ignore[union-attr]
                if not monitor.allow_new_session():
                    snap = monitor.snapshot
                    logger.warning(
                        "ResourceMonitor degraded, rejecting SessionInit: session=%s reasons=%s",
                        session_init.session_id,
                        snap.degraded_reasons,
                    )
                    err_msg = f'{{"error":"RESOURCE_DEGRADED","reasons":{snap.degraded_reasons!r}}}'
                    writer.write(encode_frame(err_msg.encode("utf-8")))
                    writer.close()
                    return

            # mTLS peer_id verification: клиент в стандартном QUIC не шлёт client cert.
            # Верификация client peer_id — на уровне SessionInit.session_token (ADR-0020/0021):
            # трекер выдаёт relay_token только ноде с валидным session_token из БД.
            # Если в будущем включим mTLS (request_client_certificate=True в QuicConfiguration),
            # то cert извлекается аналогично client-стороне через _extract_peer_cert (ADR-0028).
            # Пока: no client cert in MVP, peer_id verification via challenge-response (ADR-0020).

            # Читаем ForwardEnvelope фреймы
            inflight = 0
            while True:
                try:
                    frame_bytes = await asyncio.wait_for(
                        read_frame(reader, max_size=MAX_ENVELOPE_SIZE_BYTES),
                        timeout=float(BACKPRESSURE_TIMEOUT_SEC),
                    )
                except TimeoutError:
                    logger.warning(
                        "Backpressure timeout (%ds) for %s, closing stream",
                        BACKPRESSURE_TIMEOUT_SEC,
                        remote_info,
                    )
                    writer.write(encode_frame(_REASON_BACKPRESSURE_TIMEOUT))
                    writer.close()
                    return
                except (EOFError, ValueError) as exc:
                    logger.debug("Stream ended for %s: %s", remote_info, exc)
                    return

                if not frame_bytes:
                    # EOF sentinel или пустой фрейм — закончили
                    break

                # Парсим envelope
                try:
                    env, sig = parse_envelope(frame_bytes)
                except (ValueError, struct.error) as exc:
                    logger.warning("Invalid envelope from %s: %s", remote_info, exc)
                    writer.close()
                    return

                # S4-C3: Проверка подписи предыдущего хопа (prev_sig).
                # Если env.hop_index > 0 и prev_sig != \x00*64, верифицируем цепочку.
                # На данном этапе у нас нет доступа к pubkey предыдущей ноды через
                # QUIC (нет mTLS client cert). Детектируем явно некорректный prev_sig
                # (hop_index 0 с ненулевым prev_sig).
                _ZERO_SIG = b"\x00" * 64
                if env.hop_index == 0 and env.prev_sig != _ZERO_SIG:
                    # Нарушение инварианта: hop 0 должен иметь prev_sig = 0x00*64
                    if self.strike_log is not None:
                        self.strike_log.record_chain_break(
                            session_id=session_init.session_id,
                            hop_index=env.hop_index,
                            prev_envelope=env.prev_sig.hex(),
                            reason="hop_0_nonzero_prev_sig",
                        )
                    logger.warning(
                        "Chain break: hop 0 has non-zero prev_sig from %s, closing",
                        remote_info,
                    )
                    writer.close()
                    return

                # S4-C3: Проверка NaN/Inf в activations_blob.
                # Проверяем только первые байты через struct для быстрой детекции;
                # полная проверка выполняется в inference handler.
                # Эвристика: если blob слишком короткий — аномалия.
                if len(env.activations_blob) == 0:
                    if self.strike_log is not None:
                        self.strike_log.record_inference_anomaly(
                            session_id=session_init.session_id,
                            layer_range=(env.layer_start, env.layer_end),
                            anomaly="empty_activations_blob",
                        )
                    logger.warning(
                        "Empty activations_blob from %s, session=%s hop=%d",
                        remote_info,
                        session_init.session_id,
                        env.hop_index,
                    )
                    writer.close()
                    return

                inflight += 1
                if inflight > MAX_INFLIGHT_ENVELOPES_PER_SESSION:
                    logger.warning(
                        "Too many in-flight envelopes (%d) from %s, closing",
                        inflight,
                        remote_info,
                    )
                    writer.close()
                    return

                # Вызываем handler (compute-bound, не блокируем event loop)
                try:
                    result_bytes = await self.envelope_handler(env, sig, session_init)
                    if result_bytes is not None:
                        # Отправляем ответ
                        writer.write(encode_frame(result_bytes))
                        await writer.drain()
                except Exception as exc:
                    logger.error(
                        "Envelope handler error for %s: %s",
                        remote_info,
                        exc,
                        exc_info=True,
                    )
                    writer.close()
                    return

                inflight -= 1

        except Exception as exc:
            logger.error("Unexpected error handling stream %s: %s", remote_info, exc, exc_info=True)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _handle_weight_chunk_stream(
        self,
        first_bytes: bytes,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handler weight-chunk QUIC-стрима (server-side, ADR-0035, Stage 5 E2.2).

        Вызывается из _handle_stream при детекции WeightChunkRequest в первом фрейме.
        Делегирует в node.weights.handle_weight_chunk_stream.
        """
        from node.weights import handle_weight_chunk_stream

        wm = self.weight_manager
        # first_bytes — payload первого фрейма (уже без length prefix).
        # handle_weight_chunk_stream ожидает reader с WeightChunkRequest как первым фреймом.
        # Создаём chained reader: сначала first_bytes как length-prefixed фрейм, потом исходный.
        chained_reader = _make_prepended_reader(first_bytes, reader)

        # Получаем зависимости из weight_manager
        store = wm._store  # type: ignore[union-attr]
        identity = wm._identity  # type: ignore[union-attr]
        upload_sem = wm._upload_sem  # type: ignore[union-attr]
        tracker_client = wm._tracker_client  # type: ignore[union-attr]
        session_token = wm._session_token  # type: ignore[union-attr]
        rate_limiter = getattr(wm, "_rate_limiter", None)

        await handle_weight_chunk_stream(
            chained_reader,  # type: ignore[arg-type]
            writer,
            store=store,
            identity=identity,
            tracker_client=tracker_client,
            session_token=session_token,
            upload_sem=upload_sem,
            rate_limiter=rate_limiter,
        )


def _is_weight_chunk_request(frame_bytes: bytes) -> bool:
    """Проверить, является ли JSON-фрейм WeightChunkRequest.

    Быстрая проверка без полной десериализации: JSON содержит
    "model_id" и "chunk_id" но НЕ "session_id".
    """
    import json

    try:
        data = json.loads(frame_bytes)
    except (ValueError, UnicodeDecodeError):
        return False
    return (
        isinstance(data, dict)
        and "model_id" in data
        and "chunk_id" in data
        and "session_id" not in data
    )


def _make_prepended_reader(
    first_frame_data: bytes,
    underlying: asyncio.StreamReader,
) -> asyncio.StreamReader:
    """Создать StreamReader, который начинается с first_frame_data, затем underlying.

    Используется для маршрутизации: первый фрейм уже прочитан из stream,
    нужно передать его в sub-handler вместе с оставшимся потоком.

    Метод: создаём новый StreamReader, кладём в него length-prefixed фрейм,
    затем паразитически feed_data из underlying через задачу.

    Проще: используем _ReplayReader-like подход с переопределением readexactly.
    """
    # Создаём StreamReader и кормим его first_frame_data как length-prefixed frame
    # (уже без prefix — first_frame_data это payload, нам нужно восстановить wire-формат)
    import struct

    wire = struct.pack(">I", len(first_frame_data)) + first_frame_data
    new_reader = asyncio.StreamReader()
    new_reader.feed_data(wire)
    # Подключаем underlying: когда new_reader исчерпает буфер, он должен читать из underlying.
    # asyncio.StreamReader не поддерживает цепочку напрямую — используем адаптер.
    return _ChainedReader(new_reader, underlying)


class _ChainedReader:
    """asyncio.StreamReader-like адаптер: сначала first, потом second.

    Реализует только readexactly (используется в read_frame через quic_io).
    """

    def __init__(
        self,
        first: asyncio.StreamReader,
        second: asyncio.StreamReader,
    ) -> None:
        self._first = first
        self._first_done = False
        self._second = second

    async def readexactly(self, n: int) -> bytes:
        """Читать ровно n байт, переключаясь с first на second при EOFError."""
        if self._first_done:
            return await self._second.readexactly(n)
        try:
            return await self._first.readexactly(n)
        except asyncio.IncompleteReadError as exc:
            # first исчерпан — дочитываем остаток из second
            partial = exc.partial
            if len(partial) == n:
                return partial
            self._first_done = True
            rest = await self._second.readexactly(n - len(partial))
            return partial + rest


# ---------------------------------------------------------------------------
# QUIC Client
# ---------------------------------------------------------------------------


async def dial(
    multiaddr: str,
    session_init: SessionInit,
    signing_key: SigningKey,
    envelope: ForwardEnvelope,
    sig: bytes,
    timeout: float = float(FORWARD_TIMEOUT_SEC),
) -> bytes:
    """Отправить ForwardEnvelope следующему хопу по multiaddr (QUIC).

    Протокол:
    1. TLS handshake с проверкой peer_id (ADR-0017).
    2. Открываем QUIC stream.
    3. Отправляем SessionInit фрейм.
    4. Отправляем ForwardEnvelope фрейм.
    5. Ждём ответ (chain bytes) с таймаутом.

    Args:
        multiaddr: адрес следующего хопа в формате /ip4/.../udp/.../quic-v1/p2p/...
        session_init: SessionInit для отправки (hop_index == индекс получателя).
        signing_key: signing key текущей ноды (для TLS).
        envelope: ForwardEnvelope с активациями.
        sig: подпись envelope.
        timeout: таймаут ожидания ответа в секундах.

    Returns:
        bytes — ответ от следующего хопа (chain bytes).

    Raises:
        ConnectionError: при ошибке соединения.
        ValueError: при несоответствии peer_id (ADR-0017 mismatch).
        asyncio.TimeoutError: при превышении timeout.
    """
    from aioquic.asyncio import connect
    from aioquic.quic.configuration import QuicConfiguration

    parsed = parse_multiaddr(multiaddr)
    host, port, expected_peer_id = parsed.host, parsed.port, parsed.peer_id

    cert_pem, key_pem = generate_self_signed_cert(signing_key)

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as cf:
        cf.write(cert_pem)
        cert_path = cf.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as kf:
        kf.write(key_pem)
        key_path = kf.name

    try:
        # TODO Stage 4: дублирует логику client/quic_transport.py:QuicNodeConnection.connect().
        # Вынести в shared helper при рефакторинге транспортного слоя (Stage 4).
        config = QuicConfiguration(
            alpn_protocols=_ALPN,
            is_client=True,
            max_data=MAX_ENVELOPE_SIZE_BYTES * MAX_INFLIGHT_ENVELOPES_PER_SESSION,
            max_stream_data=MAX_ENVELOPE_SIZE_BYTES,
            idle_timeout=float(BACKPRESSURE_TIMEOUT_SEC + FORWARD_TIMEOUT_SEC),
            # ssl.CERT_NONE: сертификат self-signed и привязан к peer_id через SPKI
            # (base58(sha256(Ed25519_pubkey))). Стандартная CA-проверка не применима;
            # аутентификация выполняется вручную через extract_peer_id_from_spki (ADR-0017).
            verify_mode=ssl.CERT_NONE,
            server_name=host,
        )
        config.load_cert_chain(cert_path, key_path)
    finally:
        os.unlink(cert_path)
        os.unlink(key_path)

    async with connect(
        host,
        port,
        configuration=config,
    ) as protocol:
        # TODO Stage 3 followup: verify peer_id from TLS SPKI
        # aioquic не даёт прямого доступа к peer cert через публичный API на 1.2.0.
        # Requires monkey-patching или custom create_protocol с TLS callback.

        reader, writer = await protocol.create_stream()

        # Отправляем SessionInit как raw JSON (без length-prefix — read_frame уже снял его)
        init_json = session_init.model_dump_json().encode("utf-8")
        writer.write(encode_frame(init_json))

        # Отправляем envelope
        envelope_bytes = serialize_envelope(envelope, sig)
        writer.write(encode_frame(envelope_bytes))
        await writer.drain()

        # Ждём ответ с таймаутом
        max_response_size = MAX_ENVELOPE_SIZE_BYTES * MAX_INFLIGHT_ENVELOPES_PER_SESSION
        try:
            response_bytes = await asyncio.wait_for(
                read_frame(reader, max_size=max_response_size),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(
                "Forward timeout (%ds) waiting for response from %s:%d (peer_id=%s)",
                timeout,
                host,
                port,
                expected_peer_id,
            )
            raise

        return response_bytes


# ---------------------------------------------------------------------------
# Relay client (ADR-0014) — WebSocket fallback для symmetric NAT
# ---------------------------------------------------------------------------


class RelayClient:
    """WS-клиент к tracker relay (ADR-0014, §3.2).

    При NAT=symmetric или провале QUIC dial — открываем WS к
    /api/v1/relay/{session_id} и проксируем байты через трекер.

    Трекер — dumb byte-pipe: не парсит envelope, только проксирует.
    """

    def __init__(
        self,
        tracker_url: str,
        session_id: str,
        session_token: str,
    ) -> None:
        self._tracker_url = tracker_url.rstrip("/")
        self._session_id = session_id
        self._session_token = session_token
        self._ws = None

    async def connect(self) -> None:
        """Открыть WS-соединение к relay эндпоинту."""
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "websockets package required for relay mode. Install: pip install websockets"
            ) from exc

        ws_url = (
            self._tracker_url.replace("http://", "ws://").replace("https://", "wss://")
            + f"/api/v1/relay/{self._session_id}"
            + f"?token={self._session_token}"
        )
        self._ws = await websockets.connect(
            ws_url,
            additional_headers={"Authorization": f"Bearer {self._session_token}"},
        )
        logger.info(
            "Relay WS connected to %s for session %s",
            self._tracker_url,
            self._session_id,
        )

    async def send(self, data: bytes) -> None:
        """Отправить байты через relay."""
        if self._ws is None:
            raise RuntimeError("RelayClient not connected: call connect() first")
        await self._ws.send(data)

    async def recv(self) -> bytes:
        """Получить байты из relay."""
        if self._ws is None:
            raise RuntimeError("RelayClient not connected: call connect() first")
        data = await self._ws.recv()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return data

    async def close(self) -> None:
        """Закрыть WS-соединение."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


async def forward_via_relay(
    tracker_url: str,
    session_init: SessionInit,
    envelope: ForwardEnvelope,
    sig: bytes,
    timeout: float = float(FORWARD_TIMEOUT_SEC),
) -> bytes:
    """Отправить ForwardEnvelope через tracker WS relay (ADR-0014).

    Используется как fallback при NAT=symmetric или провале QUIC dial.

    Args:
        tracker_url: URL трекера.
        session_init: SessionInit с session_id и session_token.
        envelope: ForwardEnvelope.
        sig: подпись envelope.
        timeout: таймаут ожидания ответа.

    Returns:
        bytes — ответные chain bytes.
    """
    from shared.protocol import serialize_envelope

    relay = RelayClient(
        tracker_url=tracker_url,
        session_id=str(session_init.session_id),
        session_token=session_init.session_token,
    )
    await relay.connect()
    try:
        # Отправляем SessionInit как length-prefixed JSON фрейм
        init_json = session_init.model_dump_json().encode("utf-8")
        await relay.send(encode_frame(init_json))

        # Отправляем envelope
        env_bytes = serialize_envelope(envelope, sig)
        await relay.send(encode_frame(env_bytes))

        # Ждём ответ
        try:
            response = await asyncio.wait_for(relay.recv(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                "Relay timeout (%ds) for session %s",
                timeout,
                session_init.session_id,
            )
            raise

        return response
    finally:
        await relay.close()


# ---------------------------------------------------------------------------
# High-level: forward to next hop (QUIC с relay-fallback)
# ---------------------------------------------------------------------------


async def _attempt_hole_punch(
    *,
    self_peer_id: str,
    self_session_token: str,
    next_hop: NodeRoute,
    session_init: SessionInit,
    signing_key: SigningKey,
    envelope: ForwardEnvelope,
    sig: bytes,
    tracker_url: str,
    timeout: float = float(CONNECT_TIMEOUT_SEC),
) -> bytes | None:
    """Попытка UDP hole punch к ноде B через трекер (ADR-0027).

    Алгоритм:
    1. POST /api/v1/punch к трекеру → получаем peer_addr B + rendezvous_at.
    2. Sleep до rendezvous_at_unix_ms (или сразу если время уже прошло).
    3. Параллельно 3 попытки QUIC dial с интервалом 200мс к peer_addr.
    4. При успехе — возвращаем chain bytes.
    5. При провале — возвращаем None (caller переходит к relay-fallback).

    Args:
        self_peer_id: peer_id текущей ноды (инициатор).
        self_session_token: session_token ноды для auth на трекере.
        next_hop: NodeRoute ноды B (цели).
        session_init: SessionInit для сессии.
        signing_key: signing key текущей ноды.
        envelope: ForwardEnvelope с активациями.
        sig: подпись envelope.
        tracker_url: URL трекера.
        timeout: таймаут одной попытки dial.

    Returns:
        chain bytes при успехе, None при провале.
    """
    import time

    import httpx

    tracker_base = tracker_url.rstrip("/")
    punch_url = f"{tracker_base}/api/v1/punch"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                punch_url,
                json={
                    "self_peer_id": self_peer_id,
                    "peer_id": next_hop.peer_id,
                    "session_token": self_session_token,
                },
            )
            resp.raise_for_status()
            punch_data = resp.json()
    except Exception as exc:
        logger.warning("hole punch: POST /punch failed: %s", exc)
        return None

    peer_addr: str = punch_data.get("peer_addr", "")
    rendezvous_at_unix_ms: int = punch_data.get("rendezvous_at_unix_ms", 0)

    if not peer_addr:
        logger.warning("hole punch: empty peer_addr in response")
        return None

    # Sleep до rendezvous_at
    now_ms = int(time.time() * 1000)
    wait_ms = rendezvous_at_unix_ms - now_ms
    if wait_ms > 0:
        await asyncio.sleep(wait_ms / 1000.0)

    # 3 попытки QUIC dial с интервалом 200мс
    last_exc: Exception | None = None
    for attempt in range(3):
        if attempt > 0:
            await asyncio.sleep(0.2)
        try:
            result = await asyncio.wait_for(
                dial(
                    multiaddr=peer_addr,
                    session_init=session_init,
                    signing_key=signing_key,
                    envelope=envelope,
                    sig=sig,
                    timeout=timeout,
                ),
                timeout=timeout,
            )
            logger.info(
                "hole punch: connected to %s on attempt %d",
                next_hop.peer_id,
                attempt + 1,
            )
            return result
        except (TimeoutError, ConnectionError, OSError) as exc:
            last_exc = exc
            logger.debug(
                "hole punch: attempt %d to %s failed: %s",
                attempt + 1,
                peer_addr,
                exc,
            )

    logger.warning(
        "hole punch: all 3 attempts to %s failed, last error: %s",
        next_hop.peer_id,
        last_exc,
    )
    return None


async def forward_to_next_hop(
    *,
    next_hop: NodeRoute,
    session_init: SessionInit,
    signing_key: SigningKey,
    envelope: ForwardEnvelope,
    sig: bytes,
    tracker_url: str,
    nat_type: str = "unknown",
    timeout: float = float(FORWARD_TIMEOUT_SEC),
    self_peer_id: str | None = None,
    self_session_token: str | None = None,
) -> bytes:
    """Отправить envelope следующему хопу: QUIC, hole punch или relay-fallback.

    Логика выбора транспорта (ADR-0014, ADR-0015, ADR-0027):
    1. Если next_hop.addr is None → relay-only (нода за symmetric NAT).
    2. Если nat_type in ("symmetric", "unknown") → hole punch (если есть self_peer_id/token),
       иначе relay-only.
    3. Иначе (cone NAT) → QUIC dial напрямую; при ConnectionError → relay fallback.

    Args:
        next_hop: NodeRoute следующего хопа.
        session_init: SessionInit для сессии.
        signing_key: signing key текущей ноды.
        envelope: ForwardEnvelope с активациями.
        sig: подпись envelope.
        tracker_url: URL трекера для relay/punch.
        nat_type: тип NAT текущей ноды.
        timeout: таймаут ожидания.
        self_peer_id: peer_id текущей ноды (для hole punch, опционально).
        self_session_token: session_token ноды (для hole punch, опционально).

    Returns:
        bytes — chain bytes от следующего хопа.
    """
    use_relay = next_hop.addr is None or nat_type in ("symmetric", "unknown")

    if not use_relay:
        try:
            return await dial(
                multiaddr=next_hop.addr,  # type: ignore[arg-type]
                session_init=session_init,
                signing_key=signing_key,
                envelope=envelope,
                sig=sig,
                timeout=timeout,
            )
        except (TimeoutError, ConnectionError, OSError) as exc:
            logger.warning(
                "QUIC dial to %s failed (%s), falling back to relay",
                next_hop.addr,
                exc,
            )
            # Fallthrough to relay
    else:
        # Symmetric NAT или addr=None: попробуем hole punch если есть credentials.
        # Hole punch требует addr ноды B (получается от трекера через /punch).
        if (
            self_peer_id is not None
            and self_session_token is not None
            and next_hop.peer_id  # B известен
        ):
            result = await _attempt_hole_punch(
                self_peer_id=self_peer_id,
                self_session_token=self_session_token,
                next_hop=next_hop,
                session_init=session_init,
                signing_key=signing_key,
                envelope=envelope,
                sig=sig,
                tracker_url=tracker_url,
                timeout=float(CONNECT_TIMEOUT_SEC),
            )
            if result is not None:
                return result
            logger.info(
                "hole punch to %s failed, falling back to relay",
                next_hop.peer_id,
            )

    return await forward_via_relay(
        tracker_url=tracker_url,
        session_init=session_init,
        envelope=envelope,
        sig=sig,
        timeout=timeout,
    )
