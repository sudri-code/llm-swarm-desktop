"""Weight manager.

Stage 5 (E2): загрузка чанков весов модели, p2p-раздача через QUIC,
SHA256-верификация, GC. Полная реализация волны 3.

Реализует:
  - E2.1: ensure_chunks — загрузка с верификацией SHA256, fallback по peers.
  - E2.2: QUIC weight-chunk transport (клиентская сторона + server handler в network.py).
  - E2.3: Параллелизм через asyncio.Semaphore (fetch_concurrency / upload_concurrency).
  - E2.4: declare_to_tracker + _pending_declarations для heartbeat-интеграции.
  - E2.5: gc — вытеснение чанков вне keep_layer_range.
  - E2.6: list_local_chunks — сканирование cache_dir без SHA256.

ADR-0033: seed-fallback для получения чанков.
ADR-0035: ChunkReceipt с Ed25519-подписью downloader'а, domain tag «llm-swarm/v1/chunk-receipt».

Публичный контракт:
    WeightManager  — оркестратор: ensure_chunks, declare_to_tracker, gc,
                     list_local_chunks.
    ChunkStore     — файловый кэш: path_for, verify, gc.
    ChunkFetchError — все peers исчерпаны при загрузке чанка.
    ChunkSpec, ModelManifest, ChunkPeer, NodeChunkDeclaration — re-exported
                     из shared.protocol (источник истины).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import base58

from shared.crypto import sign_chunk_receipt
from shared.protocol import (
    ChunkPeer,
    ChunkReceipt,
    ChunkSpec,
    ModelManifest,
    NodeChunkDeclaration,
    WeightChunkError,
    WeightChunkRequest,
    WeightChunkResponse,
)
from shared.quic_io import (
    MAX_WEIGHT_CHUNK_SIZE_BYTES,
    WEIGHT_CHUNK_STREAM_CHUNK_SIZE,
    WEIGHT_RECEIPT_TIMEOUT_SEC,
    encode_frame,
    read_frame,
)

if TYPE_CHECKING:
    from node.tracker_client import TrackerClient
    from node.trust import Identity

__all__ = [
    "ChunkFetchError",
    "ChunkPeer",
    "ChunkSpec",
    "ChunkStore",
    "ModelManifest",
    "NodeChunkDeclaration",
    "WeightManager",
]

logger = logging.getLogger(__name__)

# Таймаут одного QUIC-соединения для скачивания чанка (сек).
_CHUNK_DIAL_TIMEOUT_SEC: float = 60.0

# TTL bucket'ов peer'ов (ADR-0043): 10 мин неактивности
_PEER_BUCKET_TTL_SEC: float = 600.0
# GC bucket'ов: раз в минуту
_BUCKET_GC_INTERVAL_SEC: float = 60.0


# ---------------------------------------------------------------------------
# Token bucket rate-limit (ADR-0043)
# ---------------------------------------------------------------------------


@dataclass
class PeerBucket:
    """Token bucket для rate-limit одного peer_id (ADR-0043).

    tokens: текущий запас байт (refill со скоростью bytes_per_sec).
    in_flight: число одновременно активных стримов от этого peer'а.
    last_refill_ts: момент последнего refill (time.monotonic).
    last_activity_ts: момент последней активности (для TTL GC).
    rejected: счётчик отклонённых запросов.
    """

    tokens: float
    in_flight: int = 0
    last_refill_ts: float = field(default_factory=time.monotonic)
    last_activity_ts: float = field(default_factory=time.monotonic)
    rejected: int = 0

    def refill(self, bytes_per_sec: float, burst_bytes: int) -> None:
        """Доначислить токены с момента последнего refill."""
        now = time.monotonic()
        delta = now - self.last_refill_ts
        self.tokens = min(
            float(burst_bytes),
            self.tokens + delta * bytes_per_sec,
        )
        self.last_refill_ts = now
        self.last_activity_ts = now


class UploaderRateLimiter:
    """Per-peer rate limiter для weight-chunk uploads (ADR-0043).

    Хранит dict[peer_id, PeerBucket]; GC раз в минуту удаляет протухшие.

    Не thread-safe — используется только из asyncio event loop.
    """

    def __init__(
        self,
        max_concurrent_per_peer: int = 1,
        bytes_per_sec_per_peer: float = 17_000_000.0,
        burst_bytes: int = 268_435_456,
    ) -> None:
        self._max_concurrent = max_concurrent_per_peer
        self._bytes_per_sec = bytes_per_sec_per_peer
        self._burst_bytes = burst_bytes
        self._buckets: dict[str, PeerBucket] = {}
        self._gc_task: asyncio.Task | None = None

    def _get_bucket(self, peer_id: str) -> PeerBucket:
        """Получить или создать bucket для peer_id."""
        if peer_id not in self._buckets:
            self._buckets[peer_id] = PeerBucket(tokens=float(self._burst_bytes))
        return self._buckets[peer_id]

    def check_and_reserve(self, peer_id: str, chunk_size_bytes: int) -> tuple[bool, int]:
        """Проверить допустимость нового стрима от peer_id.

        Оптимистично резервирует токены при успехе.

        Returns:
            (allowed, retry_after_ms): True если разрешено, False + подсказка если нет.
        """
        bucket = self._get_bucket(peer_id)
        bucket.refill(self._bytes_per_sec, self._burst_bytes)

        # Проверяем in_flight
        if bucket.in_flight >= self._max_concurrent:
            bucket.rejected += 1
            # retry_after_ms: примерно когда освободится слот (неопределённо → 1 сек)
            retry_ms = 1000
            logger.info(
                "rate-limit: peer %s rejected (concurrency: in_flight=%d >= max=%d), "
                "retry_after_ms=%d",
                peer_id,
                bucket.in_flight,
                self._max_concurrent,
                retry_ms,
            )
            return False, retry_ms

        # Проверяем bandwidth quota
        if bucket.tokens < chunk_size_bytes:
            bucket.rejected += 1
            deficit = chunk_size_bytes - bucket.tokens
            retry_ms = max(1, int(deficit / self._bytes_per_sec * 1000))
            logger.info(
                "rate-limit: peer %s rejected (bandwidth: tokens=%.0f < chunk=%d), "
                "retry_after_ms=%d",
                peer_id,
                bucket.tokens,
                chunk_size_bytes,
                retry_ms,
            )
            return False, retry_ms

        # Оптимистично списываем токены и увеличиваем in_flight
        bucket.tokens -= chunk_size_bytes
        bucket.in_flight += 1
        return True, 0

    def release(self, peer_id: str) -> None:
        """Освободить слот in_flight после завершения стрима."""
        if peer_id in self._buckets:
            self._buckets[peer_id].in_flight = max(0, self._buckets[peer_id].in_flight - 1)

    def _gc(self) -> int:
        """Удалить протухшие bucket'ы. Возвращает число удалённых."""
        now = time.monotonic()
        expired = [
            pid
            for pid, b in self._buckets.items()
            if now - b.last_activity_ts > _PEER_BUCKET_TTL_SEC and b.in_flight == 0
        ]
        for pid in expired:
            del self._buckets[pid]
        return len(expired)

    async def start_gc(self) -> None:
        """Запустить фоновый GC task."""
        self._gc_task = asyncio.create_task(self._gc_loop(), name="uploader_rate_gc")

    async def stop_gc(self) -> None:
        """Остановить GC task."""
        if self._gc_task is not None and not self._gc_task.done():
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass

    async def _gc_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_BUCKET_GC_INTERVAL_SEC)
            except asyncio.CancelledError:
                return
            n = self._gc()
            if n > 0:
                logger.debug("rate-limiter GC: removed %d peer buckets", n)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ChunkFetchError(RuntimeError):
    """Все peers исчерпаны при загрузке чанка."""

    def __init__(self, model_id: str, chunk_id: str, reason: str) -> None:
        self.model_id = model_id
        self.chunk_id = chunk_id
        super().__init__(f"Failed to fetch chunk {chunk_id!r} for model {model_id!r}: {reason}")


# ---------------------------------------------------------------------------
# ChunkStore — файловый кэш чанков
# ---------------------------------------------------------------------------


class ChunkStore:
    """Локальный файловый кэш чанков весов.

    Структура каталога::

        cache_dir/
            <model_id>/
                <chunk_id>.bin

    Верификация SHA256 делается только при явном вызове verify() или
    внутри ensure_chunks — не при list_local_chunks (slow path).
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def path_for(self, model_id: str, chunk_id: str) -> Path:
        """Вернуть путь к файлу чанка (файл может не существовать)."""
        return self._cache_dir / model_id / f"{chunk_id}.bin"

    def verify(self, path: Path, expected_sha256: str) -> bool:
        """Проверить SHA256 файла чанка.

        Возвращает True, если файл существует и его SHA256 совпадает
        с expected_sha256 (lowercase hex, 64 символа).
        При любой ошибке IO возвращает False.
        """
        try:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(block)
            return h.hexdigest() == expected_sha256.lower()
        except OSError:
            return False

    def gc(self, keep: set[tuple[str, str]]) -> int:
        """Удалить чанки, не входящие в множество keep={(model_id, chunk_id)}.

        Возвращает суммарное число освобождённых байт.

        Сканирует все <model_id>/<chunk_id>.bin в cache_dir.
        """
        freed = 0
        if not self._cache_dir.exists():
            return 0
        for model_dir in self._cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_id = model_dir.name
            for chunk_file in model_dir.glob("*.bin"):
                chunk_id = chunk_file.stem
                if (model_id, chunk_id) not in keep:
                    try:
                        size = chunk_file.stat().st_size
                        chunk_file.unlink()
                        freed += size
                        logger.debug("GC: removed chunk %s/%s (%d bytes)", model_id, chunk_id, size)
                    except OSError as exc:
                        logger.warning("GC: failed to remove %s: %s", chunk_file, exc)
        return freed


# ---------------------------------------------------------------------------
# WeightManager — оркестратор
# ---------------------------------------------------------------------------


class WeightManager:
    """Менеджер весов: загрузка, верификация, p2p-раздача, GC.

    Полная реализация Stage 5 волна 3 (E2):
        - ADR-0033: seed-based manifest download от трекера.
        - ADR-0035: chunk receipts (учёт раздачи для ratio).
        - spec §5: chunk size 256 МБ, SHA256 per-chunk.

    Параметры параллелизма берутся из конфига (WeightsConfig):
        fetch_concurrency — загрузка (default 4).
        upload_concurrency — раздача (default 2).
    """

    def __init__(
        self,
        cache_dir: Path,
        tracker_client: TrackerClient,
        identity: Identity,
        *,
        fetch_concurrency: int = 4,
        upload_concurrency: int = 2,
        manifest_ttl_seconds: int = 300,
        session_token: str | None = None,
        max_concurrent_per_peer: int = 1,
        bytes_per_sec_per_peer: float = 17_000_000.0,
        burst_bytes: int = 268_435_456,
    ) -> None:
        self._cache_dir = cache_dir
        self._tracker_client = tracker_client
        self._identity = identity
        self._store = ChunkStore(cache_dir)
        self._fetch_sem = asyncio.Semaphore(fetch_concurrency)
        self._upload_sem = asyncio.Semaphore(upload_concurrency)
        self._manifest_ttl = manifest_ttl_seconds
        self._session_token = session_token

        # Rate limiter (ADR-0043)
        self._rate_limiter = UploaderRateLimiter(
            max_concurrent_per_peer=max_concurrent_per_peer,
            bytes_per_sec_per_peer=bytes_per_sec_per_peer,
            burst_bytes=burst_bytes,
        )

        # Кэш манифестов: model_id -> (manifest, fetched_at)
        self._manifest_cache: dict[str, tuple[ModelManifest, float]] = {}

        # Множество chunk_id, ожидающих декларации на трекер.
        self._pending_declarations: set[str] = set()

    def set_session_token(self, token: str) -> None:
        """Установить/обновить session_token для аутентификации на трекере."""
        self._session_token = token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ensure_chunks(
        self,
        model_id: str,
        layer_range: tuple[int, int],
    ) -> list[Path]:
        """Гарантировать наличие всех чанков для диапазона слоёв.

        1. Запрашивает манифест от трекера (кэш 5 мин).
        2. Отбирает чанки, пересекающиеся с layer_range.
        3. Для каждого чанка: проверяет SHA256; при отсутствии/несовпадении —
           скачивает от peer-ноды через QUIC (ADR-0033).
        4. После всех fetch — декларирует на трекере.
        5. Возвращает список локальных путей в порядке chunks_needed.

        Raises:
            ChunkFetchError: если все peers исчерпаны для какого-либо чанка.
        """
        manifest = await self._get_manifest(model_id)
        layer_start, layer_end = layer_range
        chunks_needed = [
            c for c in manifest.chunks if _ranges_overlap(c.layer_range, (layer_start, layer_end))
        ]
        # Сортируем по ord для детерминированного порядка
        chunks_needed.sort(key=lambda c: c.ord)

        # Параллельная загрузка через Semaphore
        tasks = [self._ensure_one_chunk(model_id, chunk) for chunk in chunks_needed]
        results = await asyncio.gather(*tasks)

        # Декларируем только те, которые успешно скачали
        if self._pending_declarations:
            try:
                await self.declare_to_tracker(list(self._pending_declarations))
                self._pending_declarations.clear()
            except Exception as exc:
                logger.warning("declare_to_tracker failed: %s", exc)

        return list(results)

    async def declare_to_tracker(self, chunk_ids: list[str]) -> None:
        """Задекларировать трекеру список локально доступных чанков.

        POST /api/v1/nodes/me/chunks (ADR-0035).
        """
        if not chunk_ids:
            return
        if self._session_token is None:
            logger.warning(
                "declare_to_tracker: no session_token, skipping declaration of %d chunks",
                len(chunk_ids),
            )
            return
        try:
            resp = await self._tracker_client.declare_chunks(
                chunk_ids=chunk_ids,
                session_token=self._session_token,
            )
            logger.info(
                "Declared %d chunks to tracker (accepted=%d, unknown=%d)",
                len(chunk_ids),
                resp.accepted,
                len(resp.unknown),
            )
        except Exception as exc:
            logger.warning("declare_to_tracker: tracker call failed: %s", exc)
            raise

    async def gc(self, keep_layer_range: tuple[int, int]) -> int:
        """Вытеснить чанки вне keep_layer_range, вернуть освобождённые байты.

        Просматривает все модели в cache_dir. Для каждой ищет манифест в
        _manifest_cache и вычисляет keep-set. Если манифест не закэширован —
        пропускает модель (нельзя гарантированно определить диапазоны).
        """
        keep: set[tuple[str, str]] = set()
        layer_start, layer_end = keep_layer_range

        # Собираем keep-set из всех известных манифестов
        for model_id, (manifest, _) in self._manifest_cache.items():
            for chunk in manifest.chunks:
                if _ranges_overlap(chunk.layer_range, (layer_start, layer_end)):
                    keep.add((model_id, chunk.chunk_id))

        freed = self._store.gc(keep)
        if freed > 0:
            logger.info("GC freed %.2f MiB", freed / (1024 * 1024))
        return freed

    def list_local_chunks(self, model_id: str) -> list[str]:
        """Вернуть список chunk_id, физически присутствующих в cache_dir.

        Сканирует cache_dir/<model_id>/*.bin без SHA256-верификации.
        """
        model_dir = self._cache_dir / model_id
        if not model_dir.exists():
            return []
        return [f.stem for f in model_dir.glob("*.bin") if f.is_file()]

    # ------------------------------------------------------------------
    # Internal: manifest cache
    # ------------------------------------------------------------------

    async def _get_manifest(self, model_id: str) -> ModelManifest:
        """Получить манифест модели (с кэшированием в памяти)."""
        now = time.monotonic()
        if model_id in self._manifest_cache:
            manifest, fetched_at = self._manifest_cache[model_id]
            if self._manifest_ttl <= 0 or (now - fetched_at) < self._manifest_ttl:
                return manifest

        resp = await self._tracker_client.get_manifest(model_id)
        # Конвертируем ModelManifestResponse -> ModelManifest
        manifest = ModelManifest(
            model_id=resp.model_id,
            num_layers=resp.num_layers,
            hidden_size=resp.hidden_size,
            dtype=resp.dtype,
            cuda_capability_min=resp.cuda_capability_min,
            chunks=resp.chunks,
            manifest_sha256=resp.manifest_sha256,
        )
        self._manifest_cache[model_id] = (manifest, now)
        logger.info(
            "Fetched manifest for %s: %d chunks, %d layers",
            model_id,
            len(manifest.chunks),
            manifest.num_layers,
        )
        return manifest

    # ------------------------------------------------------------------
    # Internal: chunk fetching
    # ------------------------------------------------------------------

    async def _ensure_one_chunk(self, model_id: str, chunk: ChunkSpec) -> Path:
        """Гарантировать наличие одного чанка (с семафором)."""
        async with self._fetch_sem:
            return await self._ensure_one_chunk_inner(model_id, chunk)

    async def _ensure_one_chunk_inner(self, model_id: str, chunk: ChunkSpec) -> Path:
        """Логика загрузки одного чанка без семафора."""
        local_path = self._store.path_for(model_id, chunk.chunk_id)

        # Проверяем локальный файл
        if local_path.exists() and self._store.verify(local_path, chunk.sha256):
            logger.debug("Chunk %s already cached and verified", chunk.chunk_id)
            return local_path

        # Нужно скачать
        await self._fetch_chunk(model_id, chunk)
        return local_path

    async def _fetch_chunk(self, model_id: str, chunk: ChunkSpec) -> None:
        """Скачать чанк от одного из peers.

        Порядок: сначала не-seed peers (распределённая нагрузка), потом seed.
        При hash mismatch — репортим страйк и переходим к следующему peer.

        Raises:
            ChunkFetchError: если все peers исчерпаны.
        """
        peers_resp = await self._tracker_client.get_chunk_peers(model_id, chunk.chunk_id)
        peers = peers_resp.peers

        if not peers:
            raise ChunkFetchError(model_id, chunk.chunk_id, "no peers available")

        # ADR-0033: сначала не-seed (распределённая нагрузка), потом seed
        sorted_peers = sorted(peers, key=lambda p: 1 if p.is_seed else 0)

        local_path = self._store.path_for(model_id, chunk.chunk_id)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        for peer in sorted_peers:
            success = await self._try_fetch_from_peer(model_id, chunk, peer, local_path)
            if success:
                self._pending_declarations.add(chunk.chunk_id)
                return

        raise ChunkFetchError(model_id, chunk.chunk_id, "all peers exhausted")

    async def _try_fetch_from_peer(
        self,
        model_id: str,
        chunk: ChunkSpec,
        peer: ChunkPeer,
        local_path: Path,
    ) -> bool:
        """Попытаться скачать чанк от одного peer'а через QUIC.

        Returns:
            True при успехе, False при ошибке (hash mismatch, connection error, etc.)
        """
        temp_path = local_path.with_suffix(".tmp")
        try:
            actual_sha256 = await asyncio.wait_for(
                _download_chunk_via_quic(
                    peer=peer,
                    model_id=model_id,
                    chunk=chunk,
                    temp_path=temp_path,
                    identity=self._identity,
                ),
                timeout=_CHUNK_DIAL_TIMEOUT_SEC,
            )
        except TimeoutError:
            logger.warning("Chunk %s: timeout fetching from peer %s", chunk.chunk_id, peer.peer_id)
            _safe_unlink(temp_path)
            return False
        except Exception as exc:
            logger.warning(
                "Chunk %s: error fetching from peer %s: %s",
                chunk.chunk_id,
                peer.peer_id,
                exc,
            )
            _safe_unlink(temp_path)
            return False

        if actual_sha256 != chunk.sha256.lower():
            logger.warning(
                "Chunk %s: SHA256 mismatch from peer %s (expected %s, got %s)",
                chunk.chunk_id,
                peer.peer_id,
                chunk.sha256[:16] + "...",
                actual_sha256[:16] + "...",
            )
            _safe_unlink(temp_path)
            # Репортим страйк
            await self._report_hash_mismatch_strike(model_id, chunk, peer, actual_sha256)
            return False

        # SHA256 совпал — атомарное переименование
        temp_path.rename(local_path)
        logger.info(
            "Chunk %s fetched from peer %s (%d bytes)",
            chunk.chunk_id,
            peer.peer_id,
            chunk.byte_size,
        )
        return True

    async def _report_hash_mismatch_strike(
        self,
        model_id: str,
        chunk: ChunkSpec,
        peer: ChunkPeer,
        actual_sha256: str,
    ) -> None:
        """Отправить страйк на трекер за hash mismatch (best-effort)."""
        if self._session_token is None:
            logger.debug("No session_token, skipping strike report for %s", peer.peer_id)
            return
        try:
            import uuid

            from shared.crypto import sign_strike
            from shared.protocol import (
                StrikeEvidence,
                StrikeReportRequest,
            )

            evidence = StrikeEvidence(
                sample_hash=hashlib.sha256(f"{model_id}:{chunk.chunk_id}".encode()).hexdigest(),
                expected_blob_sha256=chunk.sha256,
                got_blob_sha256=actual_sha256,
                l2_distance=1.0,
                hop_index=0,
                envelope_chain=[],
            )
            sig_bytes = sign_strike(
                self._identity.signing_key,
                json.dumps(
                    evidence.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            import base64

            strike_req = StrikeReportRequest(
                session_id=uuid.uuid4(),
                reporter_peer_id=self._identity.peer_id,
                offender_peer_id=peer.peer_id,
                evidence=evidence,
                reporter_signature=base64.b64encode(sig_bytes).decode(),
            )
            await self._tracker_client.report_strike(
                req=strike_req,
                session_token=self._session_token,
            )
            logger.info(
                "Strike reported against peer %s for chunk_hash_mismatch on %s",
                peer.peer_id,
                chunk.chunk_id,
            )
        except Exception as exc:
            logger.warning("Failed to report strike for %s: %s", peer.peer_id, exc)


# ---------------------------------------------------------------------------
# QUIC weight-chunk transport (клиентская сторона)
# ---------------------------------------------------------------------------


async def _download_chunk_via_quic(
    *,
    peer: ChunkPeer,
    model_id: str,
    chunk: ChunkSpec,
    temp_path: Path,
    identity: Identity,
) -> str:
    """Скачать чанк от peer'а через QUIC bidirectional stream.

    Протокол (ADR-0035):
    1. Открыть QUIC соединение к peer.addr.
    2. Отправить WeightChunkRequest (JSON, length-prefixed).
    3. Получить WeightChunkResponse (JSON, length-prefixed) — метаданные.
    4. Принять тело чанка: N фреймов по 64 КБ; последний может быть меньше.
       Конец тела: пустой фрейм (length=0).
    5. Записать в temp_path.
    6. Вычислить SHA256 принятых байт.
    7. Сформировать ChunkReceipt, подписать, отправить обратно.
    8. Закрыть stream.

    Returns:
        SHA256 hex принятого содержимого.

    Raises:
        ConnectionError: при ошибке соединения.
        ValueError: при некорректном ответе от peer.
    """
    from shared.protocol import parse_multiaddr

    parsed = parse_multiaddr(peer.addr)
    host, port = parsed.host, parsed.port

    import ssl

    from aioquic.asyncio import connect
    from aioquic.quic.configuration import QuicConfiguration

    from node.network import generate_self_signed_cert
    from shared.constants import MAX_INFLIGHT_ENVELOPES_PER_SESSION

    cert_pem, key_pem = generate_self_signed_cert(identity.signing_key)

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as cf:
        cf.write(cert_pem)
        cert_path = cf.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as kf:
        kf.write(key_pem)
        key_path = kf.name

    try:
        config = QuicConfiguration(
            alpn_protocols=["llm-swarm/1"],
            is_client=True,
            max_data=MAX_WEIGHT_CHUNK_SIZE_BYTES * MAX_INFLIGHT_ENVELOPES_PER_SESSION,
            max_stream_data=MAX_WEIGHT_CHUNK_SIZE_BYTES,
            verify_mode=ssl.CERT_NONE,
            server_name=host,
        )
        config.load_cert_chain(cert_path, key_path)
    finally:
        os.unlink(cert_path)
        os.unlink(key_path)

    async with connect(host, port, configuration=config) as protocol:
        reader, writer = await protocol.create_stream()

        # Отправляем WeightChunkRequest (с requester_peer_id для rate-limit, ADR-0043)
        req = WeightChunkRequest(
            model_id=model_id,
            chunk_id=chunk.chunk_id,
            requester_peer_id=identity.peer_id,
        )
        req_json = req.model_dump_json().encode("utf-8")
        writer.write(encode_frame(req_json))
        await writer.drain()

        # Читаем WeightChunkResponse (метаданные)
        meta_bytes = await read_frame(reader, max_size=4096, timeout=10.0)
        if not meta_bytes:
            raise ValueError("Empty response from peer (expected WeightChunkResponse)")

        # Пробуем десериализовать — может быть ошибка
        meta_dict = json.loads(meta_bytes)
        if "reason" in meta_dict and "chunk_id" in meta_dict:
            # WeightChunkError
            err = WeightChunkError.model_validate(meta_dict)
            raise ValueError(f"Peer refused chunk: {err.reason}")
        meta = WeightChunkResponse.model_validate(meta_dict)

        # Читаем тело чанка фреймами по 64 КБ
        h = hashlib.sha256()
        total_received = 0
        with temp_path.open("wb") as fh:
            while True:
                frame = await read_frame(
                    reader,
                    max_size=WEIGHT_CHUNK_STREAM_CHUNK_SIZE + 16,
                    timeout=30.0,
                )
                if not frame:
                    # Пустой фрейм = конец тела
                    break
                h.update(frame)
                fh.write(frame)
                total_received += len(frame)
                if total_received >= meta.size_bytes:
                    # Получили всё
                    break

        actual_sha256 = h.hexdigest()

        # Отправляем ChunkReceipt v2 (ADR-0042): inline downloader_pubkey
        ts = int(time.time())
        sig_bytes = sign_chunk_receipt(
            signing_key=identity.signing_key,
            chunk_id=chunk.chunk_id,
            sha256_hex=actual_sha256,
            downloader_peer_id=identity.peer_id,
            uploader_peer_id=peer.peer_id,
            bytes_received=total_received,
            ts=ts,
        )
        # Подкладываем public_key_bytes в base58 (ADR-0042)
        from shared.crypto import public_key_bytes as _pk_bytes

        downloader_pubkey_b58 = base58.b58encode(_pk_bytes(identity.signing_key)).decode()
        receipt = ChunkReceipt(
            chunk_id=chunk.chunk_id,
            sha256=actual_sha256,
            downloader_peer_id=identity.peer_id,
            uploader_peer_id=peer.peer_id,
            bytes_received=total_received,
            ts=ts,
            signature=base58.b58encode(sig_bytes).decode(),
            downloader_pubkey=downloader_pubkey_b58,
        )
        receipt_json = receipt.model_dump_json().encode("utf-8")
        writer.write(encode_frame(receipt_json))
        await writer.drain()
        writer.close()

    return actual_sha256


# ---------------------------------------------------------------------------
# Server-side handler (вызывается из node/network.py)
# ---------------------------------------------------------------------------


async def handle_weight_chunk_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    store: ChunkStore,
    identity: Identity,
    tracker_client: TrackerClient | None,
    session_token: str | None,
    upload_sem: asyncio.Semaphore,
    rate_limiter: UploaderRateLimiter | None = None,
) -> None:
    """Handler входящего weight-chunk QUIC-стрима (server-side, ADR-0035, ADR-0043).

    Протокол:
    1. Читаем WeightChunkRequest.
    2. Rate-limit проверка per-peer (ADR-0043): при reject → WeightChunkError(rate_limited).
    3. Ищем файл в ChunkStore.
    4. Если нет — отвечаем WeightChunkError и закрываем.
    5. Если есть — SHA256 + метаданные + стриминг (asyncio.to_thread для sync IO, E5.4).
    6. Ждём ChunkReceipt, верифицируем (ADR-0042).
    """
    async with upload_sem:
        peer_id_for_ratelimit: str | None = None
        try:
            # Читаем WeightChunkRequest
            req_bytes = await read_frame(reader, max_size=4096, timeout=10.0)
            if not req_bytes:
                writer.close()
                return
            req = WeightChunkRequest.model_validate_json(req_bytes)
            logger.debug("weight-chunk: request for model=%s chunk=%s", req.model_id, req.chunk_id)

            # Извлекаем requester_peer_id для rate-limit (ADR-0043)
            requester_peer_id = getattr(req, "requester_peer_id", None)
            peer_id_for_ratelimit = requester_peer_id

            # Ищем файл для определения размера (нужен для rate-limit)
            chunk_path = store.path_for(req.model_id, req.chunk_id)
            if not chunk_path.exists():
                err = WeightChunkError(
                    chunk_id=req.chunk_id,
                    reason="chunk_not_found",
                    code="chunk_not_found",
                )
                writer.write(encode_frame(err.model_dump_json().encode("utf-8")))
                await writer.drain()
                writer.close()
                return

            chunk_size_bytes = chunk_path.stat().st_size

            # Rate-limit проверка (ADR-0043)
            if rate_limiter is not None and requester_peer_id:
                allowed, retry_after_ms = rate_limiter.check_and_reserve(
                    requester_peer_id, chunk_size_bytes
                )
                if not allowed:
                    err = WeightChunkError(
                        chunk_id=req.chunk_id,
                        reason=f"rate limited, retry after {retry_after_ms}ms",
                        code="rate_limited",
                        retry_after_ms=retry_after_ms,
                    )
                    writer.write(encode_frame(err.model_dump_json().encode("utf-8")))
                    await writer.drain()
                    writer.close()
                    peer_id_for_ratelimit = None  # не вызывать release (не было reserve)
                    return

            # SHA256 и размер — через asyncio.to_thread (sync IO, E5.4, ADR-0043)
            def _compute_sha256_and_size(path: Path) -> tuple[str, int]:
                h = hashlib.sha256()
                size = 0
                with path.open("rb") as fh:
                    for block in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(block)
                        size += len(block)
                return h.hexdigest(), size

            sha256_hex, size_bytes = await asyncio.to_thread(_compute_sha256_and_size, chunk_path)

            # Отправляем WeightChunkResponse (метаданные)
            meta = WeightChunkResponse(
                model_id=req.model_id,
                chunk_id=req.chunk_id,
                sha256=sha256_hex,
                size_bytes=size_bytes,
            )
            writer.write(encode_frame(meta.model_dump_json().encode("utf-8")))

            # Стримим тело чанка фреймами по 64 КБ — через asyncio.to_thread (E5.4)
            async def _stream_chunk() -> None:
                def _read_and_send() -> list[bytes]:
                    chunks = []
                    with chunk_path.open("rb") as fh:
                        while True:
                            data = fh.read(WEIGHT_CHUNK_STREAM_CHUNK_SIZE)
                            if not data:
                                break
                            chunks.append(data)
                    return chunks

                frames = await asyncio.to_thread(_read_and_send)
                for frame_data in frames:
                    writer.write(encode_frame(frame_data))
                # Пустой фрейм = конец тела
                writer.write(encode_frame(b""))

            await _stream_chunk()
            await writer.drain()

            # Ждём ChunkReceipt от downloader'а
            try:
                receipt_bytes = await asyncio.wait_for(
                    read_frame(reader, max_size=4096),
                    timeout=float(WEIGHT_RECEIPT_TIMEOUT_SEC),
                )
            except (TimeoutError, EOFError):
                # ADR-0035: раздача засчитывается best-effort без receipt
                logger.debug(
                    "weight-chunk: no receipt from downloader for chunk %s (best-effort)",
                    req.chunk_id,
                )
                writer.close()
                return

            if receipt_bytes:
                _process_chunk_receipt(
                    receipt_bytes=receipt_bytes,
                    expected_chunk_id=req.chunk_id,
                    uploader_peer_id=identity.peer_id,
                    tracker_client=tracker_client,
                    session_token=session_token,
                )

        except Exception as exc:
            logger.warning("weight-chunk handler error: %s", exc, exc_info=True)
        finally:
            # Освобождаем rate-limit slot (ADR-0043)
            if rate_limiter is not None and peer_id_for_ratelimit is not None:
                rate_limiter.release(peer_id_for_ratelimit)
            try:
                writer.close()
            except Exception:
                pass


def _process_chunk_receipt(
    *,
    receipt_bytes: bytes,
    expected_chunk_id: str,
    uploader_peer_id: str,
    tracker_client: object | None,
    session_token: str | None,
) -> None:
    """Обработать ChunkReceipt v2 от downloader'а (best-effort, без await, ADR-0042).

    Верификация по 6-шаговой процедуре ADR-0042:
      1. Парсинг.
      2. len(downloader_pubkey_bytes) == 32.
      3. peer_id_from_public_key(pubkey) == downloader_peer_id.
      4. uploader_peer_id == expected uploader.
      5. verify_chunk_receipt(VerifyKey(pubkey), signature, ...).
      6. Freshness: |now - ts| <= 300 сек (5 мин).

    Невалидные receipt'ы — drop с логом, без strike (ADR-0042).
    """
    import time as _time

    import base58 as _base58
    from nacl.signing import VerifyKey

    from shared.crypto import peer_id_from_public_key, verify_chunk_receipt

    try:
        receipt = ChunkReceipt.model_validate_json(receipt_bytes)
    except Exception as exc:
        logger.warning("weight-chunk: failed to parse receipt: %s", exc)
        return

    # 1. chunk_id match
    if receipt.chunk_id != expected_chunk_id:
        logger.warning(
            "weight-chunk: receipt chunk_id mismatch: expected %s, got %s",
            expected_chunk_id,
            receipt.chunk_id,
        )
        return

    # 2. Decode downloader_pubkey
    try:
        pubkey_bytes = _base58.b58decode(receipt.downloader_pubkey)
    except Exception:
        logger.warning(
            "weight-chunk: receipt from %s: invalid base58 downloader_pubkey",
            receipt.downloader_peer_id,
        )
        return

    if len(pubkey_bytes) != 32:
        logger.warning(
            "weight-chunk: receipt from %s: downloader_pubkey len=%d (expected 32)",
            receipt.downloader_peer_id,
            len(pubkey_bytes),
        )
        return

    # 3. peer_id_from_public_key(pubkey) == downloader_peer_id
    derived_peer_id = peer_id_from_public_key(pubkey_bytes)
    if derived_peer_id != receipt.downloader_peer_id:
        logger.warning(
            "weight-chunk: receipt peer_id mismatch: derived=%s, claimed=%s",
            derived_peer_id,
            receipt.downloader_peer_id,
        )
        return

    # 4. uploader_peer_id == self
    if receipt.uploader_peer_id != uploader_peer_id:
        logger.warning(
            "weight-chunk: receipt uploader_peer_id mismatch: expected=%s, got=%s",
            uploader_peer_id,
            receipt.uploader_peer_id,
        )
        return

    # 5. verify_chunk_receipt
    try:
        sig_bytes = _base58.b58decode(receipt.signature)
        verify_key = VerifyKey(pubkey_bytes)
        valid = verify_chunk_receipt(
            verify_key=verify_key,
            signature=sig_bytes,
            chunk_id=receipt.chunk_id,
            sha256_hex=receipt.sha256,
            downloader_peer_id=receipt.downloader_peer_id,
            uploader_peer_id=receipt.uploader_peer_id,
            bytes_received=receipt.bytes_received,
            ts=receipt.ts,
        )
    except Exception as exc:
        logger.warning(
            "weight-chunk: receipt signature verification error from %s: %s",
            receipt.downloader_peer_id,
            exc,
        )
        return

    if not valid:
        logger.warning(
            "weight-chunk: receipt invalid Ed25519 signature from %s for chunk %s",
            receipt.downloader_peer_id,
            receipt.chunk_id,
        )
        return

    # 6. Freshness: |now - ts| <= 5 минут (ADR-0042)
    now_ts = int(_time.time())
    if abs(now_ts - receipt.ts) > 300:
        logger.warning(
            "weight-chunk: receipt from %s: ts out of freshness window "
            "(receipt_ts=%d, now=%d, delta=%d)",
            receipt.downloader_peer_id,
            receipt.ts,
            now_ts,
            abs(now_ts - receipt.ts),
        )
        return

    # Все проверки пройдены — receipt валиден
    logger.info(
        "weight-chunk: verified receipt from %s for chunk %s (%d bytes, ts=%d)",
        receipt.downloader_peer_id,
        receipt.chunk_id,
        receipt.bytes_received,
        receipt.ts,
    )
    # TODO (Stage 6 followup): POST /api/v1/accounting/report type=chunk_transfer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Проверить, перекрываются ли два полуоткрытых диапазона [a0, a1) и [b0, b1)."""
    a0, a1 = a
    b0, b1 = b
    return a0 < b1 and b0 < a1


def _safe_unlink(path: Path) -> None:
    """Удалить файл, игнорируя ошибки если файл не существует."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
