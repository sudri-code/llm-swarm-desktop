"""Stage 1: httpx-обёртка над tracker REST API (register, heartbeat). Permanent (ADR-0007).

Stage 4 (ADR-0020): добавлен метод authenticate() — challenge-response.

Async context manager вокруг httpx.AsyncClient.
Ошибки 401/404 → специализированные исключения (нода должна перерегистрироваться).
5xx и сетевые ошибки → выкидывают httpx.HTTPError (caller сам решает backoff-стратегию).
"""

from __future__ import annotations

import base64
import re
import time

import httpx
from nacl.signing import SigningKey

from shared.crypto import sign_auth
from shared.protocol import (
    AuthRequest,
    AuthResponse,
    ChunkPeersResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    ModelManifestResponse,
    NodeChunkDeclaration,
    NodeChunkDeclarationResponse,
    RegisterRequest,
    RegisterResponse,
    StrikeReportRequest,
    StrikeReportResponse,
)


class TrackerError(Exception):
    """Базовый класс ошибок трекера."""


class TrackerAuthError(TrackerError):
    """401 Unauthorized — токен недействителен. Нода должна перерегистрироваться."""


class TrackerNotFoundError(TrackerError):
    """404 Not Found — нода не найдена на трекере. Нода должна перерегистрироваться."""


def canonicalize_tracker_url(raw_url: str) -> str:
    """Привести tracker URL к канонической форме для build_auth_payload (ADR-0020).

    Правила совпадают с tracker/main.py:_canonical_tracker_url:
    lowercase scheme+host, без trailing slash. Порт явный если нестандартный
    (уже содержится в raw_url — не добавляем/не убираем).

    Используется как для HTTP-вызовов, так и для подписи challenge-response.
    """
    raw_url = raw_url.rstrip("/")
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+\-.]*)(://[^/]*)(.*)?$", raw_url)
    if m:
        scheme = m.group(1).lower()
        authority = m.group(2).lower()
        path = m.group(3) or ""
        return scheme + authority + path
    return raw_url.lower()


class TrackerClient:
    """Асинхронный HTTP-клиент для общения с tracker API.

    Использование:
        async with TrackerClient(base_url) as client:
            resp = await client.register(req)
            await client.heartbeat(hb_req)
    """

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> TrackerClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def authenticate(
        self,
        challenge_id: object,  # uuid.UUID
        challenge_b64: str,
        signing_key: SigningKey,
        peer_id_raw: bytes,
        tracker_url: str,
    ) -> str:
        """POST /api/v1/nodes/auth — challenge-response (ADR-0020).

        Args:
            challenge_id: UUID challenge из RegisterResponse.
            challenge_b64: base64-encoded 32-byte challenge из RegisterResponse.
            signing_key: Ed25519 SigningKey ноды.
            peer_id_raw: 32 байта sha256(public_key) raw (НЕ base58).
            tracker_url: каноническая форма URL трекера (используется как есть
                         для подписи — вызывающий уже должен передать canonical).

        Returns:
            session_token из AuthResponse.

        Raises:
            TrackerAuthError: 401 — подпись отвергнута / challenge истёк.
            httpx.HTTPStatusError: прочие 4xx/5xx.
            httpx.HTTPError: сетевые ошибки.
        """
        challenge = base64.b64decode(challenge_b64)
        issued_at_ms = int(time.time() * 1000)
        sig_bytes = sign_auth(
            signing_key=signing_key,
            challenge=challenge,
            peer_id_bytes=peer_id_raw,
            tracker_url=tracker_url,
            issued_at_ms=issued_at_ms,
        )
        sig_b64 = base64.b64encode(sig_bytes).decode()
        auth_req = AuthRequest(
            challenge_id=challenge_id,
            signature=sig_b64,
            issued_at_ms=issued_at_ms,
        )
        resp = await self._post("/api/v1/nodes/auth", auth_req)
        auth_resp = AuthResponse.model_validate(resp.json())
        return auth_resp.session_token

    async def register(self, req: RegisterRequest) -> RegisterResponse:
        """POST /api/v1/nodes/register.

        Args:
            req: RegisterRequest DTO.

        Returns:
            RegisterResponse DTO.

        Raises:
            TrackerAuthError: если сервер вернул 401.
            TrackerNotFoundError: если сервер вернул 404.
            httpx.HTTPStatusError: для прочих 4xx/5xx.
            httpx.HTTPError: при сетевых ошибках (timeout, connect error и т.п.).
        """
        resp = await self._post("/api/v1/nodes/register", req)
        return RegisterResponse.model_validate(resp.json())

    async def get_manifest(self, model_id: str) -> ModelManifestResponse:
        """GET /api/v1/models/{model_id}/manifest (Stage 5, ADR-0030).

        Args:
            model_id: идентификатор модели.

        Returns:
            ModelManifestResponse с манифестом модели.

        Raises:
            TrackerNotFoundError: если модель не найдена (404).
            httpx.HTTPStatusError: прочие 4xx/5xx.
            httpx.HTTPError: сетевые ошибки.
        """
        assert self._client is not None, "TrackerClient must be used as async context manager"
        resp = await self._client.get(f"/api/v1/models/{model_id}/manifest")
        if resp.status_code == 404:
            raise TrackerNotFoundError(f"Model not found: {model_id!r}")
        resp.raise_for_status()
        return ModelManifestResponse.model_validate(resp.json())

    async def get_chunk_peers(self, model_id: str, chunk_id: str) -> ChunkPeersResponse:
        """GET /api/v1/models/{model_id}/chunks/{chunk_id}/peers (Stage 5, ADR-0033).

        Args:
            model_id: идентификатор модели.
            chunk_id: идентификатор чанка.

        Returns:
            ChunkPeersResponse со списком держателей чанка.

        Raises:
            TrackerNotFoundError: если чанк или модель не найдены (404).
            httpx.HTTPStatusError: прочие 4xx/5xx.
            httpx.HTTPError: сетевые ошибки.
        """
        assert self._client is not None, "TrackerClient must be used as async context manager"
        resp = await self._client.get(f"/api/v1/models/{model_id}/chunks/{chunk_id}/peers")
        if resp.status_code == 404:
            raise TrackerNotFoundError(
                f"Chunk not found: model_id={model_id!r}, chunk_id={chunk_id!r}"
            )
        resp.raise_for_status()
        return ChunkPeersResponse.model_validate(resp.json())

    async def declare_chunks(
        self,
        chunk_ids: list[str],
        session_token: str,
    ) -> NodeChunkDeclarationResponse:
        """POST /api/v1/nodes/me/chunks — декларация чанков (Stage 5, ADR-0035).

        Args:
            chunk_ids: список chunk_id, которые нода держит.
            session_token: действующий session_token ноды.

        Returns:
            NodeChunkDeclarationResponse.

        Raises:
            TrackerAuthError: 401 — токен недействителен.
            httpx.HTTPStatusError: прочие 4xx/5xx.
            httpx.HTTPError: сетевые ошибки.
        """
        assert self._client is not None, "TrackerClient must be used as async context manager"
        decl = NodeChunkDeclaration(chunk_ids=chunk_ids)
        resp = await self._client.post(
            "/api/v1/nodes/me/chunks",
            json=decl.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {session_token}"},
        )
        if resp.status_code == 401:
            raise TrackerAuthError("Tracker returned 401 for declare_chunks: token invalid.")
        resp.raise_for_status()
        return NodeChunkDeclarationResponse.model_validate(resp.json())

    async def report_strike(
        self,
        req: StrikeReportRequest,
        session_token: str,
    ) -> StrikeReportResponse:
        """POST /api/v1/strikes/report — репорт страйка (ADR-0024, Stage 5 extension).

        Args:
            req: StrikeReportRequest DTO.
            session_token: session_token ноды (для auth).

        Returns:
            StrikeReportResponse.
        """
        assert self._client is not None, "TrackerClient must be used as async context manager"
        resp = await self._client.post(
            "/api/v1/strikes/report",
            json=req.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {session_token}"},
        )
        if resp.status_code == 401:
            raise TrackerAuthError("Tracker returned 401 for report_strike: token invalid.")
        resp.raise_for_status()
        return StrikeReportResponse.model_validate(resp.json())

    async def heartbeat(self, req: HeartbeatRequest) -> HeartbeatResponse:
        """POST /api/v1/nodes/heartbeat.

        Args:
            req: HeartbeatRequest DTO.

        Returns:
            HeartbeatResponse DTO.

        Raises:
            TrackerAuthError: если сервер вернул 401 (токен протух / неверный).
            TrackerNotFoundError: если сервер вернул 404 (нода забыта трекером).
            httpx.HTTPStatusError: для прочих 4xx/5xx.
            httpx.HTTPError: при сетевых ошибках.
        """
        resp = await self._post("/api/v1/nodes/heartbeat", req)
        return HeartbeatResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(
        self, path: str, body: RegisterRequest | HeartbeatRequest | AuthRequest
    ) -> httpx.Response:
        """Отправить POST-запрос и вернуть httpx.Response.

        Конвертирует 401 → TrackerAuthError, 404 → TrackerNotFoundError,
        остальные non-2xx → httpx.HTTPStatusError.
        """
        assert self._client is not None, "TrackerClient must be used as async context manager"

        resp = await self._client.post(path, json=body.model_dump(mode="json"))

        if resp.status_code == 401:
            raise TrackerAuthError(
                f"Tracker returned 401 for {path}: token is invalid or expired. "
                "Node must re-register."
            )
        if resp.status_code == 404:
            raise TrackerNotFoundError(
                f"Tracker returned 404 for {path}: node not found. Node must re-register."
            )

        resp.raise_for_status()
        return resp
