"""Async HTTP client for BFF device authorization endpoints (ADR-0002).

Usage::

    async with BFFClient() as client:
        start_resp = await client.device_start(DeviceStartRequest(...))
        poll_resp = await client.device_poll(DevicePollRequest(...))

Sensitive fields (device_token, device_code, device_pubkey) are NEVER
written to logs.  The client logs only method name and HTTP status codes.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import PackageNotFoundError
from typing import Any

import httpx
from pydantic import TypeAdapter

from agent.auth.models import (
    DevicePollRequest,
    DevicePollResponse,
    DeviceStartRequest,
    DeviceStartResponse,
)

# ---------------------------------------------------------------------------
# Package version (single source of truth in pyproject.toml via importlib)
# ---------------------------------------------------------------------------

def _get_version() -> str:
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("llm-swarm-desktop")
    except PackageNotFoundError:
        return "0.0.0.dev"


_VERSION = _get_version()

_DEFAULT_BASE_URL = "https://sudri.ru"
_DEFAULT_TIMEOUT = 10.0  # seconds

logger = logging.getLogger(__name__)

# TypeAdapter for the discriminated union — reuse for efficiency
_poll_adapter: TypeAdapter[DevicePollResponse] = TypeAdapter(DevicePollResponse)  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BFFError(Exception):
    """Raised when BFF returns an unexpected HTTP error.

    ``status_code`` is the HTTP status (e.g. 400, 500).
    ``error`` and ``error_description`` come from the BFF JSON body when
    available.
    """

    def __init__(
        self,
        status_code: int,
        error: str,
        error_description: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.error = error
        self.error_description = error_description
        super().__init__(f"BFF error {status_code}: {error}")


class BFFRateLimitedError(BFFError):
    """Raised on HTTP 429 Too Many Requests."""

    def __init__(
        self,
        retry_after: int | None = None,
        error_description: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            status_code=429,
            error="rate_limited",
            error_description=error_description,
        )


# ---------------------------------------------------------------------------
# BFFClient
# ---------------------------------------------------------------------------


class BFFClient:
    """Thin async HTTP client wrapping BFF device authorization API.

    Parameters
    ----------
    base_url:
        Base URL for the BFF.  Falls back to ``LLM_SWARM_BFF_URL`` env var,
        then ``https://sudri.ru``.
    http_client:
        Optional pre-configured ``httpx.AsyncClient``.  If *None* an internal
        client is created; it must then be cleaned up via ``await client.aclose()``
        or the async context manager.
    """

    def __init__(
        self,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_base = (
            base_url
            or os.environ.get("LLM_SWARM_BFF_URL")
            or _DEFAULT_BASE_URL
        )
        self._base_url = resolved_base.rstrip("/")
        self._owned = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=_DEFAULT_TIMEOUT,
            headers={"User-Agent": f"llm-swarm-desktop/{_VERSION}"},
        )

    # -- context manager -------------------------------------------------

    async def __aenter__(self) -> BFFClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owned:
            await self._http.aclose()

    # -- public API -------------------------------------------------------

    async def device_start(self, req: DeviceStartRequest) -> DeviceStartResponse:
        """POST /api/v1/auth/device/start — initiate device authorization flow.

        Returns parsed ``DeviceStartResponse`` on success.
        Raises ``BFFRateLimitedError`` on 429, ``BFFError`` on other failures.
        """
        logger.debug("device_start: sending request to BFF")
        body = req.model_dump(exclude_none=True)
        response = await self._http.post(
            "/api/v1/auth/device/start",
            json=body,
        )
        logger.debug("device_start: HTTP %s", response.status_code)
        _raise_for_bff_error(response)
        return DeviceStartResponse.model_validate(response.json())

    async def device_poll(self, req: DevicePollRequest) -> DevicePollResponse:  # type: ignore[return]
        """POST /api/v1/auth/device/poll — poll for authorization result.

        Returns one of the ``DevicePoll*`` variants discriminated on ``status``.
        Raises ``BFFRateLimitedError`` on 429, ``BFFError`` on other failures.
        """
        logger.debug("device_poll: sending request to BFF")
        body = req.model_dump()
        response = await self._http.post(
            "/api/v1/auth/device/poll",
            json=body,
        )
        logger.debug("device_poll: HTTP %s", response.status_code)
        _raise_for_bff_error(response)
        return _poll_adapter.validate_python(response.json())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _raise_for_bff_error(response: httpx.Response) -> None:
    """Raise ``BFFError`` (or subclass) for 4xx/5xx responses."""
    if response.is_success:
        return

    # Try to extract structured error body
    error = "http_error"
    error_description: str | None = None
    try:
        payload = response.json()
        error = payload.get("error", error)
        error_description = payload.get("error_description")
    except Exception:
        pass

    if response.status_code == 429:
        retry_after_raw = response.headers.get("Retry-After")
        retry_after = int(retry_after_raw) if retry_after_raw is not None else None
        raise BFFRateLimitedError(
            retry_after=retry_after,
            error_description=error_description,
        )

    raise BFFError(
        status_code=response.status_code,
        error=error,
        error_description=error_description,
    )
