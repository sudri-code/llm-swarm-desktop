"""Unit tests for BFFClient and auth models (W1.3, ADR-0002).

Uses ``respx`` to mock httpx; no real network calls.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx
from pydantic import ValidationError

from agent.auth.bff_client import BFFClient, BFFError, BFFRateLimitedError
from agent.auth.models import (
    DevicePollDenied,
    DevicePollExpired,
    DevicePollPending,
    DevicePollRequest,
    DevicePollSlowDown,
    DevicePollSuccess,
    DeviceStartRequest,
    DeviceStartResponse,
    HardwareSummary,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_BASE_URL = "https://sudri.ru"
_START_PATH = "/api/v1/auth/device/start"
_POLL_PATH = "/api/v1/auth/device/poll"


def _start_req() -> DeviceStartRequest:
    return DeviceStartRequest(
        device_pubkey="dGVzdC1wdWJrZXktMzJieXRlcw",  # base64url, no padding
        device_label="Test Machine",
        client_version="0.1.0.dev0",
        hardware_summary=HardwareSummary(
            vram_gib=8.0,
            ram_gib=16.0,
            gpu_model="NVIDIA RTX 3080",
            platform="linux-x86_64",
        ),
    )


def _poll_req() -> DevicePollRequest:
    return DevicePollRequest(
        device_code="opaque-device-code-abc123",
        device_pubkey="dGVzdC1wdWJrZXktMzJieXRlcw",
    )


_START_RESPONSE_BODY: dict[str, Any] = {
    "device_code": "opaque-device-code-abc123",
    "user_code": "ABCD1234",
    "verification_uri": "https://sudri.ru/link",
    "verification_uri_complete": "https://sudri.ru/link?code=ABCD1234",
    "expires_in": 600,
    "interval": 3,
}


# ---------------------------------------------------------------------------
# device_start tests
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_device_start_success() -> None:
    """Happy path: 200 with valid JSON → DeviceStartResponse."""
    respx.post(_BASE_URL + _START_PATH).mock(
        return_value=httpx.Response(200, json=_START_RESPONSE_BODY)
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        resp = await client.device_start(_start_req())

    assert isinstance(resp, DeviceStartResponse)
    assert resp.device_code == "opaque-device-code-abc123"
    assert resp.user_code == "ABCD1234"
    assert resp.expires_in == 600
    assert resp.interval == 3


@respx.mock
@pytest.mark.asyncio
async def test_device_start_request_body() -> None:
    """Verify that the correct JSON body is sent to BFF."""
    route = respx.post(_BASE_URL + _START_PATH).mock(
        return_value=httpx.Response(200, json=_START_RESPONSE_BODY)
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        await client.device_start(_start_req())

    sent_json = route.calls[0].request.read()
    import json

    body = json.loads(sent_json)
    assert body["device_pubkey"] == "dGVzdC1wdWJrZXktMzJieXRlcw"
    assert body["device_label"] == "Test Machine"
    assert body["client_version"] == "0.1.0.dev0"
    assert body["hardware_summary"]["vram_gib"] == 8.0


@respx.mock
@pytest.mark.asyncio
async def test_device_start_request_headers() -> None:
    """User-Agent header must contain 'llm-swarm-desktop'."""
    respx.post(_BASE_URL + _START_PATH).mock(
        return_value=httpx.Response(200, json=_START_RESPONSE_BODY)
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        await client.device_start(_start_req())

    sent_headers = dict(respx.calls[0].request.headers)
    user_agent = sent_headers.get("user-agent", "")
    assert "llm-swarm-desktop" in user_agent


@respx.mock
@pytest.mark.asyncio
async def test_device_start_without_hardware_summary() -> None:
    """hardware_summary=None must not appear in serialised body."""
    route = respx.post(_BASE_URL + _START_PATH).mock(
        return_value=httpx.Response(200, json=_START_RESPONSE_BODY)
    )
    req = DeviceStartRequest(
        device_pubkey="dGVzdA",
        device_label="minimal",
        client_version="0.1.0",
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        await client.device_start(req)

    import json

    body = json.loads(route.calls[0].request.read())
    assert "hardware_summary" not in body


@respx.mock
@pytest.mark.asyncio
async def test_device_start_500_raises_bff_error() -> None:
    respx.post(_BASE_URL + _START_PATH).mock(
        return_value=httpx.Response(
            500, json={"error": "internal_error", "error_description": "oops"}
        )
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        with pytest.raises(BFFError) as exc_info:
            await client.device_start(_start_req())

    err = exc_info.value
    assert err.status_code == 500
    assert err.error == "internal_error"
    assert err.error_description == "oops"


@respx.mock
@pytest.mark.asyncio
async def test_device_start_429_raises_rate_limited() -> None:
    respx.post(_BASE_URL + _START_PATH).mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "15"},
            json={"error": "rate_limited"},
        )
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        with pytest.raises(BFFRateLimitedError) as exc_info:
            await client.device_start(_start_req())

    err = exc_info.value
    assert err.status_code == 429
    assert err.retry_after == 15


@respx.mock
@pytest.mark.asyncio
async def test_device_start_400_raises_bff_error() -> None:
    respx.post(_BASE_URL + _START_PATH).mock(
        return_value=httpx.Response(
            400, json={"error": "invalid_pubkey", "error_description": "bad key"}
        )
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        with pytest.raises(BFFError) as exc_info:
            await client.device_start(_start_req())

    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# device_poll tests — all status variants
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_poll_authorization_pending() -> None:
    """RFC 8628 spelling: 'authorization_pending', NOT 'pending'."""
    respx.post(_BASE_URL + _POLL_PATH).mock(
        return_value=httpx.Response(200, json={"status": "authorization_pending"})
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        resp = await client.device_poll(_poll_req())

    assert isinstance(resp, DevicePollPending)
    assert resp.status == "authorization_pending"


@respx.mock
@pytest.mark.asyncio
async def test_poll_slow_down() -> None:
    respx.post(_BASE_URL + _POLL_PATH).mock(
        return_value=httpx.Response(200, json={"status": "slow_down", "interval": 7})
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        resp = await client.device_poll(_poll_req())

    assert isinstance(resp, DevicePollSlowDown)
    assert resp.interval == 7


@respx.mock
@pytest.mark.asyncio
async def test_poll_expired_token() -> None:
    respx.post(_BASE_URL + _POLL_PATH).mock(
        return_value=httpx.Response(200, json={"status": "expired_token"})
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        resp = await client.device_poll(_poll_req())

    assert isinstance(resp, DevicePollExpired)
    assert resp.status == "expired_token"


@respx.mock
@pytest.mark.asyncio
async def test_poll_access_denied() -> None:
    respx.post(_BASE_URL + _POLL_PATH).mock(
        return_value=httpx.Response(200, json={"status": "access_denied"})
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        resp = await client.device_poll(_poll_req())

    assert isinstance(resp, DevicePollDenied)
    assert resp.status == "access_denied"


@respx.mock
@pytest.mark.asyncio
async def test_poll_success_ok() -> None:
    """Status 'ok' (ADR-0002 §A) with all fields."""
    body = {
        "status": "ok",
        "device_token": "opaque-token-xyz",
        "device_id": "dev-uuid-1234",
        "user_id": "user-uuid-5678",
        "token_expires_at": "2027-05-06T00:00:00Z",
        "scope": ["node:run", "accounting:report", "balance:read"],
    }
    respx.post(_BASE_URL + _POLL_PATH).mock(
        return_value=httpx.Response(200, json=body)
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        resp = await client.device_poll(_poll_req())

    assert isinstance(resp, DevicePollSuccess)
    assert resp.status == "ok"
    assert resp.device_id == "dev-uuid-1234"
    assert resp.user_id == "user-uuid-5678"
    assert resp.scope == ["node:run", "accounting:report", "balance:read"]
    assert resp.token_expires_at == "2027-05-06T00:00:00Z"
    # device_token is present in the model but we intentionally don't log it


@respx.mock
@pytest.mark.asyncio
async def test_poll_429_raises_rate_limited() -> None:
    respx.post(_BASE_URL + _POLL_PATH).mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "3"},
            json={"error": "rate_limited"},
        )
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        with pytest.raises(BFFRateLimitedError) as exc_info:
            await client.device_poll(_poll_req())

    assert exc_info.value.retry_after == 3


@respx.mock
@pytest.mark.asyncio
async def test_poll_request_body() -> None:
    """device_code and device_pubkey must be sent in poll body."""
    route = respx.post(_BASE_URL + _POLL_PATH).mock(
        return_value=httpx.Response(200, json={"status": "authorization_pending"})
    )
    async with BFFClient(base_url=_BASE_URL) as client:
        await client.device_poll(_poll_req())

    import json

    body = json.loads(route.calls[0].request.read())
    assert body["device_code"] == "opaque-device-code-abc123"
    assert body["device_pubkey"] == "dGVzdC1wdWJrZXktMzJieXRlcw"


# ---------------------------------------------------------------------------
# Regression guard: bare "pending" must NOT parse as a valid status
# ---------------------------------------------------------------------------


def test_pending_bare_status_rejected_by_pydantic() -> None:
    """'pending' (without 'authorization_') must fail pydantic validation.

    This is the regression guard from the changelog nit (2026-05-06):
    the BFF uses RFC 8628 spelling 'authorization_pending', never 'pending'.
    """
    from pydantic import TypeAdapter

    from agent.auth.models import DevicePollResponse

    adapter: TypeAdapter[DevicePollResponse] = TypeAdapter(DevicePollResponse)  # type: ignore[type-arg]

    with pytest.raises(ValidationError):
        adapter.validate_python({"status": "pending"})


# ---------------------------------------------------------------------------
# base_url resolution
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When env LLM_SWARM_BFF_URL is set, it should be used as base_url."""
    custom_url = "https://staging.sudri.ru"
    monkeypatch.setenv("LLM_SWARM_BFF_URL", custom_url)

    respx.post(custom_url + _START_PATH).mock(
        return_value=httpx.Response(200, json=_START_RESPONSE_BODY)
    )

    async with BFFClient() as client:
        resp = await client.device_start(_start_req())

    assert isinstance(resp, DeviceStartResponse)


@respx.mock
@pytest.mark.asyncio
async def test_base_url_fallback_when_env_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """When env var absent and no explicit base_url, fallback to https://sudri.ru."""
    monkeypatch.delenv("LLM_SWARM_BFF_URL", raising=False)

    respx.post(_BASE_URL + _START_PATH).mock(
        return_value=httpx.Response(200, json=_START_RESPONSE_BODY)
    )

    async with BFFClient() as client:
        resp = await client.device_start(_start_req())

    assert isinstance(resp, DeviceStartResponse)


@respx.mock
@pytest.mark.asyncio
async def test_explicit_base_url_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit base_url arg takes priority over env var."""
    monkeypatch.setenv("LLM_SWARM_BFF_URL", "https://should-not-be-used.example.com")

    respx.post(_BASE_URL + _START_PATH).mock(
        return_value=httpx.Response(200, json=_START_RESPONSE_BODY)
    )

    async with BFFClient(base_url=_BASE_URL) as client:
        resp = await client.device_start(_start_req())

    assert isinstance(resp, DeviceStartResponse)


# ---------------------------------------------------------------------------
# Model contract / extra fields
# ---------------------------------------------------------------------------


def test_start_response_rejects_extra_fields() -> None:
    """extra='forbid' must catch contract drift (unknown new fields)."""
    with pytest.raises(ValidationError):
        DeviceStartResponse(
            device_code="x",
            user_code="ABCD1234",
            verification_uri="https://sudri.ru/link",
            verification_uri_complete="https://sudri.ru/link?code=ABCD1234",
            expires_in=600,
            interval=3,
            unknown_new_field="oops",  # type: ignore[call-arg]
        )


def test_hardware_summary_optional_fields() -> None:
    """HardwareSummary with all None fields is valid (all optional)."""
    h = HardwareSummary()
    assert h.vram_gib is None
    assert h.platform is None


def test_poll_success_scope_list() -> None:
    """scope must be a list of strings in DevicePollSuccess."""
    s = DevicePollSuccess(
        status="ok",
        device_token="tok",
        device_id="did",
        user_id="uid",
        token_expires_at="2027-01-01T00:00:00Z",
        scope=["node:run"],
    )
    assert s.scope == ["node:run"]
