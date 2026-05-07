"""Unit tests for PairingController (pairing.md Stage 1 W3, ADR-0002).

Uses:
- ``respx`` to mock BFF HTTP calls.
- ``FakeKeychain`` (in-memory dict) implementing the Keychain API.
- Injected ``now``, ``sleep``, ``identity_factory`` for deterministic tests.
- ``pytest-asyncio`` in ``asyncio_mode = "auto"``.
"""

from __future__ import annotations

import asyncio
import base64 as _b64
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from agent.auth.pairing import (
    PairingBridge,
    PairingController,
    PairingFailed,
    PairingFailureReason,
    PairingPolling,
    PairingStarted,
    PairingState,
    PairingSucceeded,
    _default_label_sanitizer,
    _format_user_code,
)
from agent.identity import compute_fingerprint, compute_peer_id

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_FIXED_NOW = datetime(2026, 5, 7, 12, 0, 0, tzinfo=UTC)
_EXPIRES_AT = _FIXED_NOW + timedelta(seconds=600)

# Deterministic keypair for tests
_FAKE_PRIVKEY = b"\x01" * 32
_FAKE_PUBKEY = b"\x02" * 32


def _fake_identity_factory() -> tuple[bytes, bytes]:
    return _FAKE_PRIVKEY, _FAKE_PUBKEY


# Compute expected values from the fake public key
_EXPECTED_PEER_ID = compute_peer_id(_FAKE_PUBKEY)
_EXPECTED_FINGERPRINT = compute_fingerprint(_FAKE_PUBKEY)
_EXPECTED_PUBKEY_B64 = _b64.urlsafe_b64encode(_FAKE_PUBKEY).rstrip(b"=").decode()

# ---------------------------------------------------------------------------
# FakeKeychain
# ---------------------------------------------------------------------------


class FakeKeychain:
    """In-memory keychain for tests."""

    def __init__(
        self, *, fail_on_store_privkey: bool = False, fail_on_store_token: bool = False
    ) -> None:
        self._store: dict[str, Any] = {}
        self._fail_on_store_privkey = fail_on_store_privkey
        self._fail_on_store_token = fail_on_store_token

    def store_private_key(self, peer_id: str, private_key_bytes: bytes) -> None:
        if self._fail_on_store_privkey:
            import keyring.errors
            raise keyring.errors.KeyringError("keychain locked")
        self._store[f"{peer_id}:privkey"] = private_key_bytes

    def load_private_key(self, peer_id: str) -> bytes | None:
        return self._store.get(f"{peer_id}:privkey")

    def store_device_token(self, peer_id: str, token: str) -> None:
        if self._fail_on_store_token:
            import keyring.errors
            raise keyring.errors.KeyringError("keychain locked")
        self._store[f"{peer_id}:device_token"] = token

    def load_device_token(self, peer_id: str) -> str | None:
        return self._store.get(f"{peer_id}:device_token")

    def delete_device(self, peer_id: str) -> None:
        self._store.pop(f"{peer_id}:privkey", None)
        self._store.pop(f"{peer_id}:device_token", None)

    def has_device_token(self, peer_id: str) -> bool:
        return f"{peer_id}:device_token" in self._store


# ---------------------------------------------------------------------------
# Async sleep replacement (instant)
# ---------------------------------------------------------------------------


async def _instant_sleep(_: float) -> None:
    await asyncio.sleep(0)  # yield to event loop without real delay


# ---------------------------------------------------------------------------
# Standard BFF response helpers
# ---------------------------------------------------------------------------

_START_RESPONSE = {
    "device_code": "FAKE_DEVICE_CODE_AAAA",
    "user_code": "ABCDEFGH",
    "verification_uri": "https://sudri.ru/link",
    "verification_uri_complete": "https://sudri.ru/link?code=ABCD-EFGH",
    "expires_in": 600,
    "interval": 3,
}

_PENDING_RESPONSE = {"status": "authorization_pending"}

_OK_RESPONSE = {
    "status": "ok",
    "device_token": "tok_secret_abc123",
    "device_id": "dev-uuid-1234",
    "user_id": "user-uuid-5678",
    "token_expires_at": "2027-05-07T12:00:00Z",
    "scope": ["node:run", "accounting:report", "balance:read"],
}

# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------


def _make_controller(
    *,
    keychain: FakeKeychain | None = None,
    now: Any = None,
    sleep: Any = None,
) -> tuple[PairingController, PairingBridge, list[Any]]:
    """Create controller + bridge + captured events list."""
    bridge = PairingBridge()
    events: list[Any] = []
    bridge.event_received.connect(events.append)

    ctrl = PairingController(
        bridge=bridge,
        keychain=keychain or FakeKeychain(),
        identity_factory=_fake_identity_factory,
        now=now or (lambda: _FIXED_NOW),
        sleep=sleep or _instant_sleep,
    )
    return ctrl, bridge, events


async def _wait_for_terminal(ctrl: PairingController, timeout: float = 2.0) -> None:
    """Wait until controller reaches a terminal or error state."""
    deadline = asyncio.get_event_loop().time() + timeout
    terminal = {
        PairingState.SUCCESS,
        PairingState.ERROR_EXPIRED,
        PairingState.ERROR_DENIED,
        PairingState.ERROR_NETWORK,
        PairingState.ERROR_INVALID,
        PairingState.ERROR_KEYCHAIN,
        PairingState.CANCELLED,
        PairingState.UNPAIRED,
    }
    while ctrl.state not in terminal:
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"Controller stuck in state: {ctrl.state}")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------


@respx.mock
async def test_happy_path_two_pending_then_ok(tmp_path: Any) -> None:
    """start → 2x authorization_pending → ok → SUCCESS; device_token in FakeKeychain."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    poll_route = respx.post("https://sudri.ru/api/v1/auth/device/poll")
    poll_route.side_effect = [
        httpx.Response(200, json=_PENDING_RESPONSE),
        httpx.Response(200, json=_PENDING_RESPONSE),
        httpx.Response(200, json=_OK_RESPONSE),
    ]

    keychain = FakeKeychain()
    ctrl, _, events = _make_controller(keychain=keychain)

    # Patch state.json dir to tmp_path
    with patch("agent.auth.pairing._state_json_path", return_value=tmp_path / "state.json"):
        ctrl.start_pairing("My MacBook")
        await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.SUCCESS

    # Events: PairingStarted, 3× PairingPolling, PairingSucceeded
    assert isinstance(events[0], PairingStarted)
    assert events[0].fingerprint_raw == _EXPECTED_FINGERPRINT
    assert len(events[0].fingerprint_raw) == 12

    polling_events = [e for e in events if isinstance(e, PairingPolling)]
    assert len(polling_events) == 3
    assert polling_events[0].attempt == 1
    assert polling_events[2].attempt == 3

    succeeded = next(e for e in events if isinstance(e, PairingSucceeded))
    assert succeeded.device_id == "dev-uuid-1234"
    assert succeeded.user_id == "user-uuid-5678"
    assert succeeded.fingerprint_raw == _EXPECTED_FINGERPRINT
    assert len(succeeded.fingerprint_raw) == 12
    # device_token is NOT in the event
    assert not hasattr(succeeded, "device_token")

    # device_token stored in keychain
    assert keychain.has_device_token(_EXPECTED_PEER_ID)

    # state.json created
    import json as _json
    state = _json.loads((tmp_path / "state.json").read_text())
    assert "paired_at" in state
    assert state["fingerprint_raw"] == _EXPECTED_FINGERPRINT
    assert state["peer_id"] == _EXPECTED_PEER_ID


# ---------------------------------------------------------------------------
# Tests: slow_down — interval grows
# ---------------------------------------------------------------------------


@respx.mock
async def test_slow_down_increases_interval() -> None:
    """slow_down response must increase (or preserve) interval, never decrease."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )

    intervals_used: list[float] = []
    original_sleep = _instant_sleep

    async def recording_sleep(delay: float) -> None:
        intervals_used.append(delay)
        await original_sleep(delay)

    slow_down_resp = {"status": "slow_down", "interval": 8}
    poll_route = respx.post("https://sudri.ru/api/v1/auth/device/poll")
    poll_route.side_effect = [
        httpx.Response(200, json=slow_down_resp),
        httpx.Response(200, json=slow_down_resp),
        httpx.Response(200, json=_OK_RESPONSE),
    ]

    ctrl, _, events = _make_controller(sleep=recording_sleep)
    ctrl.start_pairing("Test")
    await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.SUCCESS
    # After slow_down with interval=8 the sleep delay should be ≥ 8 - 0.5
    assert all(d >= 7.5 for d in intervals_used[1:]), f"unexpected delays: {intervals_used}"


@respx.mock
async def test_slow_down_never_decreases_interval() -> None:
    """slow_down with smaller interval than current must be ignored."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )

    # First slow_down bumps to 10, second tries to go back to 2
    poll_route = respx.post("https://sudri.ru/api/v1/auth/device/poll")
    poll_route.side_effect = [
        httpx.Response(200, json={"status": "slow_down", "interval": 10}),
        httpx.Response(200, json={"status": "slow_down", "interval": 2}),
        httpx.Response(200, json=_OK_RESPONSE),
    ]

    intervals_used: list[float] = []

    async def recording_sleep(delay: float) -> None:
        intervals_used.append(delay)
        await _instant_sleep(delay)

    ctrl, _, _ = _make_controller(sleep=recording_sleep)
    ctrl.start_pairing("Test")
    await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.SUCCESS
    # After second slow_down(2) the interval must remain ≥ 10 - 0.5
    assert all(d >= 9.5 for d in intervals_used[1:])


# ---------------------------------------------------------------------------
# Tests: terminal error states
# ---------------------------------------------------------------------------


@respx.mock
async def test_expired_token_from_bff() -> None:
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json={"status": "expired_token"})
    )

    ctrl, _, events = _make_controller()
    ctrl.start_pairing("Test")
    await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.ERROR_EXPIRED
    failed = next(e for e in events if isinstance(e, PairingFailed))
    assert failed.reason == PairingFailureReason.EXPIRED


@respx.mock
async def test_access_denied_from_bff() -> None:
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json={"status": "access_denied"})
    )

    ctrl, _, events = _make_controller()
    ctrl.start_pairing("Test")
    await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.ERROR_DENIED
    failed = next(e for e in events if isinstance(e, PairingFailed))
    assert failed.reason == PairingFailureReason.DENIED


# ---------------------------------------------------------------------------
# Tests: 429 does NOT terminate loop
# ---------------------------------------------------------------------------


@respx.mock
async def test_429_does_not_terminate_loop() -> None:
    """429 on poll should retry, not fail immediately."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    poll_route = respx.post("https://sudri.ru/api/v1/auth/device/poll")
    # Two 429s then success
    poll_route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(200, json=_OK_RESPONSE),
    ]

    ctrl, _, events = _make_controller()
    ctrl.start_pairing("Test")
    await _wait_for_terminal(ctrl)

    # Should still succeed
    assert ctrl.state == PairingState.SUCCESS


# ---------------------------------------------------------------------------
# Tests: network error → exp backoff → ERROR_NETWORK after MAX_NETWORK_RETRIES
# ---------------------------------------------------------------------------


@respx.mock
async def test_network_error_backoff_then_error_network() -> None:
    """MAX_NETWORK_RETRIES consecutive failures → ERROR_NETWORK."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )

    # All polls raise network error
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    ctrl, _, events = _make_controller()
    ctrl.start_pairing("Test")
    await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.ERROR_NETWORK
    failed = next(e for e in events if isinstance(e, PairingFailed))
    assert failed.reason == PairingFailureReason.NETWORK


# ---------------------------------------------------------------------------
# Tests: deadline (expires_at < now) controller detects itself
# ---------------------------------------------------------------------------


@respx.mock
async def test_deadline_detected_by_controller() -> None:
    """Controller emits ERROR_EXPIRED when expires_at is reached during poll."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    # start/poll route not needed — controller detects expiry before polling

    # Simulate time advancing past the deadline:
    # - _do_flow calls now() to compute expires_at = now() + 600s
    # - _poll_loop calls now() again to compute remaining
    # We advance clock by 700s on the second call so remaining < 0.
    call_count = 0

    def advancing_now() -> datetime:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: used to compute expires_at
            return _FIXED_NOW
        # Subsequent calls: already past expires_at (600s TTL)
        return _FIXED_NOW + timedelta(seconds=700)

    ctrl, _, events = _make_controller(now=advancing_now)
    ctrl.start_pairing("Test")
    await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.ERROR_EXPIRED
    failed = next(e for e in events if isinstance(e, PairingFailed))
    assert failed.reason == PairingFailureReason.EXPIRED


# ---------------------------------------------------------------------------
# Tests: cancel in various states
# ---------------------------------------------------------------------------


@respx.mock
async def test_cancel_during_polling_no_warnings() -> None:
    """Cancel during poll loop; task finishes without RuntimeWarning."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_PENDING_RESPONSE)
    )

    sleep_calls = 0

    bridge = PairingBridge()
    evts: list[Any] = []
    bridge.event_received.connect(evts.append)

    # We need ctrl in the closure, so define sleep after creating ctrl
    ctrl2: PairingController

    async def cancel_on_second_sleep(delay: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            ctrl2.cancel_pairing()
        await asyncio.sleep(0)

    ctrl2 = PairingController(
        bridge=bridge,
        keychain=FakeKeychain(),
        identity_factory=_fake_identity_factory,
        now=lambda: _FIXED_NOW,
        sleep=cancel_on_second_sleep,
    )

    ctrl2.start_pairing("Test")
    await _wait_for_terminal(ctrl2)

    # Should end in UNPAIRED (CANCELLED → UNPAIRED)
    assert ctrl2.state == PairingState.UNPAIRED


@respx.mock
async def test_cancel_in_starting_state() -> None:
    """Cancel while task is in STARTING (before AWAITING_USER)."""
    # We inject a sleep that triggers cancel to ensure cancellation happens
    # within the flow (after the first await in _do_flow).
    start_blocked = asyncio.Event()

    async def slow_start(req: httpx.Request) -> httpx.Response:
        # Signal that we've entered the network call, then yield
        start_blocked.set()
        await asyncio.sleep(0)  # yield so cancel can propagate
        return httpx.Response(200, json=_START_RESPONSE)

    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(side_effect=slow_start)

    ctrl, _, events = _make_controller()

    ctrl.start_pairing("Test")

    # Wait until the HTTP call is in progress (task is awaiting)
    await asyncio.wait_for(start_blocked.wait(), timeout=1.0)

    # Now cancel
    ctrl.cancel_pairing()

    task = ctrl._task
    assert task is not None

    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)

    assert ctrl.state in (PairingState.UNPAIRED, PairingState.CANCELLED)


# ---------------------------------------------------------------------------
# Tests: keychain unavailable at store_private_key → ERROR_KEYCHAIN, BFF not called
# ---------------------------------------------------------------------------


@respx.mock
async def test_keychain_fail_on_privkey_no_bff_call() -> None:
    """If keychain fails on store_private_key, BFF start is never called."""
    start_route = respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )

    keychain = FakeKeychain(fail_on_store_privkey=True)
    ctrl, _, events = _make_controller(keychain=keychain)
    ctrl.start_pairing("Test")
    await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.ERROR_KEYCHAIN
    failed = next(e for e in events if isinstance(e, PairingFailed))
    assert failed.reason == PairingFailureReason.KEYCHAIN_UNAVAILABLE

    # BFF should NOT have been called
    assert start_route.call_count == 0


# ---------------------------------------------------------------------------
# Tests: keychain unavailable at store_device_token → ERROR_KEYCHAIN
# ---------------------------------------------------------------------------


@respx.mock
async def test_keychain_fail_on_device_token(caplog: pytest.LogCaptureFixture) -> None:
    """If keychain fails on store_device_token, ERROR_KEYCHAIN; token not in logs."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    keychain = FakeKeychain(fail_on_store_token=True)
    ctrl, _, events = _make_controller(keychain=keychain)

    with caplog.at_level(logging.DEBUG, logger="agent"):
        ctrl.start_pairing("Test")
        await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.ERROR_KEYCHAIN
    failed = next(e for e in events if isinstance(e, PairingFailed))
    assert failed.reason == PairingFailureReason.KEYCHAIN_UNAVAILABLE

    # Token must not appear in logs
    for record in caplog.records:
        assert "tok_secret_abc123" not in record.getMessage()


# ---------------------------------------------------------------------------
# Tests: deny-list — no sensitive values in logs after full flow
# ---------------------------------------------------------------------------


@respx.mock
async def test_deny_list_no_sensitive_in_logs(
    caplog: pytest.LogCaptureFixture, tmp_path: Any
) -> None:
    """Full successful flow: caplog contains no device_token, device_code, private_key."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    ctrl, _, _ = _make_controller()

    with (
        caplog.at_level(logging.DEBUG),
        patch("agent.auth.pairing._state_json_path", return_value=tmp_path / "state.json"),
    ):
        ctrl.start_pairing("Test")
        await _wait_for_terminal(ctrl)

    sensitive_tokens = [
        "tok_secret_abc123",   # device_token value
        "FAKE_DEVICE_CODE_AAAA",  # device_code value
    ]
    for record in caplog.records:
        msg = record.getMessage()
        for token in sensitive_tokens:
            assert token not in msg, (
                f"Sensitive value '{token}' found in log record: {msg!r}"
            )

    # private_key bytes should never appear
    import binascii
    privkey_hex = binascii.hexlify(_FAKE_PRIVKEY).decode()
    for record in caplog.records:
        assert privkey_hex not in record.getMessage()


# ---------------------------------------------------------------------------
# Tests: PairingSucceeded.fingerprint_raw is exactly 12 chars
# ---------------------------------------------------------------------------


@respx.mock
async def test_pairing_succeeded_fingerprint_raw_12_chars(tmp_path: Any) -> None:
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    ctrl, _, events = _make_controller()
    with patch("agent.auth.pairing._state_json_path", return_value=tmp_path / "state.json"):
        ctrl.start_pairing("Test")
        await _wait_for_terminal(ctrl)

    succeeded = next(e for e in events if isinstance(e, PairingSucceeded))
    assert len(succeeded.fingerprint_raw) == 12


# ---------------------------------------------------------------------------
# Tests: state.json created on SUCCESS
# ---------------------------------------------------------------------------


@respx.mock
async def test_state_json_created_on_success(tmp_path: Any) -> None:
    import json as _json

    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    state_path = tmp_path / "state.json"
    ctrl, _, _ = _make_controller()

    with patch("agent.auth.pairing._state_json_path", return_value=state_path):
        ctrl.start_pairing("My Device")
        await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.SUCCESS
    assert state_path.exists()

    data = _json.loads(state_path.read_text())
    assert "paired_at" in data
    assert "T" in data["paired_at"]  # ISO-8601 check
    assert data["fingerprint_raw"] == _EXPECTED_FINGERPRINT
    assert data["peer_id"] == _EXPECTED_PEER_ID


# ---------------------------------------------------------------------------
# Tests: idempotent start_pairing
# ---------------------------------------------------------------------------


@respx.mock
async def test_start_pairing_idempotent_in_starting() -> None:
    """Double-click during STARTING must be no-op."""
    start_called = 0
    original_resp = httpx.Response(200, json=_START_RESPONSE)

    def counting_start(req: httpx.Request) -> httpx.Response:
        nonlocal start_called
        start_called += 1
        return original_resp

    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(side_effect=counting_start)
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    ctrl, _, _ = _make_controller()
    ctrl.start_pairing("Test")
    ctrl.start_pairing("Test")  # second call — idempotent
    ctrl.start_pairing("Test")  # third call — idempotent

    await _wait_for_terminal(ctrl)

    # Only one BFF /start call
    assert start_called == 1


@respx.mock
async def test_start_pairing_noop_in_success() -> None:
    """start_pairing in SUCCESS state must be completely ignored."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    ctrl, _, _ = _make_controller()

    with patch("agent.auth.pairing._state_json_path", return_value="/tmp/state_noop.json"):
        ctrl.start_pairing("Test")
        await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.SUCCESS
    # Calling start_pairing again should not change state
    ctrl.start_pairing("Test again")
    assert ctrl.state == PairingState.SUCCESS


# ---------------------------------------------------------------------------
# Tests: retry from error state
# ---------------------------------------------------------------------------


@respx.mock
async def test_retry_from_error_state(tmp_path: Any) -> None:
    """retry_pairing() after ERROR_NETWORK restarts the flow."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        side_effect=[
            httpx.ConnectError("no connection"),  # first attempt fails
            httpx.Response(200, json=_START_RESPONSE),  # retry succeeds
        ]
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    ctrl, _, events = _make_controller()
    ctrl.start_pairing("Test")
    await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.ERROR_NETWORK

    with patch("agent.auth.pairing._state_json_path", return_value=tmp_path / "state.json"):
        ctrl.retry_pairing()
        await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.SUCCESS


# ---------------------------------------------------------------------------
# Tests: _default_label_sanitizer
# ---------------------------------------------------------------------------


def test_label_sanitizer_nfc_and_truncate() -> None:
    # NFC normalization: é (e + combining acute) → single char
    import unicodedata
    composed_e = unicodedata.normalize("NFC", "é")  # é
    result = _default_label_sanitizer(composed_e)
    assert result == composed_e

    # Truncate to 64 codepoints
    long_label = "A" * 100
    assert len(_default_label_sanitizer(long_label)) == 64


def test_label_sanitizer_strips_control_chars() -> None:
    label_with_control = "My\x00Device\x01\x02"
    result = _default_label_sanitizer(label_with_control)
    assert "\x00" not in result
    assert result == "MyDevice"


def test_label_sanitizer_fallback_on_empty() -> None:
    assert _default_label_sanitizer("") == "Устройство"
    assert _default_label_sanitizer("   ") == "Устройство"
    assert _default_label_sanitizer("\x00\x01\x02") == "Устройство"


# ---------------------------------------------------------------------------
# Tests: _format_user_code
# ---------------------------------------------------------------------------


def test_format_user_code_adds_dash() -> None:
    assert _format_user_code("ABCDEFGH") == "ABCD-EFGH"
    assert _format_user_code("abcdefgh") == "ABCD-EFGH"


def test_format_user_code_already_dashed() -> None:
    assert _format_user_code("ABCD-EFGH") == "ABCD-EFGH"


# ---------------------------------------------------------------------------
# Tests: cancel in AWAITING_USER / POLLING — Task finishes cleanly
# ---------------------------------------------------------------------------


@respx.mock
async def test_cancel_no_pending_tasks() -> None:
    """After cancel, no lingering tasks (no RuntimeWarning)."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_PENDING_RESPONSE)
    )

    cancelled_at_attempt: list[int] = []

    bridge = PairingBridge()
    events: list[Any] = []
    bridge.event_received.connect(events.append)

    ctrl: PairingController | None = None

    async def cancel_on_first_polling(delay: float) -> None:
        assert ctrl is not None
        # Cancel right after first sleep (before any poll attempt)
        if not cancelled_at_attempt:
            cancelled_at_attempt.append(1)
            ctrl.cancel_pairing()
        await asyncio.sleep(0)

    ctrl = PairingController(
        bridge=bridge,
        keychain=FakeKeychain(),
        identity_factory=_fake_identity_factory,
        now=lambda: _FIXED_NOW,
        sleep=cancel_on_first_polling,
    )

    ctrl.start_pairing("Test")
    task = ctrl._task
    assert task is not None

    # Wait for task to complete (cancelled)
    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)

    # Task must be done
    assert task.done()

    # No remaining pairing_flow tasks
    remaining = [t for t in asyncio.all_tasks() if t.get_name() == "pairing_flow"]
    assert len(remaining) == 0
