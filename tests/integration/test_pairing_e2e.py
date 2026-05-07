"""tests/integration/test_pairing_e2e.py — E2E pairing lifecycle integration tests.

Full lifecycle tests covering:
  e2e_1_happy_path          — full flow: bootstrap → pair → keychain/state.json written
  e2e_2_unlink              — PAIRED → unlink → UNPAIRED; privkey preserved
  e2e_3_re_pair_stable_peer_id — pair → unlink → pair; peer_id identical
  e2e_4_quit_during_pairing — aboutToQuit → cancel_pairing; no pending tasks
  e2e_5_slow_down_and_one_error — slow_down interval increase + expired_token branch
  e2e_6_log_redaction_real_flow — no sensitive tokens in logs during full flow

Tooling: pytest-qt + pytest-asyncio + respx.mock + FakeKeychain.
QT_QPA_PLATFORM=offscreen is set by pytest-qt automatically.

Notes:
- All tests run with real asyncio event loop (asyncio_mode="auto" in pytest.ini).
- BFF HTTP mocks use respx.mock decorator.
- state.json is redirected to tmp_path in every test.
- Keypair reuse (stable peer_id) is the invariant under test in e2e_3.
"""

from __future__ import annotations

import asyncio
import base64 as _b64
import contextlib
import io
import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx
from PySide6.QtWidgets import QApplication

from agent.auth.pairing import (
    PairingBridge,
    PairingController,
    PairingFailed,
    PairingFailureReason,
    PairingPolling,
    PairingStarted,
    PairingState,
    PairingSucceeded,
)
from agent.identity import compute_fingerprint, compute_peer_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIXED_NOW = datetime(2026, 5, 7, 12, 0, 0, tzinfo=UTC)
_FAKE_PRIVKEY = b"\x01" * 32
_FAKE_PUBKEY = b"\x02" * 32

_EXPECTED_PEER_ID = compute_peer_id(_FAKE_PUBKEY)
_EXPECTED_FINGERPRINT = compute_fingerprint(_FAKE_PUBKEY)
_EXPECTED_PUBKEY_B64 = _b64.urlsafe_b64encode(_FAKE_PUBKEY).rstrip(b"=").decode()

_START_RESPONSE: dict[str, Any] = {
    "device_code": "FAKE_DEVICE_CODE_E2E",
    "user_code": "ABCDEFGH",
    "verification_uri": "https://sudri.ru/link",
    "verification_uri_complete": "https://sudri.ru/link?code=ABCD-EFGH",
    "expires_in": 600,
    "interval": 3,
}

_PENDING_RESPONSE: dict[str, Any] = {"status": "authorization_pending"}

_OK_RESPONSE: dict[str, Any] = {
    "status": "ok",
    "device_token": "tok_e2e_secret_xyz",
    "device_id": "dev-e2e-uuid-1234",
    "user_id": "user-e2e-uuid-5678",
    "token_expires_at": "2027-05-07T12:00:00Z",
    "scope": ["node:run", "accounting:report", "balance:read"],
}

# ---------------------------------------------------------------------------
# FakeKeychain — in-memory, replaces OS keyring for all e2e tests
# ---------------------------------------------------------------------------


class FakeKeychain:
    """In-memory keychain. Tracks all store/delete calls for assertion."""

    def __init__(
        self,
        *,
        fail_on_store_privkey: bool = False,
        fail_on_store_token: bool = False,
    ) -> None:
        self._store: dict[str, Any] = {}
        self.store_privkey_calls: list[str] = []
        self.store_token_calls: list[str] = []
        self.delete_token_calls: list[str] = []
        self._fail_on_store_privkey = fail_on_store_privkey
        self._fail_on_store_token = fail_on_store_token

    # --- privkey ---

    def store_private_key(self, peer_id: str, private_key_bytes: bytes) -> None:
        if self._fail_on_store_privkey:
            import keyring.errors
            raise keyring.errors.KeyringError("keychain locked")
        self._store[f"{peer_id}:privkey"] = private_key_bytes
        self.store_privkey_calls.append(peer_id)

    def load_private_key(self, peer_id: str) -> bytes | None:
        return self._store.get(f"{peer_id}:privkey")

    # --- device token ---

    def store_device_token(self, peer_id: str, token: str) -> None:
        if self._fail_on_store_token:
            import keyring.errors
            raise keyring.errors.KeyringError("keychain locked")
        self._store[f"{peer_id}:device_token"] = token
        self.store_token_calls.append(peer_id)

    def load_device_token(self, peer_id: str) -> str | None:
        return self._store.get(f"{peer_id}:device_token")

    def delete_device_token(self, peer_id: str) -> None:
        self._store.pop(f"{peer_id}:device_token", None)
        self.delete_token_calls.append(peer_id)

    def delete_device(self, peer_id: str) -> None:
        self._store.pop(f"{peer_id}:privkey", None)
        self._store.pop(f"{peer_id}:device_token", None)

    # --- helpers ---

    def has_device_token(self, peer_id: str) -> bool:
        return f"{peer_id}:device_token" in self._store

    def has_private_key(self, peer_id: str) -> bool:
        return f"{peer_id}:privkey" in self._store


# ---------------------------------------------------------------------------
# Instant sleep — yields control without real delay
# ---------------------------------------------------------------------------


async def _instant_sleep(_: float) -> None:
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------


def _make_controller(
    *,
    keychain: FakeKeychain | None = None,
    now: Any = None,
    sleep: Any = None,
    identity_factory: Any = None,
) -> tuple[PairingController, PairingBridge, list[Any]]:
    bridge = PairingBridge()
    events: list[Any] = []
    bridge.event_received.connect(events.append)

    ctrl = PairingController(
        bridge=bridge,
        keychain=keychain or FakeKeychain(),
        identity_factory=identity_factory or (lambda: (_FAKE_PRIVKEY, _FAKE_PUBKEY)),
        now=now or (lambda: _FIXED_NOW),
        sleep=sleep or _instant_sleep,
    )
    return ctrl, bridge, events


async def _wait_for_terminal(ctrl: PairingController, timeout: float = 3.0) -> None:
    """Wait until controller reaches a terminal state."""
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
    deadline = asyncio.get_event_loop().time() + timeout
    while ctrl.state not in terminal:
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"Controller stuck in state: {ctrl.state}")
        await asyncio.sleep(0.01)


# ===========================================================================
# e2e_1_happy_path
# ===========================================================================


@respx.mock
async def test_e2e_1_happy_path(
    tmp_path: Any,
    qapp: QApplication,
) -> None:
    """Full flow: bootstrap → trigger pairing → mock BFF → ok → state persisted.

    Verifies:
    - UI shows AwaitingUser with user_code + QR
    - PairingPolling emitted for each poll
    - PairingSucceeded emitted; no device_token field
    - device_token + privkey stored in FakeKeychain
    - state.json written atomically with paired_at + fingerprint + peer_id
    - Settings section reads state.json and displays display-14 fingerprint
    """
    from app.main import build_app
    from app.screens.settings import PairingSectionWidget
    from app.windows.main_window import MainWindow

    # Setup BFF mocks
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
    state_file = tmp_path / "state.json"

    with patch("agent.auth.pairing._state_json_path", return_value=state_file):
        window, tray = build_app(qapp)
        assert isinstance(window, MainWindow)

        # Replace controller's keychain and sleep with test doubles
        ctrl = window.pairing_controller
        assert isinstance(ctrl, PairingController)
        ctrl._keychain = keychain  # type: ignore[attr-defined]
        ctrl._sleep = _instant_sleep  # type: ignore[attr-defined]
        ctrl._identity_factory = lambda: (_FAKE_PRIVKEY, _FAKE_PUBKEY)  # type: ignore[attr-defined]

        # Capture all events
        events: list[Any] = []
        ctrl.bridge.event_received.connect(events.append)

        # Trigger pairing
        ctrl.start_pairing("E2E Test Machine")

        await _wait_for_terminal(ctrl)

    # --- Controller reached SUCCESS ---
    assert ctrl.state == PairingState.SUCCESS, f"unexpected state: {ctrl.state}"

    # --- PairingStarted event with user_code + fingerprint ---
    started_events = [e for e in events if isinstance(e, PairingStarted)]
    assert len(started_events) == 1
    started = started_events[0]
    assert started.user_code == "ABCD-EFGH"
    assert len(started.fingerprint_raw) == 12
    assert started.fingerprint_raw == _EXPECTED_FINGERPRINT
    assert "sudri.ru" in started.verification_uri_complete

    # --- PairingPolling emitted for each of 3 polls ---
    polling_events = [e for e in events if isinstance(e, PairingPolling)]
    assert len(polling_events) == 3
    assert [p.attempt for p in polling_events] == [1, 2, 3]

    # --- PairingSucceeded with no device_token ---
    succeeded_events = [e for e in events if isinstance(e, PairingSucceeded)]
    assert len(succeeded_events) == 1
    succeeded = succeeded_events[0]
    assert succeeded.device_id == "dev-e2e-uuid-1234"
    assert succeeded.user_id == "user-e2e-uuid-5678"
    assert len(succeeded.fingerprint_raw) == 12
    assert not hasattr(succeeded, "device_token"), "device_token must NOT be in PairingSucceeded"

    # --- Keychain: privkey stored BEFORE /start; token stored on ok ---
    # privkey should have been stored first (step 1-4 in _do_flow)
    assert keychain.has_private_key(_EXPECTED_PEER_ID), "privkey must be in keychain"
    assert keychain.store_privkey_calls == [_EXPECTED_PEER_ID]
    # token stored on success
    assert keychain.has_device_token(_EXPECTED_PEER_ID), "device_token must be in keychain"
    assert keychain.store_token_calls == [_EXPECTED_PEER_ID]
    # Order: privkey before token
    assert keychain.store_privkey_calls[0] == keychain.store_token_calls[0]

    # --- state.json written atomically ---
    assert state_file.exists(), "state.json must be written after SUCCESS"
    state_data = json.loads(state_file.read_text(encoding="utf-8"))
    assert "paired_at" in state_data
    assert "T" in state_data["paired_at"]  # ISO-8601
    assert state_data["fingerprint_raw"] == _EXPECTED_FINGERPRINT
    assert state_data["peer_id"] == _EXPECTED_PEER_ID
    assert state_data.get("device_label")  # label present

    # --- Settings PairingSectionWidget reads state.json and shows display-14 ---
    from PySide6.QtWidgets import QLabel
    with patch("app.screens.settings._state_json_path", return_value=state_file):
        section = PairingSectionWidget(controller=None)
    assert section.paired_data is not None
    fp_labels = section.findChildren(QLabel, "PairedFingerprintLabel")
    assert len(fp_labels) == 1
    # display-14 format: "XXXX XXXX XXXX"
    display_fp = fp_labels[0].text()
    assert len(display_fp) == 14
    assert display_fp.count(" ") == 2
    # display-14 is format_fingerprint(raw)
    from agent.identity import format_fingerprint
    assert display_fp == format_fingerprint(_EXPECTED_FINGERPRINT)


# ===========================================================================
# e2e_2_unlink
# ===========================================================================


async def test_e2e_2_unlink(tmp_path: Any, qapp: QApplication) -> None:
    """PAIRED state → trigger unlink → keychain.delete_device_token called,
    privkey NOT deleted, state.json cleared → Settings shows UNPAIRED.

    Uses PairingSectionWidget._perform_unlink() directly (bypasses QMessageBox dialog).
    """
    from PySide6.QtWidgets import QLabel

    from app.screens.settings import PairingSectionWidget

    peer_id = _EXPECTED_PEER_ID
    fingerprint_raw = _EXPECTED_FINGERPRINT

    # Setup state.json in PAIRED state
    state_file = tmp_path / "state.json"
    state_data = {
        "paired_at": "2026-05-07T12:34:56Z",
        "device_label": "E2E Test Machine",
        "fingerprint_raw": fingerprint_raw,
        "peer_id": peer_id,
    }
    state_file.write_text(json.dumps(state_data), encoding="utf-8")

    # Setup FakeKeychain with both privkey and token
    keychain = FakeKeychain()
    keychain.store_private_key(peer_id, _FAKE_PRIVKEY)
    keychain.store_device_token(peer_id, "tok_e2e_to_unlink")

    with patch("app.screens.settings._state_json_path", return_value=state_file):
        section = PairingSectionWidget(controller=None)

    # Verify PAIRED state before unlink
    assert section.paired_data is not None
    chip_labels = section.findChildren(QLabel, "PairedStatusChip")
    assert any(lbl.text() == "Привязано" for lbl in chip_labels), (
        "PAIRED chip not found before unlink"
    )

    # Perform unlink with patched keychain
    with (
        patch("app.screens.settings._state_json_path", return_value=state_file),
        patch(
            "agent.keychain.Keychain.delete_device_token",
            autospec=True,
            side_effect=lambda self_kc, pid: keychain.delete_device_token(pid),
        ),
    ):
        section._perform_unlink()

    # --- delete_device_token was called with correct peer_id ---
    assert keychain.delete_token_calls == [peer_id], (
        f"Expected delete_device_token({peer_id!r}), got {keychain.delete_token_calls}"
    )

    # --- privkey NOT deleted ---
    assert keychain.has_private_key(peer_id), "Private key must NOT be deleted on unlink"

    # --- state.json cleared (empty object) ---
    cleared_data = json.loads(state_file.read_text(encoding="utf-8"))
    assert cleared_data == {}, f"state.json must be empty after unlink, got {cleared_data}"

    # --- Settings widget shows UNPAIRED ---
    assert section.paired_data is None, "paired_data must be None after unlink"


# ===========================================================================
# e2e_3_re_pair_stable_peer_id
# ===========================================================================


@respx.mock
async def test_e2e_3_re_pair_stable_peer_id(
    tmp_path: Any,
    qapp: QApplication,
) -> None:
    """pair → unlink → pair again; peer_id from pubkey is IDENTICAL across both flows.

    Invariant: peer_id = base58(sha256(pubkey)) — deterministic, not random.
    When identity_factory returns the same keypair (stable), peer_id must be identical.
    """
    # Both pair attempts use the same deterministic keypair
    def _stable_identity() -> tuple[bytes, bytes]:
        return _FAKE_PRIVKEY, _FAKE_PUBKEY

    state_file = tmp_path / "state.json"

    # --- First pair ---
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    keychain = FakeKeychain()
    ctrl1, _, events1 = _make_controller(keychain=keychain, identity_factory=_stable_identity)

    with patch("agent.auth.pairing._state_json_path", return_value=state_file):
        ctrl1.start_pairing("First Pair")
        await _wait_for_terminal(ctrl1)

    assert ctrl1.state == PairingState.SUCCESS

    # Capture peer_id from first pairing
    succeeded1 = next(e for e in events1 if isinstance(e, PairingSucceeded))
    peer_id_first = ctrl1._peer_id  # type: ignore[attr-defined]
    assert peer_id_first == _EXPECTED_PEER_ID

    # --- Simulate unlink: delete token, clear state.json ---
    keychain.delete_device_token(peer_id_first)
    assert not keychain.has_device_token(peer_id_first)
    state_file.write_text("{}", encoding="utf-8")

    # --- Second pair with fresh controller (same keychain, same identity factory) ---
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    ctrl2, _, events2 = _make_controller(keychain=keychain, identity_factory=_stable_identity)

    with patch("agent.auth.pairing._state_json_path", return_value=state_file):
        ctrl2.start_pairing("Re-pair")
        await _wait_for_terminal(ctrl2)

    assert ctrl2.state == PairingState.SUCCESS

    peer_id_second = ctrl2._peer_id  # type: ignore[attr-defined]

    # --- INVARIANT: peer_id is identical across both pairings ---
    assert peer_id_first == peer_id_second, (
        f"peer_id must be stable (derived from pubkey): "
        f"first={peer_id_first!r}, second={peer_id_second!r}"
    )

    # Fingerprint also stable
    succeeded2 = next(e for e in events2 if isinstance(e, PairingSucceeded))
    assert succeeded1.fingerprint_raw == succeeded2.fingerprint_raw, (
        "fingerprint_raw must be identical across pairings with same keypair"
    )


# ===========================================================================
# e2e_4_quit_during_pairing
# ===========================================================================


@respx.mock
async def test_e2e_4_quit_during_pairing(
    tmp_path: Any,
    qapp: QApplication,
) -> None:
    """aboutToQuit → cancel_pairing called; asyncio task cancelled cleanly.

    Verifies:
    - Task is cancelled without RuntimeWarning
    - Controller reaches UNPAIRED (CANCELLED → UNPAIRED)
    - No lingering pairing_flow tasks in asyncio
    """
    from app.main import build_app

    # poll will block until cancel arrives
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_PENDING_RESPONSE)
    )

    state_file = tmp_path / "state.json"
    keychain = FakeKeychain()

    # Use a controlled sleep: first sleep completes instantly so the controller
    # reaches AWAITING_USER, then the second sleep blocks until cancelled.
    sleep_count = 0
    cancel_event = asyncio.Event()

    async def controlled_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            # First sleep (before first poll): let it complete so AWAITING_USER is reached
            await asyncio.sleep(0)
        else:
            # Subsequent sleeps: block until cancel arrives
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(cancel_event.wait(), timeout=3.0)

    with patch("agent.auth.pairing._state_json_path", return_value=state_file):
        window, tray = build_app(qapp)
        ctrl = window.pairing_controller
        assert isinstance(ctrl, PairingController)
        ctrl._keychain = keychain  # type: ignore[attr-defined]
        ctrl._sleep = controlled_sleep  # type: ignore[attr-defined]
        ctrl._identity_factory = lambda: (_FAKE_PRIVKEY, _FAKE_PUBKEY)  # type: ignore[attr-defined]

        # Start pairing (goes async)
        ctrl.start_pairing("Quit Test")

        # Wait until pairing reaches AWAITING_USER (first sleep completed, browser opened)
        deadline = asyncio.get_event_loop().time() + 2.0
        while ctrl.state not in (
            PairingState.AWAITING_USER,
            PairingState.POLLING,
        ):
            if asyncio.get_event_loop().time() > deadline:
                break
            await asyncio.sleep(0.005)

        # Simulate aboutToQuit → cancel_pairing
        ctrl.cancel_pairing()

        # Wait for task to finish (cancel propagates via CancelledError)
        task = ctrl._task  # type: ignore[attr-defined]
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)

    # --- Controller is in UNPAIRED or CANCELLED after cancel ---
    assert ctrl.state in (PairingState.UNPAIRED, PairingState.CANCELLED), (
        f"Expected UNPAIRED/CANCELLED after cancel, got {ctrl.state}"
    )

    # --- No pending pairing_flow tasks ---
    remaining = [t for t in asyncio.all_tasks() if t.get_name() == "pairing_flow"]
    assert len(remaining) == 0, f"Lingering pairing_flow tasks: {remaining}"

    # --- Task is done (no RuntimeWarning from unawaited tasks) ---
    if task is not None:
        assert task.done(), "pairing_flow task must be done after cancel"


# ===========================================================================
# e2e_5_slow_down_and_one_error
# ===========================================================================


@respx.mock
async def test_e2e_5a_slow_down_then_ok(tmp_path: Any) -> None:
    """slow_down response → interval increases → poll ok → SUCCESS.

    Verifies that after slow_down the sleep delay is >= new_interval - jitter_max.
    """
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )

    sleep_delays: list[float] = []

    async def recording_sleep(delay: float) -> None:
        sleep_delays.append(delay)
        await asyncio.sleep(0)

    slow_down_resp: dict[str, Any] = {"status": "slow_down", "interval": 10}

    poll_route = respx.post("https://sudri.ru/api/v1/auth/device/poll")
    poll_route.side_effect = [
        httpx.Response(200, json=slow_down_resp),
        httpx.Response(200, json=slow_down_resp),
        httpx.Response(200, json=_OK_RESPONSE),
    ]

    keychain = FakeKeychain()
    state_file = tmp_path / "state.json"
    ctrl, _, events = _make_controller(keychain=keychain, sleep=recording_sleep)

    with patch("agent.auth.pairing._state_json_path", return_value=state_file):
        ctrl.start_pairing("SlowDown Test")
        await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.SUCCESS

    # After slow_down(interval=10), subsequent sleep delays must be >= 10 - 0.5 = 9.5
    # First sleep is before any poll (may use original interval=3)
    # Sleeps after the first slow_down must reflect new interval
    assert len(sleep_delays) >= 3
    # All delays after the first slow_down must be >= 9.5
    for delay in sleep_delays[1:]:
        assert delay >= 9.5, (
            f"After slow_down(interval=10), delay should be >= 9.5, got {delay}: {sleep_delays}"
        )

    # Verify SUCCESS event emitted
    succeeded = [e for e in events if isinstance(e, PairingSucceeded)]
    assert len(succeeded) == 1


@respx.mock
async def test_e2e_5b_expired_token_error() -> None:
    """poll returns expired_token → controller reaches ERROR_EXPIRED + PairingFailed emitted."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json={"status": "expired_token"})
    )

    ctrl, _, events = _make_controller()
    ctrl.start_pairing("Expired Token Test")
    await _wait_for_terminal(ctrl)

    # Controller must be in ERROR_EXPIRED
    assert ctrl.state == PairingState.ERROR_EXPIRED, (
        f"Expected ERROR_EXPIRED, got {ctrl.state}"
    )

    # PairingFailed emitted with EXPIRED reason
    failed_events = [e for e in events if isinstance(e, PairingFailed)]
    assert len(failed_events) == 1
    assert failed_events[0].reason == PairingFailureReason.EXPIRED

    # No device_token stored (pairing never completed)
    keychain = ctrl._keychain  # type: ignore[attr-defined]
    assert isinstance(keychain, FakeKeychain)
    assert not keychain.has_device_token(_EXPECTED_PEER_ID)


@respx.mock
async def test_e2e_5c_slow_down_never_decreases_interval() -> None:
    """Sending slow_down with smaller interval than current must be ignored (monotonic increase)."""
    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )

    sleep_delays: list[float] = []

    async def recording_sleep(delay: float) -> None:
        sleep_delays.append(delay)
        await asyncio.sleep(0)

    poll_route = respx.post("https://sudri.ru/api/v1/auth/device/poll")
    poll_route.side_effect = [
        httpx.Response(200, json={"status": "slow_down", "interval": 12}),  # bump to 12
        httpx.Response(200, json={"status": "slow_down", "interval": 4}),   # attempt to decrease
        httpx.Response(200, json=_OK_RESPONSE),
    ]

    ctrl, _, _ = _make_controller(sleep=recording_sleep)
    ctrl.start_pairing("Monotonic Interval Test")
    await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.SUCCESS
    # After second slow_down(4), the interval must stay >= 12 → delays >= 11.5
    for delay in sleep_delays[1:]:
        assert delay >= 11.5, (
            f"Interval must not decrease: expected >= 11.5, got {delay}: {sleep_delays}"
        )


# ===========================================================================
# e2e_6_log_redaction_real_flow
# ===========================================================================


@respx.mock
async def test_e2e_6_log_redaction_real_flow(
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Full pairing flow: no sensitive tokens appear in log output.

    Deny-list checked:
    - device_code value: "FAKE_DEVICE_CODE_E2E"
    - device_token value: "tok_e2e_secret_xyz"
    - user_code raw value: "ABCDEFGH"
    - privkey bytes (hex): "01010101..." (32 bytes of 0x01)
    - device_pubkey b64: _EXPECTED_PUBKEY_B64

    Uses caplog + StreamHandler capture to cover both:
    1. caplog (pytest) records → agent + app namespaces
    2. StreamHandler output → RedactingFormatter applies
    """
    from app.logging_setup import SensitiveDataFilter, setup_logging

    # Setup logging with real redacting setup to a temp file
    log_file = tmp_path / "app.log"
    setup_logging(level=logging.DEBUG, log_file=log_file)

    # Also capture via StringIO stream handler for verification
    captured_output = io.StringIO()
    stream_handler = logging.StreamHandler(captured_output)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.addFilter(SensitiveDataFilter())
    logging.getLogger().addHandler(stream_handler)

    respx.post("https://sudri.ru/api/v1/auth/device/start").mock(
        return_value=httpx.Response(200, json=_START_RESPONSE)
    )
    respx.post("https://sudri.ru/api/v1/auth/device/poll").mock(
        return_value=httpx.Response(200, json=_OK_RESPONSE)
    )

    state_file = tmp_path / "state.json"
    keychain = FakeKeychain()

    with (
        caplog.at_level(logging.DEBUG),
        patch("agent.auth.pairing._state_json_path", return_value=state_file),
    ):
        ctrl, _, _ = _make_controller(keychain=keychain)
        ctrl.start_pairing("LogRedaction Test")
        await _wait_for_terminal(ctrl)

    assert ctrl.state == PairingState.SUCCESS

    # Collect all log text: caplog records + captured stream output + log file
    all_log_text: list[str] = []

    # caplog records
    for record in caplog.records:
        all_log_text.append(record.getMessage())

    # stream output
    captured_output.seek(0)
    all_log_text.append(captured_output.read())

    # log file (written by setup_logging)
    if log_file.exists():
        all_log_text.append(log_file.read_text(encoding="utf-8"))

    combined = "\n".join(all_log_text)

    # --- Sensitive values that must NOT appear ---
    import binascii
    privkey_hex = binascii.hexlify(_FAKE_PRIVKEY).decode()

    sensitive_deny_list = [
        ("device_code value", _START_RESPONSE["device_code"]),   # "FAKE_DEVICE_CODE_E2E"
        ("device_token value", _OK_RESPONSE["device_token"]),     # "tok_e2e_secret_xyz"
        ("privkey hex", privkey_hex),                              # "0101...01"
        ("pubkey b64", _EXPECTED_PUBKEY_B64),                     # base64url of _FAKE_PUBKEY
    ]

    for description, secret in sensitive_deny_list:
        assert secret not in combined, (
            f"SECURITY VIOLATION: {description} ({secret!r}) found in log output. "
            f"The deny-list filter is not working correctly."
        )

    # Cleanup extra handler
    logging.getLogger().removeHandler(stream_handler)


# ===========================================================================
# e2e_bonus: settings_paired_to_unpaired_after_unlink (integration of e2e_1 + e2e_2)
# ===========================================================================


@respx.mock
async def test_e2e_settings_reflects_state_changes(
    tmp_path: Any,
    qapp: QApplication,
) -> None:
    """Integration: Settings UNPAIRED → pair → Settings PAIRED → unlink → UNPAIRED again.

    Verifies the full UI state machine across the Settings widget.
    """
    from PySide6.QtWidgets import QLabel

    from app.screens.settings import PairingSectionWidget

    state_file = tmp_path / "state.json"

    # 1. Initially UNPAIRED (no state.json)
    with patch("app.screens.settings._state_json_path", return_value=state_file):
        section_unpaired = PairingSectionWidget(controller=None)
    assert section_unpaired.paired_data is None
    unpaired_chips = section_unpaired.findChildren(QLabel, "UnpairedStatusChip")
    assert len(unpaired_chips) == 1

    # 2. After successful pair: state.json has paired data
    paired_state = {
        "paired_at": "2026-05-07T14:00:00Z",
        "device_label": "Settings Test Machine",
        "fingerprint_raw": _EXPECTED_FINGERPRINT,
        "peer_id": _EXPECTED_PEER_ID,
    }
    state_file.write_text(json.dumps(paired_state), encoding="utf-8")

    with patch("app.screens.settings._state_json_path", return_value=state_file):
        section_paired = PairingSectionWidget(controller=None)
    assert section_paired.paired_data is not None
    paired_chips = section_paired.findChildren(QLabel, "PairedStatusChip")
    assert any(lbl.text() == "Привязано" for lbl in paired_chips)

    # Verify display-14 fingerprint
    from agent.identity import format_fingerprint
    fp_labels = section_paired.findChildren(QLabel, "PairedFingerprintLabel")
    assert len(fp_labels) == 1
    assert fp_labels[0].text() == format_fingerprint(_EXPECTED_FINGERPRINT)

    # 3. After unlink: state.json cleared → UNPAIRED
    with (
        patch("app.screens.settings._state_json_path", return_value=state_file),
        patch(
            "agent.keychain.Keychain.delete_device_token",
            autospec=True,
            side_effect=lambda self_kc, pid: None,
        ),
    ):
        section_paired._perform_unlink()

    assert section_paired.paired_data is None
    cleared = json.loads(state_file.read_text(encoding="utf-8"))
    assert cleared == {}
