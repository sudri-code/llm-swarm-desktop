"""Pairing flow controller — device authorization (ADR-0002, pairing.md Stage 1 W3).

State machine, events, bridge (Qt Signal), and async poll loop for linking a
desktop node to a sudri.ru account via RFC 8628-like device authorization.

Security invariants (enforced here, not delegated to callers):
- Ed25519 private key is zeroed from local variables as soon as it is stored in
  keychain.  ``generate_keypair()`` returns raw bytes; we hold them only until
  ``keychain.store_private_key()`` returns successfully.
- ``device_token`` from BFF ``ok`` response is stored in keychain immediately;
  the pydantic model is *not* cached anywhere (frozen dataclass, but the field
  is discarded from local scope with ``del``).
- ``device_code`` is never passed to ``logger.*``; the deny-list in
  ``app.logging_setup`` provides defence-in-depth, but we do not rely on it.
- Any ``keyring.errors.*`` exception during store → ``ERROR_KEYCHAIN``; the
  private key (already stored) is intentionally *not* deleted (user may retry).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import keyring.errors
import platformdirs
from PySide6.QtCore import QObject, Signal

from agent.auth.bff_client import BFFClient, BFFError, BFFRateLimitedError
from agent.auth.models import (
    DevicePollDenied,
    DevicePollExpired,
    DevicePollPending,
    DevicePollRequest,
    DevicePollSlowDown,
    DevicePollSuccess,
    DeviceStartRequest,
)
from agent.identity import compute_fingerprint, compute_peer_id, generate_keypair
from agent.keychain import Keychain

if TYPE_CHECKING:

    from agent.auth.models import HardwareSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class PairingState(StrEnum):
    UNPAIRED = "unpaired"
    STARTING = "starting"
    AWAITING_USER = "awaiting_user"
    POLLING = "polling"
    SUCCESS = "success"
    ERROR_EXPIRED = "error_expired"
    ERROR_DENIED = "error_denied"
    ERROR_NETWORK = "error_network"
    ERROR_INVALID = "error_invalid"
    ERROR_KEYCHAIN = "error_keychain"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Failure reasons
# ---------------------------------------------------------------------------


class PairingFailureReason(StrEnum):
    EXPIRED = "expired"
    DENIED = "denied"
    NETWORK = "network"
    INVALID_PUBKEY = "invalid_pubkey"
    RATE_LIMITED = "rate_limited"
    KEYCHAIN_UNAVAILABLE = "keychain_unavailable"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Events (controller → UI)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairingStarted:
    user_code: str
    verification_uri_complete: str
    expires_at: datetime
    interval: int
    fingerprint_raw: str  # 12 chars; UI formats via format_fingerprint


@dataclass(frozen=True, slots=True)
class PairingPolling:
    attempt: int  # 1-based; first emit marks AWAITING_USER → POLLING


@dataclass(frozen=True, slots=True)
class PairingSucceeded:
    device_id: str
    user_id: str
    scope: list[str]
    fingerprint_raw: str
    token_expires_at: datetime
    label: str  # confirmed by BFF, may differ from user input
    # device_token is intentionally absent — only in keychain


@dataclass(frozen=True, slots=True)
class PairingFailed:
    reason: PairingFailureReason
    message: str  # Russian, for UI display


PairingEvent = PairingStarted | PairingPolling | PairingSucceeded | PairingFailed

# ---------------------------------------------------------------------------
# Qt bridge
# ---------------------------------------------------------------------------


class PairingBridge(QObject):
    """Qt bridge exposing a single Signal for all PairingEvent payloads.

    The controller calls ``event_received.emit(event)`` from within the
    qasync event loop (single OS thread) — no thread-safety concerns.
    """

    event_received = Signal(object)  # payload: PairingEvent


# ---------------------------------------------------------------------------
# Label sanitizer
# ---------------------------------------------------------------------------

_FALLBACK_LABEL = "Устройство"


def _default_label_sanitizer(label: str) -> str:
    """NFC-normalize, strip Cc control chars, truncate to 64 codepoints.

    Returns *_FALLBACK_LABEL* if the result is empty (ADR-0002 §AC8).
    """
    normalized = unicodedata.normalize("NFC", label)
    # Strip Unicode category Cc (control characters)
    cleaned = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cc")
    cleaned = cleaned.strip()
    # Truncate to 64 codepoints (not bytes)
    truncated = cleaned[:64]
    return truncated if truncated else _FALLBACK_LABEL


# ---------------------------------------------------------------------------
# state.json helpers
# ---------------------------------------------------------------------------

_STATE_FILE_NAME = "state.json"


def _state_json_path() -> Path:
    config_dir = Path(platformdirs.user_config_dir("llm-swarm-desktop"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / _STATE_FILE_NAME


def _write_state_json(data: dict[str, str]) -> None:
    """Atomic write of state.json via tempfile + os.replace."""
    path = _state_json_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def clear_state_json() -> None:
    """Remove state.json if present (on cancel/unlink).

    Called by the unlink flow (Settings → «Отвязать устройство»).
    """
    path = _state_json_path()
    if path.exists():
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PairingController
# ---------------------------------------------------------------------------

_MAX_BACKOFF = 30.0
_MAX_NETWORK_RETRIES = 10


class PairingController:
    """Async controller for the device pairing flow (ADR-0002, pairing.md).

    Plain Python class (not QObject).  Emits events through ``PairingBridge``
    which holds the Qt Signal.

    All dependencies are injected via keyword arguments for testability (§12).

    Parameters
    ----------
    bridge:
        ``PairingBridge`` instance owned by the GUI layer.  ``event_received``
        is emitted on every state transition.  Mandatory — no default.
    bff_client:
        HTTP client for BFF device auth endpoints.  Defaults to
        ``BFFClient()``.
    keychain:
        Credential store for Ed25519 private key and device_token.  Defaults
        to ``Keychain()``.
    identity_factory:
        Callable returning ``(private_key_bytes, public_key_bytes)`` — both
        32 bytes raw.  Defaults to ``generate_keypair``.
    now:
        Callable returning current UTC datetime.  Defaults to
        ``datetime.now(tz=timezone.utc)``.  Injectable for TTL tests.
    sleep:
        Async callable accepting a float delay.  Defaults to
        ``asyncio.sleep``.  Injectable to make poll-loop tests instant.
    label_sanitizer:
        Callable ``str → str`` applied to device label before sending to BFF.
        Defaults to ``_default_label_sanitizer``.
    """

    def __init__(
        self,
        *,
        bridge: PairingBridge,
        bff_client: BFFClient | None = None,
        keychain: Keychain | None = None,
        identity_factory: Callable[[], tuple[bytes, bytes]] | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        label_sanitizer: Callable[[str], str] | None = None,
    ) -> None:
        self._bridge = bridge
        self._bff_client = bff_client if bff_client is not None else BFFClient()
        self._keychain = keychain if keychain is not None else Keychain()
        self._identity_factory = (
            identity_factory if identity_factory is not None else generate_keypair
        )
        self._now = now if now is not None else lambda: datetime.now(tz=UTC)
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._label_sanitizer = (
            label_sanitizer if label_sanitizer is not None else _default_label_sanitizer
        )

        self._state: PairingState = PairingState.UNPAIRED
        self._task: asyncio.Task[None] | None = None

        # Stored across retries (keypair stable between retries per §6)
        self._peer_id: str | None = None
        self._public_key_bytes: bytes | None = None

        # Last label/hardware for retry
        self._last_label: str = ""
        self._last_hardware: HardwareSummary | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> PairingState:
        return self._state

    @property
    def bridge(self) -> PairingBridge:
        """Qt-bridge для подписки GUI на события привязки."""
        return self._bridge

    # ------------------------------------------------------------------
    # Commands (UI → controller)
    # ------------------------------------------------------------------

    def start_pairing(
        self,
        label: str,
        hardware: HardwareSummary | None = None,
    ) -> None:
        """Start the device pairing flow.

        Idempotent: no-op in STARTING / AWAITING_USER / POLLING / SUCCESS.
        In ERROR_* / CANCELLED — equivalent to retry (restarts flow).
        """
        if self._state in (
            PairingState.STARTING,
            PairingState.AWAITING_USER,
            PairingState.POLLING,
            PairingState.SUCCESS,
        ):
            logger.debug("start_pairing: no-op in state=%s", self._state)
            return

        self._last_label = label
        self._last_hardware = hardware
        self._transition(PairingState.STARTING)

        self._task = asyncio.get_event_loop().create_task(
            self._run_flow(label, hardware),
            name="pairing_flow",
        )

    def cancel_pairing(self) -> None:
        """Cancel the active pairing flow; transition to CANCELLED then UNPAIRED.

        Does not delete the Ed25519 keypair (§6 — reuse across retries).

        - Active task: cancel it; state transition happens in _run_flow CancelledError handler.
        - No active task (ERROR_*, AWAITING_USER, STARTING, POLLING with done task):
          transition to CANCELLED + emit PairingFailed immediately.
        - SUCCESS: no-op (use unlink flow instead).
        - UNPAIRED / CANCELLED: no-op.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
            # State transition handled in _run_flow's except CancelledError block.
            return
        # Task is None or already done — handle remaining states directly.
        if self._state not in (
            PairingState.SUCCESS,
            PairingState.UNPAIRED,
            PairingState.CANCELLED,
        ):
            self._transition(PairingState.CANCELLED)
            self._emit(PairingFailed(reason=PairingFailureReason.CANCELLED, message=""))
            self._transition(PairingState.UNPAIRED)

    def retry_pairing(self) -> None:
        """Retry: equivalent to start_pairing with last label/hardware."""
        self.start_pairing(self._last_label, self._last_hardware)

    # ------------------------------------------------------------------
    # Internal flow
    # ------------------------------------------------------------------

    def _transition(self, new_state: PairingState) -> None:
        logger.info(
            "pairing state: %s → %s",
            self._state,
            new_state,
        )
        self._state = new_state

    def _emit(self, event: PairingEvent) -> None:
        self._bridge.event_received.emit(event)

    async def _run_flow(
        self,
        label: str,
        hardware: HardwareSummary | None,
    ) -> None:
        """Full pairing flow; cancellation is handled via CancelledError."""
        try:
            await self._do_flow(label, hardware)
        except asyncio.CancelledError:
            self._transition(PairingState.CANCELLED)
            # Per §6: clear in-memory pairing state, keep keypair.
            self._transition(PairingState.UNPAIRED)
            raise  # Must re-raise so asyncio marks the Task properly

    async def _do_flow(
        self,
        label: str,
        hardware: HardwareSummary | None,
    ) -> None:
        import base64

        sanitized_label = self._label_sanitizer(label)

        # -----------------------------------------------------------
        # Step 1–4: keypair + keychain (§8)
        # -----------------------------------------------------------
        private_key_bytes, public_key_bytes = self._identity_factory()
        peer_id = compute_peer_id(public_key_bytes)
        fingerprint = compute_fingerprint(public_key_bytes)

        self._peer_id = peer_id
        self._public_key_bytes = public_key_bytes

        try:
            self._keychain.store_private_key(peer_id, private_key_bytes)
        except (keyring.errors.KeyringError, OSError) as exc:
            logger.error("Keychain unavailable when storing private key: %s", type(exc).__name__)
            self._transition(PairingState.ERROR_KEYCHAIN)
            self._emit(
                PairingFailed(
                    reason=PairingFailureReason.KEYCHAIN_UNAVAILABLE,
                    message=(
                        "Системное хранилище паролей недоступно. "
                        "Разблокируйте Связку ключей и повторите."
                    ),
                )
            )
            return
        finally:
            # Zero from local scope regardless of outcome
            del private_key_bytes

        # -----------------------------------------------------------
        # Step 5: POST /device/start
        # -----------------------------------------------------------
        device_pubkey_b64 = base64.urlsafe_b64encode(public_key_bytes).rstrip(b"=").decode()

        from importlib.metadata import PackageNotFoundError

        try:
            from importlib.metadata import version as _pkg_ver

            client_version = _pkg_ver("llm-swarm-desktop")
        except PackageNotFoundError:
            client_version = "0.0.0.dev"

        start_req = DeviceStartRequest(
            device_pubkey=device_pubkey_b64,
            device_label=sanitized_label,
            client_version=client_version,
            hardware_summary=hardware,
        )

        try:
            start_resp = await self._bff_client.device_start(start_req)
        except BFFRateLimitedError as exc:
            logger.warning("device/start rate-limited (retry_after=%s)", exc.retry_after)
            self._transition(PairingState.ERROR_NETWORK)
            self._emit(
                PairingFailed(
                    reason=PairingFailureReason.RATE_LIMITED,
                    message="Слишком много попыток подряд. Попробуйте позже.",
                )
            )
            return
        except BFFError as exc:
            if exc.status_code == 400:
                logger.error("device/start: 400 from BFF (invalid_pubkey)")
                self._transition(PairingState.ERROR_INVALID)
                self._emit(
                    PairingFailed(
                        reason=PairingFailureReason.INVALID_PUBKEY,
                        message=(
                            "Не удалось подготовить устройство к привязке. "
                            "Перезапустите приложение."
                        ),
                    )
                )
            else:
                logger.error("device/start: BFF error status=%s", exc.status_code)
                self._transition(PairingState.ERROR_NETWORK)
                self._emit(
                    PairingFailed(
                        reason=PairingFailureReason.UNKNOWN,
                        message="Не удалось связаться с sudri.ru.",
                    )
                )
            return
        except Exception as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.error("device/start: network error: %s", type(exc).__name__)
            self._transition(PairingState.ERROR_NETWORK)
            self._emit(
                PairingFailed(
                    reason=PairingFailureReason.NETWORK,
                    message="Не удалось связаться с sudri.ru. Проверьте интернет.",
                )
            )
            return

        # -----------------------------------------------------------
        # Transition STARTING → AWAITING_USER, emit PairingStarted
        # -----------------------------------------------------------
        expires_at = self._now() + timedelta(seconds=start_resp.expires_in)
        self._transition(PairingState.AWAITING_USER)
        self._emit(
            PairingStarted(
                user_code=_format_user_code(start_resp.user_code),
                verification_uri_complete=start_resp.verification_uri_complete,
                expires_at=expires_at,
                interval=start_resp.interval,
                fingerprint_raw=fingerprint,
            )
        )

        # -----------------------------------------------------------
        # Step 6: poll loop
        # -----------------------------------------------------------
        await self._poll_loop(
            device_code=start_resp.device_code,
            public_key_bytes=public_key_bytes,
            expires_at=expires_at,
            initial_interval=start_resp.interval,
            peer_id=peer_id,
            fingerprint=fingerprint,
            label=sanitized_label,
        )

    async def _poll_loop(  # noqa: PLR0912, PLR0915
        self,
        device_code: str,
        public_key_bytes: bytes,
        expires_at: datetime,
        initial_interval: int,
        peer_id: str,
        fingerprint: str,
        label: str,
    ) -> None:
        """Polling loop per pairing.md §7 pseudocode."""
        import base64

        device_pubkey_b64 = base64.urlsafe_b64encode(public_key_bytes).rstrip(b"=").decode()
        poll_req = DevicePollRequest(
            device_code=device_code,
            device_pubkey=device_pubkey_b64,
        )

        interval = float(initial_interval)
        attempt = 0
        network_failures = 0

        while True:
            # Check deadline before sleeping
            remaining = (expires_at - self._now()).total_seconds()
            if remaining <= 0:
                self._transition(PairingState.ERROR_EXPIRED)
                self._emit(
                    PairingFailed(
                        reason=PairingFailureReason.EXPIRED,
                        message="Время вышло. Начните привязку заново.",
                    )
                )
                return

            # Calculate sleep delay (§7 formula)
            jitter = random.uniform(-0.5, 0.5)
            if network_failures > 0:
                exp_backoff = min(2.0 ** network_failures, _MAX_BACKOFF)
                delay = max(interval + jitter, exp_backoff)
            else:
                delay = interval + jitter

            delay = min(delay, remaining - 0.5)
            if delay <= 0:
                self._transition(PairingState.ERROR_EXPIRED)
                self._emit(
                    PairingFailed(
                        reason=PairingFailureReason.EXPIRED,
                        message="Время вышло. Начните привязку заново.",
                    )
                )
                return

            # CancelledError here = cancel_pairing(); must not be caught
            await self._sleep(delay)

            attempt += 1

            # Transition AWAITING_USER → POLLING on first attempt
            if self._state == PairingState.AWAITING_USER:
                self._transition(PairingState.POLLING)

            self._emit(PairingPolling(attempt=attempt))
            logger.debug("pairing poll attempt=%d", attempt)

            # Perform the actual poll
            try:
                response = await self._bff_client.device_poll(poll_req)
                network_failures = 0
            except BFFRateLimitedError:
                # 429 = slow down, but not a terminal error (§ task spec)
                logger.warning("pairing poll: 429 rate-limited, backing off")
                network_failures += 1
                if network_failures >= _MAX_NETWORK_RETRIES:
                    self._transition(PairingState.ERROR_NETWORK)
                    self._emit(
                        PairingFailed(
                            reason=PairingFailureReason.NETWORK,
                            message="Слишком много ошибок сети. Проверьте соединение.",
                        )
                    )
                    return
                continue
            except BFFError as exc:
                logger.error("pairing poll: BFF error status=%s", exc.status_code)
                if exc.status_code == 400:
                    self._transition(PairingState.ERROR_INVALID)
                    self._emit(
                        PairingFailed(
                            reason=PairingFailureReason.INVALID_PUBKEY,
                            message="Ошибка сервера. Перезапустите приложение.",
                        )
                    )
                else:
                    self._transition(PairingState.ERROR_NETWORK)
                    self._emit(
                        PairingFailed(
                            reason=PairingFailureReason.UNKNOWN,
                            message="Не удалось связаться с sudri.ru.",
                        )
                    )
                return
            except Exception as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                network_failures += 1
                logger.warning(
                    "pairing poll: network error (attempt=%d failures=%d): %s",
                    attempt,
                    network_failures,
                    type(exc).__name__,
                )
                if network_failures >= _MAX_NETWORK_RETRIES:
                    self._transition(PairingState.ERROR_NETWORK)
                    self._emit(
                        PairingFailed(
                            reason=PairingFailureReason.NETWORK,
                            message=(
                                "Нет соединения с sudri.ru. "
                                "Проверьте интернет-соединение и повторите."
                            ),
                        )
                    )
                    return
                continue

            # Dispatch on discriminated poll response
            if isinstance(response, DevicePollPending):
                continue
            elif isinstance(response, DevicePollSlowDown):
                # Only accept if new interval >= current (§7)
                new_interval = float(response.interval)
                interval = max(interval, new_interval)
                logger.debug("pairing poll: slow_down, new interval=%.1f", interval)
                continue
            elif isinstance(response, DevicePollExpired):
                self._transition(PairingState.ERROR_EXPIRED)
                self._emit(
                    PairingFailed(
                        reason=PairingFailureReason.EXPIRED,
                        message="Время вышло. Код подтверждения истёк.",
                    )
                )
                return
            elif isinstance(response, DevicePollDenied):
                self._transition(PairingState.ERROR_DENIED)
                self._emit(
                    PairingFailed(
                        reason=PairingFailureReason.DENIED,
                        message=(
                            "Привязка отклонена. Если это была ошибка — попробуйте ещё раз."
                        ),
                    )
                )
                return
            else:
                # DevicePollSuccess — only remaining variant in discriminated union
                await self._handle_success(
                    ok=response,
                    peer_id=peer_id,
                    fingerprint=fingerprint,
                    label=label,
                )
                return

    async def _handle_success(
        self,
        ok: DevicePollSuccess,
        peer_id: str,
        fingerprint: str,
        label: str,
    ) -> None:
        """Store device_token, emit PairingSucceeded, save state.json (§8 steps 7–11)."""
        # Step 7: store device_token in keychain (MUST NOT log the token)
        try:
            self._keychain.store_device_token(peer_id, ok.device_token)
        except (keyring.errors.KeyringError, OSError) as exc:
            logger.error(
                "Keychain unavailable when storing device_token: %s",
                type(exc).__name__,
            )
            self._transition(PairingState.ERROR_KEYCHAIN)
            self._emit(
                PairingFailed(
                    reason=PairingFailureReason.KEYCHAIN_UNAVAILABLE,
                    message=(
                        "Системное хранилище паролей недоступно. "
                        "Разблокируйте Связку ключей и повторите."
                    ),
                )
            )
            return

        # Parse token_expires_at
        try:
            token_expires_at = datetime.fromisoformat(ok.token_expires_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            # Fallback: use 365 days from now
            token_expires_at = self._now() + timedelta(days=365)

        # Step 10: emit PairingSucceeded (no device_token in payload)
        self._transition(PairingState.SUCCESS)
        self._emit(
            PairingSucceeded(
                device_id=ok.device_id,
                user_id=ok.user_id,
                scope=list(ok.scope),
                fingerprint_raw=fingerprint,
                token_expires_at=token_expires_at,
                label=label,
            )
        )

        # Step 11: save paired_at to state.json
        try:
            _write_state_json(
                {
                    "paired_at": self._now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "device_label": label,
                    "fingerprint_raw": fingerprint,
                    "peer_id": peer_id,
                }
            )
        except OSError as exc:
            # Non-fatal: UI can still proceed, but state.json won't show paired_at
            logger.warning("Could not write state.json: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_user_code(raw: str) -> str:
    """Format 8-char user_code as XXXX-XXXX.

    If the code already contains a dash it is returned as-is (normalized to
    uppercase).  Otherwise a dash is inserted after position 4.
    """
    cleaned = raw.upper().replace("-", "").replace(" ", "")
    if len(cleaned) == 8:
        return f"{cleaned[:4]}-{cleaned[4:]}"
    # Unexpected format — return normalized uppercase
    return raw.upper()
