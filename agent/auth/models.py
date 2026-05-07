"""Pydantic v2 models for BFF device authorization endpoints (ADR-0002)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    """Base with extra='forbid' to detect contract drift."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Hardware summary (optional, attached to device_start)
# ---------------------------------------------------------------------------


class HardwareSummary(_StrictModel):
    vram_gib: float | None = None
    ram_gib: float | None = None
    gpu_model: str | None = None
    platform: str | None = None  # "darwin-arm64" | "win32-x86_64" | "linux-x86_64"


# ---------------------------------------------------------------------------
# POST /api/v1/auth/device/start
# ---------------------------------------------------------------------------


class DeviceStartRequest(_StrictModel):
    """Request body for device/start.

    ``device_pubkey`` — base64url-encoded raw Ed25519 public key (32 bytes,
    URL-safe alphabet, no padding), as specified in ADR-0002 §A.
    ``device_label`` — up to 64 codepoints after sanitisation; defaults to
    hostname.  BFF applies the same sanitisation as defence-in-depth.
    """

    device_pubkey: str
    device_label: str
    client_version: str
    hardware_summary: HardwareSummary | None = None


class DeviceStartResponse(_StrictModel):
    """Response body from device/start (ADR-0002 §A)."""

    device_code: str
    user_code: str  # 8 chars [A-HJ-NP-Z2-9], displayed as XXXX-XXXX
    verification_uri: str
    verification_uri_complete: str
    expires_in: int  # 600
    interval: int  # 3


# ---------------------------------------------------------------------------
# POST /api/v1/auth/device/poll
# ---------------------------------------------------------------------------


class DevicePollRequest(_StrictModel):
    """Request body for device/poll.

    ``device_pubkey`` is sent on every poll so BFF can compare it against the
    value stored in ``device:code:<device_code>`` at start time, preventing
    device_code hijacking (ADR-0002 §A).
    """

    device_code: str
    device_pubkey: str


# -- discriminated union for poll response -----------------------------------


class _PollBase(_StrictModel):
    pass


class DevicePollPending(_PollBase):
    """Poll returned while the user has not yet acted in SPA."""

    status: Literal["authorization_pending"]


class DevicePollSlowDown(_PollBase):
    """BFF detected polling faster than interval; new interval provided."""

    status: Literal["slow_down"]
    interval: int  # updated value; desktop must adopt this


class DevicePollExpired(_PollBase):
    """device_code TTL elapsed before the user approved."""

    status: Literal["expired_token"]


class DevicePollDenied(_PollBase):
    """User clicked Deny in SPA."""

    status: Literal["access_denied"]


class DevicePollSuccess(_PollBase):
    """User approved; device_token is ready (status='ok' per ADR-0002 §A)."""

    status: Literal["ok"]
    device_token: str
    device_id: str
    user_id: str
    token_expires_at: str  # ISO-8601 UTC (abs cap)
    scope: list[str]  # ["node:run", "accounting:report", "balance:read"]


DevicePollResponse = Annotated[
    DevicePollPending
    | DevicePollSlowDown
    | DevicePollExpired
    | DevicePollDenied
    | DevicePollSuccess,
    Field(discriminator="status"),
]
"""Discriminated union parsed from POST /api/v1/auth/device/poll response.

The ``status`` field drives the discriminator.  Pydantic will raise
``ValidationError`` for any unknown status value, catching contract drift.

Statuses (RFC 8628-like, normalised to BFF contract, ADR-0002 §B):
  "authorization_pending" — user has not acted yet (NOT the bare "pending"
      variant — see changelog nit 2026-05-06).
  "slow_down"             — desktop polled too fast; adopt new interval.
  "expired_token"         — device_code TTL elapsed.
  "access_denied"         — user clicked Deny.
  "ok"                    — approved; device_token present.
"""


# ---------------------------------------------------------------------------
# Error response (shared BFF error envelope)
# ---------------------------------------------------------------------------


class DeviceErrorResponse(_StrictModel):
    """Standard BFF error envelope returned on 4xx/5xx."""

    error: str
    error_description: str | None = None
    status_code: int | None = None  # populated by BFFClient from HTTP status
