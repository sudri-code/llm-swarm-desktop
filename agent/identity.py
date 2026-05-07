"""
Identity primitives for llm-swarm-desktop.

Provides Ed25519 keypair generation and peer_id / fingerprint derivation.
All computations are deterministic and produce no side effects.

Conventions (from ADR-0002):
- peer_id  = base58(sha256(pubkey))          — full-length, used by swarm protocol
- fingerprint raw  = peer_id[:12]            — stored in DB and API responses
- fingerprint display = "aBc1 dEf2 gH3j"    — shown in UI only, never stored
"""

from __future__ import annotations

import hashlib

import base58 as _b58
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate a new Ed25519 keypair.

    Returns:
        (private_key_bytes, public_key_bytes) — both 32 bytes raw.
    """
    private_key = Ed25519PrivateKey.generate()
    private_key_bytes: bytes = private_key.private_bytes_raw()
    public_key_bytes: bytes = private_key.public_key().public_bytes_raw()
    return private_key_bytes, public_key_bytes


def compute_peer_id(public_key_bytes: bytes) -> str:
    """Derive peer_id from an Ed25519 public key.

    peer_id = base58(sha256(pubkey))

    This is the canonical swarm identity invariant: peer_id is always derived
    from the public key, never generated independently.

    Args:
        public_key_bytes: 32-byte raw Ed25519 public key.

    Returns:
        Base58-encoded string (44 characters for a 32-byte sha256 digest).
    """
    digest = hashlib.sha256(public_key_bytes).digest()
    return _b58.b58encode(digest).decode()


def compute_fingerprint(public_key_bytes: bytes) -> str:
    """Derive the raw 12-character fingerprint from an Ed25519 public key.

    fingerprint_raw = base58(sha256(pubkey))[:12]

    This is the value stored in DB, API responses, and Redis.
    Do NOT store the display form (with spaces) in persistent storage.

    Args:
        public_key_bytes: 32-byte raw Ed25519 public key.

    Returns:
        12-character base58 string (raw fingerprint, no spaces).
    """
    return compute_peer_id(public_key_bytes)[:12]


def format_fingerprint(raw: str) -> str:
    """Format a raw 12-character fingerprint for display in the UI.

    Splits the 12-character string into 3 groups of 4, separated by spaces,
    producing a 14-character display string: "aBc1 dEf2 gH3j".

    This form is used only in UI rendering (GUI, SPA, email body).
    Never store the display form in DB or pass it in API payloads.

    Args:
        raw: 12-character raw fingerprint from compute_fingerprint().

    Returns:
        14-character display string with spaces every 4 characters.

    Raises:
        ValueError: if raw is not exactly 12 characters.
    """
    if len(raw) != 12:
        raise ValueError(f"fingerprint raw must be exactly 12 characters, got {len(raw)}")
    return f"{raw[0:4]} {raw[4:8]} {raw[8:12]}"
