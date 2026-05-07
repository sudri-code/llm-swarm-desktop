"""
Keychain abstraction for llm-swarm-desktop.

Wraps the ``keyring`` library to securely store and retrieve Ed25519 private
keys and device tokens in the OS-native credential store:

- macOS: Keychain (via ``keyring`` default backend)
- Windows: Credential Manager (DPAPI)
- Linux: libsecret / GNOME Keyring (D-Bus session)

Security invariants (ADR-0002):
- Ed25519 private keys are stored base64-encoded (keyring requires str values).
- ``device_token`` is stored as-is (str).
- Neither value is ever written to plain files, logged, or passed through ENV.
- Private key and token are accessible only by peer_id-scoped username keys.

Username conventions in keyring:
- private key : ``"<peer_id>:privkey"``
- device token: ``"<peer_id>:device_token"``

The ``list_peer_ids()`` method maintains a JSON index at
``platformdirs.user_config_dir("llm-swarm-desktop") / "peers.json"``
because the keyring API does not provide enumeration.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
from pathlib import Path

import keyring
import keyring.errors
import platformdirs

logger = logging.getLogger(__name__)

SERVICE = "ru.sudri.llm-swarm-desktop"

_PEERS_FILE_NAME = "peers.json"


def _peers_index_path() -> Path:
    config_dir = Path(platformdirs.user_config_dir("llm-swarm-desktop"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / _PEERS_FILE_NAME


def _load_index() -> list[str]:
    path = _peers_index_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item) for item in data]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_index(peer_ids: list[str]) -> None:
    path = _peers_index_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(sorted(set(peer_ids)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


class Keychain:
    """OS-native credential store wrapper for desktop node secrets.

    All secrets (Ed25519 private keys and device tokens) are routed through
    the OS keychain — never through plain files, environment variables, or
    log output.

    Private key bytes are stored base64-encoded because ``keyring`` stores
    values as Unicode strings.  The ``load_private_key`` method decodes them
    back to ``bytes`` transparently.
    """

    # ------------------------------------------------------------------ #
    # Private key                                                          #
    # ------------------------------------------------------------------ #

    def store_private_key(self, peer_id: str, private_key_bytes: bytes) -> None:
        """Store a 32-byte Ed25519 raw private key in the OS keychain.

        The bytes are base64-encoded before storage because ``keyring``
        requires Unicode string values.

        Args:
            peer_id: The peer's public identity (base58-encoded, no secrets).
            private_key_bytes: 32-byte raw Ed25519 private key.
        """
        encoded = base64.b64encode(private_key_bytes).decode("ascii")
        keyring.set_password(SERVICE, f"{peer_id}:privkey", encoded)
        # Update the index (no sensitive data touches the index file).
        peers = _load_index()
        if peer_id not in peers:
            peers.append(peer_id)
            _save_index(peers)
        logger.debug("Stored private key for peer_id=%s", peer_id)

    def load_private_key(self, peer_id: str) -> bytes | None:
        """Load the raw Ed25519 private key from the OS keychain.

        Args:
            peer_id: The peer's public identity.

        Returns:
            32-byte raw private key, or ``None`` if not found.
        """
        encoded = keyring.get_password(SERVICE, f"{peer_id}:privkey")
        if encoded is None:
            logger.debug("Private key not found for peer_id=%s", peer_id)
            return None
        logger.debug("Loaded private key for peer_id=%s", peer_id)
        return base64.b64decode(encoded.encode("ascii"))

    # ------------------------------------------------------------------ #
    # Device token                                                         #
    # ------------------------------------------------------------------ #

    def store_device_token(self, peer_id: str, token: str) -> None:
        """Store the opaque device_token in the OS keychain.

        The token must NOT be logged or written to any file.

        Args:
            peer_id: The peer's public identity (used as namespace).
            token: Opaque device token received from BFF after approval.
        """
        keyring.set_password(SERVICE, f"{peer_id}:device_token", token)
        logger.debug("Stored device_token for peer_id=%s", peer_id)

    def load_device_token(self, peer_id: str) -> str | None:
        """Load the device_token from the OS keychain.

        Args:
            peer_id: The peer's public identity.

        Returns:
            Opaque token string, or ``None`` if not present.
        """
        token = keyring.get_password(SERVICE, f"{peer_id}:device_token")
        if token is None:
            logger.debug("device_token not found for peer_id=%s", peer_id)
        else:
            logger.debug("Loaded device_token for peer_id=%s", peer_id)
        return token

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def delete_device(self, peer_id: str) -> None:
        """Remove both the private key and device_token for a peer_id.

        Also removes the peer_id from the peers.json index.

        Args:
            peer_id: The peer's public identity to unlink.
        """
        for username in (f"{peer_id}:privkey", f"{peer_id}:device_token"):
            with contextlib.suppress(keyring.errors.PasswordDeleteError):
                keyring.delete_password(SERVICE, username)
        peers = _load_index()
        updated = [p for p in peers if p != peer_id]
        _save_index(updated)
        logger.debug("Deleted device credentials for peer_id=%s", peer_id)

    def list_peer_ids(self) -> list[str]:
        """Return all peer_ids that have been registered in this keychain.

        The list is maintained in a JSON index file inside
        ``platformdirs.user_config_dir("llm-swarm-desktop")``.
        The file contains no sensitive data — only peer_ids (public identities).

        Returns:
            Sorted list of peer_id strings.
        """
        return sorted(_load_index())
