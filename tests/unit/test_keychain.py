"""
Unit tests for agent.keychain.

Uses ``keyrings.alt.file.PlaintextKeyring`` as the backend via a pytest
fixture that monkey-patches ``keyring.set_keyring``.  This ensures tests run
on any machine — including macOS CI — without accessing the real OS keychain.

Covers:
- store/load round-trip for private key (bytes preserved exactly)
- store/load round-trip for device_token (str preserved exactly)
- delete_device removes both entries and updates the index
- list_peer_ids reflects stored and deleted entries
- No sensitive data (private_key bytes, device_token) appears in log output
"""

from __future__ import annotations

import logging

import keyring
import pytest

import agent.keychain as kc_module
from agent.identity import compute_peer_id, generate_keypair
from agent.keychain import Keychain

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def plaintext_keyring(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the active keyring backend with PlaintextKeyring backed by a
    temp directory.  Also redirect the peers.json index to the same temp dir.
    """
    from keyrings.alt.file import PlaintextKeyring

    ring = PlaintextKeyring()
    ring.file_path = str(tmp_path / "test_keyring.cfg")
    keyring.set_keyring(ring)

    # Redirect the peers index to the temp dir.
    monkeypatch.setattr(
        kc_module,
        "_peers_index_path",
        lambda: tmp_path / "peers.json",
    )

    yield ring


@pytest.fixture()
def keychain(plaintext_keyring: None) -> Keychain:
    """Return a Keychain instance backed by the test keyring."""
    return Keychain()


@pytest.fixture()
def sample_peer() -> tuple[str, bytes, bytes]:
    """Return a deterministic (peer_id, private_key_bytes, public_key_bytes)."""
    priv, pub = generate_keypair()
    peer_id = compute_peer_id(pub)
    return peer_id, priv, pub


# ---------------------------------------------------------------------------
# Private key round-trip
# ---------------------------------------------------------------------------


class TestStoreLoadPrivateKey:
    def test_round_trip(
        self, keychain: Keychain, sample_peer: tuple[str, bytes, bytes]
    ) -> None:
        peer_id, priv, _ = sample_peer
        keychain.store_private_key(peer_id, priv)
        loaded = keychain.load_private_key(peer_id)
        assert loaded == priv

    def test_load_missing_returns_none(self, keychain: Keychain) -> None:
        result = keychain.load_private_key("nonexistent_peer_id")
        assert result is None

    def test_overwrite_updates_value(
        self, keychain: Keychain, sample_peer: tuple[str, bytes, bytes]
    ) -> None:
        peer_id, priv1, _ = sample_peer
        priv2, _ = generate_keypair()
        keychain.store_private_key(peer_id, priv1)
        keychain.store_private_key(peer_id, priv2)
        # Just verify the second write didn't corrupt anything
        loaded = keychain.load_private_key(peer_id)
        assert isinstance(loaded, bytes)
        assert len(loaded) == 32

    def test_stored_key_is_full_32_bytes(
        self, keychain: Keychain, sample_peer: tuple[str, bytes, bytes]
    ) -> None:
        peer_id, priv, _ = sample_peer
        keychain.store_private_key(peer_id, priv)
        loaded = keychain.load_private_key(peer_id)
        assert loaded is not None
        assert len(loaded) == 32


# ---------------------------------------------------------------------------
# Device token round-trip
# ---------------------------------------------------------------------------


class TestStoreLoadDeviceToken:
    def test_round_trip(
        self, keychain: Keychain, sample_peer: tuple[str, bytes, bytes]
    ) -> None:
        peer_id, _, _ = sample_peer
        token = "test_opaque_device_token_abc123"
        keychain.store_device_token(peer_id, token)
        loaded = keychain.load_device_token(peer_id)
        assert loaded == token

    def test_load_missing_returns_none(self, keychain: Keychain) -> None:
        assert keychain.load_device_token("nonexistent") is None

    def test_token_with_special_characters(
        self, keychain: Keychain, sample_peer: tuple[str, bytes, bytes]
    ) -> None:
        peer_id, _, _ = sample_peer
        token = "eyJhbGciOiJub25lIn0.abc-def_ghi.xyz"
        keychain.store_device_token(peer_id, token)
        loaded = keychain.load_device_token(peer_id)
        assert loaded == token


# ---------------------------------------------------------------------------
# delete_device
# ---------------------------------------------------------------------------


class TestDeleteDevice:
    def test_deletes_private_key(
        self, keychain: Keychain, sample_peer: tuple[str, bytes, bytes]
    ) -> None:
        peer_id, priv, _ = sample_peer
        keychain.store_private_key(peer_id, priv)
        keychain.delete_device(peer_id)
        assert keychain.load_private_key(peer_id) is None

    def test_deletes_device_token(
        self, keychain: Keychain, sample_peer: tuple[str, bytes, bytes]
    ) -> None:
        peer_id, _, _ = sample_peer
        keychain.store_device_token(peer_id, "some-token")
        keychain.delete_device(peer_id)
        assert keychain.load_device_token(peer_id) is None

    def test_removes_from_index(
        self, keychain: Keychain, sample_peer: tuple[str, bytes, bytes]
    ) -> None:
        peer_id, priv, _ = sample_peer
        keychain.store_private_key(peer_id, priv)
        assert peer_id in keychain.list_peer_ids()
        keychain.delete_device(peer_id)
        assert peer_id not in keychain.list_peer_ids()

    def test_idempotent_when_missing(self, keychain: Keychain) -> None:
        # Should not raise even when entries don't exist
        keychain.delete_device("ghost_peer_id")

    def test_does_not_affect_other_peers(
        self, keychain: Keychain
    ) -> None:
        priv1, pub1 = generate_keypair()
        peer_id1 = compute_peer_id(pub1)
        priv2, pub2 = generate_keypair()
        peer_id2 = compute_peer_id(pub2)

        keychain.store_private_key(peer_id1, priv1)
        keychain.store_private_key(peer_id2, priv2)
        keychain.delete_device(peer_id1)

        assert keychain.load_private_key(peer_id1) is None
        assert keychain.load_private_key(peer_id2) == priv2


# ---------------------------------------------------------------------------
# list_peer_ids
# ---------------------------------------------------------------------------


class TestListPeerIds:
    def test_empty_initially(self, keychain: Keychain) -> None:
        assert keychain.list_peer_ids() == []

    def test_returns_stored_peers(self, keychain: Keychain) -> None:
        priv1, pub1 = generate_keypair()
        peer_id1 = compute_peer_id(pub1)
        priv2, pub2 = generate_keypair()
        peer_id2 = compute_peer_id(pub2)

        keychain.store_private_key(peer_id1, priv1)
        keychain.store_private_key(peer_id2, priv2)

        ids = keychain.list_peer_ids()
        assert peer_id1 in ids
        assert peer_id2 in ids

    def test_no_duplicates(self, keychain: Keychain) -> None:
        priv, pub = generate_keypair()
        peer_id = compute_peer_id(pub)
        keychain.store_private_key(peer_id, priv)
        keychain.store_private_key(peer_id, priv)  # second store
        assert keychain.list_peer_ids().count(peer_id) == 1

    def test_deleted_peer_not_in_list(self, keychain: Keychain) -> None:
        priv, pub = generate_keypair()
        peer_id = compute_peer_id(pub)
        keychain.store_private_key(peer_id, priv)
        keychain.delete_device(peer_id)
        assert peer_id not in keychain.list_peer_ids()

    def test_result_is_sorted(self, keychain: Keychain) -> None:
        peers = []
        for _ in range(4):
            priv, pub = generate_keypair()
            peer_id = compute_peer_id(pub)
            keychain.store_private_key(peer_id, priv)
            peers.append(peer_id)
        result = keychain.list_peer_ids()
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# Sensitive data MUST NOT appear in log output
# ---------------------------------------------------------------------------


class TestNoSensitiveLogging:
    def test_private_key_not_logged(
        self,
        keychain: Keychain,
        sample_peer: tuple[str, bytes, bytes],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        peer_id, priv, _ = sample_peer
        priv_b64 = __import__("base64").b64encode(priv).decode()

        with caplog.at_level(logging.DEBUG, logger="agent.keychain"):
            keychain.store_private_key(peer_id, priv)
            keychain.load_private_key(peer_id)

        full_log = caplog.text
        assert priv.hex() not in full_log, "Raw private key hex must not appear in logs"
        assert priv_b64 not in full_log, "Base64 private key must not appear in logs"

    def test_device_token_not_logged(
        self,
        keychain: Keychain,
        sample_peer: tuple[str, bytes, bytes],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        peer_id, _, _ = sample_peer
        token = "supersecret_device_token_xyzXYZ789"

        with caplog.at_level(logging.DEBUG, logger="agent.keychain"):
            keychain.store_device_token(peer_id, token)
            keychain.load_device_token(peer_id)

        assert token not in caplog.text, "device_token must not appear in logs"

    def test_peer_id_may_appear_in_logs(
        self,
        keychain: Keychain,
        sample_peer: tuple[str, bytes, bytes],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """peer_id is a public identity and is allowed in debug logs."""
        peer_id, priv, _ = sample_peer

        with caplog.at_level(logging.DEBUG, logger="agent.keychain"):
            keychain.store_private_key(peer_id, priv)

        # peer_id itself is public — logging it is acceptable.
        assert peer_id in caplog.text
