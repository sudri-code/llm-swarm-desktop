"""
Unit tests for agent.identity.

Covers:
- generate_keypair returns correct byte lengths
- compute_peer_id determinism (known vectors from ADR-0002)
- compute_fingerprint returns first 12 chars of peer_id
- format_fingerprint produces correct 14-char display string
- format_fingerprint raises ValueError for wrong-length input
"""

from __future__ import annotations

import hashlib

import pytest

from agent.identity import (
    compute_fingerprint,
    compute_peer_id,
    format_fingerprint,
    generate_keypair,
)

# ---------------------------------------------------------------------------
# Test vectors from ADR-0002
# pubkey = sha256(b"vector-N").digest()  for N in 1..5
# ---------------------------------------------------------------------------

VECTORS: list[tuple[str, str, str]] = [
    # (input label, expected peer_id, expected fingerprint_raw)
    (
        "vector-1",
        "3P3aWpCTmgvnAJbQWNfrMNjZJjmcZ8Ue9mhVJ3dvzuiY",
        "3P3aWpCTmgvn",
    ),
    (
        "vector-2",
        "4QimCrVMsaPeUuF7oWT6Ns6zV7rqtXFyXC7SX3edCxbw",
        "4QimCrVMsaPe",
    ),
    (
        "vector-3",
        "3HCtNabzSNKqA1JurHRD2Ro1jaHZM9ZfT4y6GkPQ94s4",
        "3HCtNabzSNKq",
    ),
    (
        "vector-4",
        "B1bF7uxHg23isV1yJcvVGej1rJzCqgcnXEPtXTTnjJxA",
        "B1bF7uxHg23i",
    ),
    (
        "vector-5",
        "AUSKP2Gc4s3hL6DmLNwQBmiLJi4n3iXLTPa9rjfQfR2B",
        "AUSKP2Gc4s3h",
    ),
]


class TestGenerateKeypair:
    def test_returns_two_byte_arrays(self) -> None:
        priv, pub = generate_keypair()
        assert isinstance(priv, bytes)
        assert isinstance(pub, bytes)

    def test_private_key_length(self) -> None:
        priv, _ = generate_keypair()
        assert len(priv) == 32

    def test_public_key_length(self) -> None:
        _, pub = generate_keypair()
        assert len(pub) == 32

    def test_each_call_produces_unique_keypair(self) -> None:
        priv1, pub1 = generate_keypair()
        priv2, pub2 = generate_keypair()
        assert priv1 != priv2
        assert pub1 != pub2


class TestComputePeerId:
    @pytest.mark.parametrize("label, expected_peer_id, _fp", VECTORS)
    def test_known_vectors(self, label: str, expected_peer_id: str, _fp: str) -> None:
        pubkey = hashlib.sha256(label.encode()).digest()
        result = compute_peer_id(pubkey)
        assert result == expected_peer_id, f"Failed for {label}"

    def test_deterministic(self) -> None:
        _, pub = generate_keypair()
        assert compute_peer_id(pub) == compute_peer_id(pub)

    def test_different_keys_produce_different_peer_ids(self) -> None:
        _, pub1 = generate_keypair()
        _, pub2 = generate_keypair()
        assert compute_peer_id(pub1) != compute_peer_id(pub2)


class TestComputeFingerprint:
    @pytest.mark.parametrize("label, _peer_id, expected_fp", VECTORS)
    def test_known_vectors(self, label: str, _peer_id: str, expected_fp: str) -> None:
        pubkey = hashlib.sha256(label.encode()).digest()
        result = compute_fingerprint(pubkey)
        assert result == expected_fp, f"Failed for {label}"

    def test_length_is_12(self) -> None:
        _, pub = generate_keypair()
        assert len(compute_fingerprint(pub)) == 12

    def test_is_prefix_of_peer_id(self) -> None:
        _, pub = generate_keypair()
        peer_id = compute_peer_id(pub)
        fingerprint = compute_fingerprint(pub)
        assert peer_id.startswith(fingerprint)


class TestFormatFingerprint:
    def test_standard_input(self) -> None:
        assert format_fingerprint("abcdefghijkm") == "abcd efgh ijkm"

    def test_format_output_length(self) -> None:
        result = format_fingerprint("abcdefghijkm")
        assert len(result) == 14

    def test_spaces_at_positions_4_and_9(self) -> None:
        result = format_fingerprint("abcdefghijkm")
        assert result[4] == " "
        assert result[9] == " "

    @pytest.mark.parametrize("label, _peer_id, expected_fp", VECTORS)
    def test_known_vector_display(self, label: str, _peer_id: str, expected_fp: str) -> None:
        display = format_fingerprint(expected_fp)
        parts = display.split(" ")
        assert len(parts) == 3
        assert all(len(p) == 4 for p in parts)
        assert "".join(parts) == expected_fp

    def test_raises_on_short_input(self) -> None:
        with pytest.raises(ValueError, match="12"):
            format_fingerprint("short")

    def test_raises_on_long_input(self) -> None:
        with pytest.raises(ValueError, match="12"):
            format_fingerprint("toolongstring!!")

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError, match="12"):
            format_fingerprint("")
