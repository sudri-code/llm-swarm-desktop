"""TLS peer_id binding utilities (ADR-0028).

extract_peer_id_from_cert — парсит DER X.509 сертификат, извлекает Ed25519 SPKI,
возвращает peer_id = base58(sha256(raw_32B_pubkey)).

Инвариант §4.5 / CLAUDE.md:
    peer_id = base58(sha256(raw_ed25519_public_key_bytes))

Используется в:
    client/quic_transport.py — проверка TLS peer_id после QUIC handshake.
    node/network.py — (mTLS, если клиент шлёт cert).

Зависимость: cryptography >= 41 (уже в transitive deps PyNaCl + aioquic).
"""

from __future__ import annotations

import hashlib

import base58


class TLSPeerIdError(Exception):
    """Ошибка извлечения peer_id из TLS-сертификата.

    Поднимается при:
    - нечитаемом/невалидном DER cert
    - non-Ed25519 публичном ключе в cert (например RSA или ECDSA)
    """


def extract_peer_id_from_cert(raw_der: bytes) -> str:
    """Парсит X.509 DER cert, извлекает Ed25519 SPKI raw 32B, возвращает base58(sha256).

    Алгоритм §4.5 / CLAUDE.md:
        peer_id = base58(sha256(raw_ed25519_public_key_bytes))

    Args:
        raw_der: DER-encoded X.509 сертификат (bytes).

    Returns:
        peer_id строка (base58-encoded).

    Raises:
        TLSPeerIdError: если cert нечитаем или public key не Ed25519.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        from cryptography.x509 import load_der_x509_certificate
    except ImportError as exc:
        raise TLSPeerIdError(
            "cryptography package required for TLS peer_id binding (ADR-0028)"
        ) from exc

    try:
        cert = load_der_x509_certificate(raw_der)
    except Exception as exc:
        raise TLSPeerIdError(f"Failed to parse DER certificate: {exc}") from exc

    try:
        pub_key = cert.public_key()
    except Exception as exc:
        raise TLSPeerIdError(f"Failed to extract public key from certificate: {exc}") from exc

    if not isinstance(pub_key, Ed25519PublicKey):
        key_type = type(pub_key).__name__
        raise TLSPeerIdError(
            f"Certificate uses non-Ed25519 key: {key_type}. "
            "Node must use Ed25519 self-signed cert (node/network.py:generate_self_signed_cert)."
        )

    try:
        raw_pk = pub_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    except Exception as exc:
        raise TLSPeerIdError(f"Failed to serialize Ed25519 public key to raw bytes: {exc}") from exc

    digest = hashlib.sha256(raw_pk).digest()
    return base58.b58encode(digest).decode()
