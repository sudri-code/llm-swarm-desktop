"""Ed25519 utilities + peer_id derivation + envelope signing (ADR-0009).

Инвариант (CLAUDE.md, §4.5 спеки):
    peer_id = base58(sha256(public_key))
    peer_id всегда выводится из public_key, никогда не задаётся снаружи.

Envelope подпись (ADR-0009, §7.2):
    sig = Ed25519_sign(sk, sha256(serialize_envelope_for_signing(env)))
    Цепочка: verify_chain проверяет prev_sig каждого хопа.

Auth domain (ADR-0020):
    AUTH_DOMAIN_TAG — отдельный domain tag для challenge-response auth.
    build_auth_payload / sign_auth / verify_auth — детерминированный payload.

Нет глобальных синглтонов. Чистые функции без побочных эффектов.
"""

import hashlib
import struct

import base58
from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

from shared.protocol import ForwardEnvelope, serialize_envelope_for_signing

# Документирующая константа: алгоритм хеширования public_key → peer_id (§4.5).
PEER_ID_PREFIX_HASH = "sha256"

# Domain tag для challenge-response auth (ADR-0020).
# "llm-swarm/v1/auth" — 17 байт; поле в payload считается фиксированным 17B
# (аналогично DOMAIN_TAG forward = 20B; ADR описывает «16B» как цель, но строка 17 chars).
AUTH_DOMAIN_TAG: bytes = b"llm-swarm/v1/auth"
assert len(AUTH_DOMAIN_TAG) == 17, (
    f"AUTH_DOMAIN_TAG must be exactly 17 bytes, got {len(AUTH_DOMAIN_TAG)}"
)


class ChainVerificationError(Exception):
    """Ошибка верификации цепочки подписей.

    Атрибут hop_index указывает на первый виновный хоп.
    """

    def __init__(self, hop_index: int, reason: str) -> None:
        self.hop_index = hop_index
        self.reason = reason
        super().__init__(f"Chain verification failed at hop {hop_index}: {reason}")


def generate_seed() -> bytes:
    """Вернуть 32 случайных байта — Ed25519 seed."""
    import nacl.utils

    return nacl.utils.random(32)


def load_signing_key(seed: bytes) -> SigningKey:
    """Восстановить SigningKey из raw seed (ровно 32 байта).

    Raises:
        ValueError: если len(seed) != 32.
    """
    if len(seed) != 32:
        raise ValueError(f"Ed25519 seed must be exactly 32 bytes, got {len(seed)}")
    return SigningKey(seed)


def public_key_bytes(signing_key: SigningKey) -> bytes:
    """Вернуть 32 байта Ed25519 public key."""
    return bytes(signing_key.verify_key)


def peer_id_from_public_key(public_key: bytes) -> str:
    """Вычислить peer_id = base58(sha256(public_key)).

    Инвариант: формула зафиксирована в §4.5 спеки — не менять.
    """
    digest = hashlib.sha256(public_key).digest()
    return base58.b58encode(digest).decode()


def sign(message: bytes, signing_key: SigningKey) -> bytes:
    """Вернуть detached Ed25519 подпись (64 байта)."""
    return bytes(signing_key.sign(message).signature)


def verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Проверить Ed25519 подпись.

    Возвращает True если подпись валидна, False в любом ином случае.
    Не поднимает исключений — мягкая семантика для use-case spot-check.

    Raises:
        ValueError: если public_key имеет некорректный размер (не 32 байта).
                    Остальные ошибки поглощаются и возвращают False.
    """
    if len(public_key) != 32:
        raise ValueError(f"Ed25519 public key must be exactly 32 bytes, got {len(public_key)}")
    try:
        vk = VerifyKey(public_key)
        vk.verify(message, signature)
        return True
    except BadSignatureError:
        return False


def sign_envelope(sk: SigningKey, env: ForwardEnvelope) -> bytes:
    """Подписать ForwardEnvelope ключом ноды.

    Формула (ADR-0009):
        sig = Ed25519_sign(sk, sha256(serialize_envelope_for_signing(env)))

    Args:
        sk: Ed25519 SigningKey ноды.
        env: ForwardEnvelope для подписи.

    Returns:
        64-байтовая detached Ed25519-подпись.
    """
    payload = serialize_envelope_for_signing(env)
    digest = hashlib.sha256(payload).digest()
    return bytes(sk.sign(digest).signature)


def verify_envelope(pk: VerifyKey, env: ForwardEnvelope, sig: bytes) -> bool:
    """Проверить подпись ForwardEnvelope.

    Args:
        pk: Ed25519 VerifyKey ноды-отправителя.
        env: ForwardEnvelope.
        sig: 64-байтовая Ed25519-подпись.

    Returns:
        True если подпись валидна, False иначе.
        Не поднимает исключений при невалидной подписи.
    """
    payload = serialize_envelope_for_signing(env)
    digest = hashlib.sha256(payload).digest()
    try:
        pk.verify(digest, sig)
        return True
    except BadSignatureError:
        return False


def build_auth_payload(
    challenge: bytes,
    peer_id_bytes: bytes,
    tracker_url: str,
    issued_at_ms: int,
) -> bytes:
    """Детерминированный байтстрим для подписи challenge-response (ADR-0020).

    Структура:
        AUTH_DOMAIN_TAG(17) || challenge(32) || peer_id_bytes(32)
        || tracker_url_hash(32) || issued_at(u64 BE)

    Args:
        challenge: 32 байта challenge от трекера (secrets.token_bytes(32)).
        peer_id_bytes: 32 байта = sha256(public_key) raw (НЕ base58).
        tracker_url: каноническая форма URL трекера (lowercase scheme+host, явный порт,
                     без trailing slash).
        issued_at_ms: unix timestamp в миллисекундах (момент подписи нодой), u64 BE.

    Returns:
        Байтовый буфер для Ed25519-подписи.
    """
    if len(challenge) != 32:
        raise ValueError(f"challenge must be exactly 32 bytes, got {len(challenge)}")
    if len(peer_id_bytes) != 32:
        raise ValueError(f"peer_id_bytes must be exactly 32 bytes, got {len(peer_id_bytes)}")
    tracker_url_hash = hashlib.sha256(tracker_url.encode("utf-8")).digest()
    return (
        AUTH_DOMAIN_TAG
        + challenge
        + peer_id_bytes
        + tracker_url_hash
        + struct.pack(">Q", issued_at_ms)
    )


def sign_auth(
    signing_key: SigningKey,
    challenge: bytes,
    peer_id_bytes: bytes,
    tracker_url: str,
    issued_at_ms: int,
) -> bytes:
    """Подписать challenge-response payload (ADR-0020).

    Returns:
        64 байта Ed25519-подписи.
    """
    payload = build_auth_payload(challenge, peer_id_bytes, tracker_url, issued_at_ms)
    return bytes(signing_key.sign(payload).signature)


def verify_auth(
    verify_key: VerifyKey,
    signature: bytes,
    challenge: bytes,
    peer_id_bytes: bytes,
    tracker_url: str,
    issued_at_ms: int,
) -> bool:
    """Проверить подпись challenge-response (ADR-0020).

    Returns:
        True если подпись валидна, False иначе. Не поднимает исключений.
    """
    if len(signature) != 64:
        return False
    payload = build_auth_payload(challenge, peer_id_bytes, tracker_url, issued_at_ms)
    try:
        verify_key.verify(payload, signature)
        return True
    except BadSignatureError:
        return False


# Domain tag для strike report (ADR-0024).
STRIKE_DOMAIN_TAG: bytes = b"llm-swarm/v1/strike"
assert len(STRIKE_DOMAIN_TAG) == 19, (
    f"STRIKE_DOMAIN_TAG must be exactly 19 bytes, got {len(STRIKE_DOMAIN_TAG)}"
)

# Domain tag для accounting report (ADR-0022).
ACCOUNTING_DOMAIN_TAG: bytes = b"llm-swarm/v1/accounting"
assert len(ACCOUNTING_DOMAIN_TAG) == 23, (
    f"ACCOUNTING_DOMAIN_TAG must be exactly 23 bytes, got {len(ACCOUNTING_DOMAIN_TAG)}"
)


def build_strike_payload(evidence_canonical_bytes: bytes) -> bytes:
    """Детерминированный байтстрим для подписи strike report (ADR-0024).

    Структура:
        STRIKE_DOMAIN_TAG(19) || evidence_canonical_bytes

    Args:
        evidence_canonical_bytes: детерминированная сериализация evidence
                                   (canonical JSON UTF-8 sorted keys).

    Returns:
        Байтовый буфер для Ed25519-подписи.
    """
    return STRIKE_DOMAIN_TAG + evidence_canonical_bytes


def sign_strike(signing_key: SigningKey, evidence_canonical_bytes: bytes) -> bytes:
    """Подписать strike report evidence (ADR-0024).

    Returns:
        64 байта Ed25519-подписи.
    """
    payload = build_strike_payload(evidence_canonical_bytes)
    return bytes(signing_key.sign(payload).signature)


def verify_strike(verify_key: VerifyKey, signature: bytes, evidence_canonical_bytes: bytes) -> bool:
    """Проверить подпись strike report (ADR-0024).

    Returns:
        True если подпись валидна, False иначе. Не поднимает исключений.
    """
    if len(signature) != 64:
        return False
    payload = build_strike_payload(evidence_canonical_bytes)
    try:
        verify_key.verify(payload, signature)
        return True
    except BadSignatureError:
        return False


def build_accounting_payload(accounting_canonical_bytes: bytes) -> bytes:
    """Детерминированный байтстрим для подписи accounting report (ADR-0022).

    Структура:
        ACCOUNTING_DOMAIN_TAG(23) || accounting_canonical_bytes

    Args:
        accounting_canonical_bytes: детерминированная сериализация body
                                     без поля client_signature (canonical JSON).

    Returns:
        Байтовый буфер для Ed25519-подписи.
    """
    return ACCOUNTING_DOMAIN_TAG + accounting_canonical_bytes


def sign_accounting(signing_key: SigningKey, accounting_canonical_bytes: bytes) -> bytes:
    """Подписать accounting report (ADR-0022).

    Returns:
        64 байта Ed25519-подписи.
    """
    payload = build_accounting_payload(accounting_canonical_bytes)
    return bytes(signing_key.sign(payload).signature)


def verify_accounting(
    verify_key: VerifyKey, signature: bytes, accounting_canonical_bytes: bytes
) -> bool:
    """Проверить подпись accounting report (ADR-0022).

    Returns:
        True если подпись валидна, False иначе. Не поднимает исключений.
    """
    if len(signature) != 64:
        return False
    payload = build_accounting_payload(accounting_canonical_bytes)
    try:
        verify_key.verify(payload, signature)
        return True
    except BadSignatureError:
        return False


# Domain tag для chunk receipt (ADR-0035).
CHUNK_RECEIPT_DOMAIN_TAG: bytes = b"llm-swarm/v1/chunk-receipt"
assert len(CHUNK_RECEIPT_DOMAIN_TAG) == 26, (
    f"CHUNK_RECEIPT_DOMAIN_TAG must be exactly 26 bytes, got {len(CHUNK_RECEIPT_DOMAIN_TAG)}"
)


def build_chunk_receipt_payload(
    chunk_id: str,
    sha256_hex: str,
    downloader_peer_id: str,
    uploader_peer_id: str,
    bytes_received: int,
    ts: int,
) -> bytes:
    """Детерминированный байтстрим для подписи chunk receipt (ADR-0035).

    Структура:
        CHUNK_RECEIPT_DOMAIN_TAG(26)
        || chunk_id_hash(32)          — sha256(chunk_id.encode())
        || sha256_raw(32)             — bytes.fromhex(sha256_hex)
        || downloader_peer_id_hash(32) — sha256(downloader_peer_id.encode())
        || uploader_peer_id_hash(32)  — sha256(uploader_peer_id.encode())
        || bytes_received(u64 BE)
        || ts(u64 BE)

    Args:
        chunk_id: строковый идентификатор чанка.
        sha256_hex: hex SHA256 чанка (64 символа).
        downloader_peer_id: peer_id получателя.
        uploader_peer_id: peer_id отправителя.
        bytes_received: размер полученных данных в байтах.
        ts: unix timestamp в секундах.

    Returns:
        Байтовый буфер для Ed25519-подписи.
    """
    chunk_id_hash = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    sha256_raw = bytes.fromhex(sha256_hex)
    downloader_hash = hashlib.sha256(downloader_peer_id.encode("utf-8")).digest()
    uploader_hash = hashlib.sha256(uploader_peer_id.encode("utf-8")).digest()
    return (
        CHUNK_RECEIPT_DOMAIN_TAG
        + chunk_id_hash
        + sha256_raw
        + downloader_hash
        + uploader_hash
        + struct.pack(">Q", bytes_received)
        + struct.pack(">Q", ts)
    )


def sign_chunk_receipt(
    signing_key: SigningKey,
    chunk_id: str,
    sha256_hex: str,
    downloader_peer_id: str,
    uploader_peer_id: str,
    bytes_received: int,
    ts: int,
) -> bytes:
    """Подписать chunk receipt (ADR-0035).

    Returns:
        64 байта Ed25519-подписи.
    """
    payload = build_chunk_receipt_payload(
        chunk_id, sha256_hex, downloader_peer_id, uploader_peer_id, bytes_received, ts
    )
    return bytes(signing_key.sign(payload).signature)


def verify_chunk_receipt(
    verify_key: VerifyKey,
    signature: bytes,
    chunk_id: str,
    sha256_hex: str,
    downloader_peer_id: str,
    uploader_peer_id: str,
    bytes_received: int,
    ts: int,
) -> bool:
    """Проверить подпись chunk receipt (ADR-0035).

    Returns:
        True если подпись валидна, False иначе. Не поднимает исключений.
    """
    if len(signature) != 64:
        return False
    payload = build_chunk_receipt_payload(
        chunk_id, sha256_hex, downloader_peer_id, uploader_peer_id, bytes_received, ts
    )
    try:
        verify_key.verify(payload, signature)
        return True
    except BadSignatureError:
        return False


def verify_chain(
    envelopes: list[tuple[ForwardEnvelope, bytes]],
    pubkeys: list[VerifyKey],
) -> None:
    """Проверить цепочку подписанных envelope'ов.

    Для каждого хопа i:
      1. Проверяет подпись sig[i] ключом pubkeys[i].
      2. Для i > 0: проверяет, что env[i].prev_sig == sigs[i-1].
      3. Для i == 0: проверяет, что env[i].prev_sig == b"\\x00" * 64.

    Args:
        envelopes: список кортежей (ForwardEnvelope, sig_bytes) в порядке хопов.
        pubkeys: список VerifyKey, по одному на каждый хоп.

    Raises:
        ValueError: если длины envelopes и pubkeys не совпадают.
        ChainVerificationError: при первом нарушении с указанием hop_index.
    """
    if len(envelopes) != len(pubkeys):
        raise ValueError(
            f"envelopes and pubkeys must have equal length: {len(envelopes)} != {len(pubkeys)}"
        )

    _ZERO_SIG = b"\x00" * 64

    for i, ((env, sig), pk) in enumerate(zip(envelopes, pubkeys, strict=False)):
        # Проверка prev_sig
        if i == 0:
            if env.prev_sig != _ZERO_SIG:
                raise ChainVerificationError(
                    hop_index=i,
                    reason="hop 0 prev_sig must be zero bytes, got non-zero",
                )
        else:
            prev_sig = envelopes[i - 1][1]
            if env.prev_sig != prev_sig:
                raise ChainVerificationError(
                    hop_index=i,
                    reason=f"prev_sig mismatch: expected sig of hop {i - 1}, got different bytes",
                )

        # Проверка Ed25519-подписи текущего хопа
        if not verify_envelope(pk, env, sig):
            raise ChainVerificationError(
                hop_index=i,
                reason="Ed25519 signature verification failed",
            )
