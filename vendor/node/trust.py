"""Stage 4: trust agent — keypair load/generate, peer_id derivation, Ed25519 identity.

Поглощает node/identity.py (ADR-0007): все криптографические операции ноды
сосредоточены здесь.

Текущий этап (S4-A1): keypair и peer_id (перенесено из node/identity.py).
Challenge-response подписание добавилось в S4-A2.
Strike-evidence подписи добавлены в S4-C3.

Инвариант: identity.key — раз сгенерирован, неизменен. Повреждённый файл → ошибка, не
молчаливая перегенерация.

Формат хранения: raw 32-byte seed, chmod 600 (ADR-0004).
peer_id = base58(sha256(public_key)) — строго так (CLAUDE.md).

S4-C3: StrikeLog — локальный JSONL-лог аномалий цепочки и инференса.
Лог пишется нодой для post-mortem; не отправляется в трекер автоматически.
Только клиент с активным session_id может репортить страйки в трекер (ADR-0024).
sign_evidence — helper для подписи evidence blob перед возможной отправкой (Post-MVP).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import base58
from nacl.signing import SigningKey

from shared.crypto import (
    generate_seed,
    load_signing_key,
    peer_id_from_public_key,
    public_key_bytes,
    sign_strike,
)

logger = logging.getLogger(__name__)

_SEED_SIZE = 32
_KEY_FILENAME = "identity.key"


class IdentityError(OSError):
    """Ошибка загрузки или генерации identity (повреждённый файл, неверный размер)."""


@dataclass(frozen=True)
class Identity:
    """Неизменяемый контейнер Ed25519-идентичности ноды."""

    signing_key: SigningKey
    public_key: bytes  # raw 32 bytes
    peer_id: str  # base58(sha256(public_key))
    public_key_b58: str  # base58(public_key) — для протокола RegisterRequest

    def peer_id_bytes(self) -> bytes:
        """Вернуть 32 байта sha256(public_key) raw — для build_auth_payload (ADR-0020).

        peer_id_bytes = sha256(public_key), не base58.
        Инвариант: peer_id = base58(peer_id_bytes()) — строго так.
        """
        import hashlib

        return hashlib.sha256(self.public_key).digest()


def load_or_create_identity(data_dir: Path) -> Identity:
    """Загрузить или сгенерировать Ed25519 keypair.

    Файл: <data_dir>/identity.key — raw 32 bytes seed, chmod 600.

    Если файла нет — генерируем, пишем, логируем INFO.
    Если файл есть — читаем, проверяем размер ровно 32 байта, логируем INFO.
    Повреждённый файл (не 32 байта) → IdentityError, не перегенерация.

    Args:
        data_dir: Каталог, в котором хранится identity.key. Должен существовать.

    Returns:
        Identity — неизменяемый датакласс с ключами и peer_id.

    Raises:
        IdentityError: Если файл повреждён (неверный размер).
    """
    key_path = data_dir / _KEY_FILENAME

    if key_path.exists():
        seed = key_path.read_bytes()
        if len(seed) != _SEED_SIZE:
            raise IdentityError(
                f"identity.key is corrupted: expected {_SEED_SIZE} bytes, got {len(seed)}. "
                f"Path: {key_path}. Remove the file manually to regenerate."
            )
        signing_key = load_signing_key(seed)
        pk = public_key_bytes(signing_key)
        pid = peer_id_from_public_key(pk)
        logger.info("loaded identity, peer_id=%s", pid)
    else:
        seed = generate_seed()
        key_path.write_bytes(seed)
        _set_permissions_600(key_path)
        signing_key = load_signing_key(seed)
        pk = public_key_bytes(signing_key)
        pid = peer_id_from_public_key(pk)
        logger.info("generated new identity at %s, peer_id=%s", key_path, pid)

    pk_b58 = base58.b58encode(pk).decode()
    return Identity(
        signing_key=signing_key,
        public_key=pk,
        peer_id=pid,
        public_key_b58=pk_b58,
    )


def _set_permissions_600(path: Path) -> None:
    """Установить права 0600 на posix; на Windows — пропустить без ошибки."""
    if sys.platform != "win32":
        os.chmod(path, 0o600)


# ---------------------------------------------------------------------------
# S4-C3: StrikeLog — локальный JSONL-лог аномалий цепочки и инференса
# ---------------------------------------------------------------------------

_STRIKE_LOG_FILENAME = "strikes.log"


class StrikeLog:
    """Локальный JSONL-лог аномалий цепочки и инференса (S4-C3).

    Записывает события в <data_dir>/strikes.log в формате JSONL (одна запись = одна строка JSON).
    Не отправляет ничего в трекер — это только post-mortem лог.
    Только клиент с активным session_id может репортить страйки в трекер (ADR-0024).

    Использование:
        strike_log = StrikeLog(data_dir=Path("~/.llm-swarm").expanduser())
        strike_log.record_chain_break(session_id, hop_index, prev_sig_hex, "bad_signature")
        strike_log.record_inference_anomaly(session_id, (0, 16), "NaN in input activations")
    """

    def __init__(self, data_dir: Path) -> None:
        """Инициализировать StrikeLog.

        Args:
            data_dir: Каталог, в котором создаётся strikes.log.
                      Каталог должен существовать (создаётся load_or_create_identity).
        """
        self._log_path = data_dir / _STRIKE_LOG_FILENAME

    def record_chain_break(
        self,
        session_id: Any,
        hop_index: int,
        prev_envelope: Any,
        reason: str,
    ) -> None:
        """Записать событие обрыва цепочки подписей.

        Вызывается, когда нода обнаруживает невалидную подпись prev_envelope
        от предыдущего хопа.

        Args:
            session_id: UUID сессии (или строка).
            hop_index: Индекс хопа, на котором обнаружен обрыв.
            prev_envelope: Объект или строка для идентификации env (e.g. hex подписи).
            reason: Строковое описание причины.
        """
        entry: dict[str, Any] = {
            "event": "chain_break",
            "timestamp": time.time(),
            "session_id": str(session_id),
            "hop_index": hop_index,
            "prev_envelope": str(prev_envelope),
            "reason": reason,
        }
        self._append(entry)
        logger.warning(
            "StrikeLog: chain_break session=%s hop=%d reason=%s",
            session_id,
            hop_index,
            reason,
        )

    def record_inference_anomaly(
        self,
        session_id: Any,
        layer_range: tuple[int, int],
        anomaly: str,
    ) -> None:
        """Записать событие аномалии инференса (NaN/Inf в активациях).

        Args:
            session_id: UUID сессии (или строка).
            layer_range: Диапазон слоёв (start, end) в котором обнаружена аномалия.
            anomaly: Строковое описание аномалии.
        """
        entry: dict[str, Any] = {
            "event": "inference_anomaly",
            "timestamp": time.time(),
            "session_id": str(session_id),
            "layer_start": layer_range[0],
            "layer_end": layer_range[1],
            "anomaly": anomaly,
        }
        self._append(entry)
        logger.warning(
            "StrikeLog: inference_anomaly session=%s layers=%s anomaly=%s",
            session_id,
            layer_range,
            anomaly,
        )

    def _append(self, entry: dict[str, Any]) -> None:
        """Атомарно дописать запись в JSONL-файл."""
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line)


# ---------------------------------------------------------------------------
# S4-C3: sign_evidence — helper для подписи evidence blob (Post-MVP)
# ---------------------------------------------------------------------------


def sign_evidence(signing_key: SigningKey, evidence_canonical: bytes) -> bytes:
    """Подписать evidence blob ключом ноды (S4-C3, Post-MVP helper).

    Обёртка над shared.crypto.sign_strike с domain tag "llm-swarm/v1/strike" (ADR-0024).
    Используется, когда нода хочет приложить свою подпись к evidence
    перед передачей клиенту для дальнейшего репорта.

    Args:
        signing_key: Ed25519 SigningKey ноды.
        evidence_canonical: Детерминированная сериализация evidence
                            (canonical JSON UTF-8, sorted keys).

    Returns:
        64 байта Ed25519-подписи с domain tag "llm-swarm/v1/strike".
    """
    return sign_strike(signing_key, evidence_canonical)
