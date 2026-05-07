"""Общие QUIC stream I/O helpers (ADR-0019, ADR-0012).

Используется как в node/network.py (серверная сторона), так и в
client/quic_transport.py (клиентская сторона).

Wire-формат для всех сообщений поверх QUIC-стрима:
    length(u32 BE) || msg_bytes

Сообщения:
  - SessionInit — первый фрейм; JSON, length-prefixed.
  - ForwardEnvelope — последующие фреймы; бинарные.

ADR-0019: MAX_ENVELOPE_SIZE_BYTES enforced при получении.
"""

from __future__ import annotations

import asyncio
import struct

from shared.constants import MAX_ENVELOPE_SIZE_BYTES

# Размер чанка для стриминга тела чанка весов (64 КБ).
WEIGHT_CHUNK_STREAM_CHUNK_SIZE: int = 64 * 1024  # 64 KiB

# Максимальный размер одного weight-chunk фрейма (256 МБ + запас для метаданных).
# Совпадает со spec §5 chunk size. MAX_ENVELOPE_SIZE_BYTES не применяется к весам.
MAX_WEIGHT_CHUNK_SIZE_BYTES: int = 256 * 1024 * 1024 + 4096  # 256 MiB + metadata overhead

# Таймаут ожидания receipt от downloader'а (сек, ADR-0035).
WEIGHT_RECEIPT_TIMEOUT_SEC: int = 30


def encode_frame(msg_bytes: bytes) -> bytes:
    """Упаковать сообщение в length-prefixed фрейм (u32 BE).

    Args:
        msg_bytes: полезная нагрузка.

    Returns:
        bytes: length(u32 BE) || msg_bytes.

    Raises:
        ValueError: если msg_bytes > MAX_ENVELOPE_SIZE_BYTES.
    """
    if len(msg_bytes) > MAX_ENVELOPE_SIZE_BYTES:
        raise ValueError(
            f"Message too large: {len(msg_bytes)} bytes > "
            f"MAX_ENVELOPE_SIZE_BYTES={MAX_ENVELOPE_SIZE_BYTES}"
        )
    return struct.pack(">I", len(msg_bytes)) + msg_bytes


async def read_frame(
    reader: asyncio.StreamReader,
    max_size: int = MAX_ENVELOPE_SIZE_BYTES,
    timeout: float | None = None,
) -> bytes:
    """Прочитать один length-prefixed фрейм из StreamReader.

    Args:
        reader: asyncio.StreamReader QUIC-стрима.
        max_size: максимальный допустимый размер фрейма.
        timeout: таймаут ожидания в секундах (None = без таймаута).

    Returns:
        bytes полезной нагрузки (без length-prefix).

    Raises:
        ValueError: если фрейм > max_size.
        EOFError: если соединение закрыто до получения фрейма.
        asyncio.TimeoutError: если истёк timeout.
    """

    async def _read() -> bytes:
        header = await reader.readexactly(4)
        (length,) = struct.unpack_from(">I", header)
        if length > max_size:
            raise ValueError(
                f"Frame too large: {length} bytes > max_size={max_size}. "
                f"Closing stream (ADR-0019 ENVELOPE_TOO_LARGE)."
            )
        if length == 0:
            return b""
        data = await reader.readexactly(length)
        return data

    try:
        if timeout is not None:
            return await asyncio.wait_for(_read(), timeout=timeout)
        return await _read()
    except asyncio.IncompleteReadError as exc:
        raise EOFError(f"Connection closed mid-frame: {exc}") from exc
