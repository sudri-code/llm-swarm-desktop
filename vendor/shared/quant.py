"""Stage 2: blockwise INT8 квантизация активаций.

Wire-формат activations_blob (ADR-0009, §4.2):
    ndim       : u8
    shape      : u16[ndim]             big-endian
    num_blocks : u32                   big-endian
    scales     : fp16[num_blocks]      little-endian (numpy/torch-совместимо)
    quantized  : i8[total_padded]      row-major

Блок = 64 элемента. Если total_elements % 64 != 0, последний блок паддится
нулями до 64; num_blocks = ceil(total / 64).
scale = max(|x_block|) / 127  (fp16). При scale == 0 → scale = 1.
Симметричная квантизация: q = round(clip(x / scale, -127, 127)).
"""

from __future__ import annotations

import math
import struct

import numpy as np
import torch

_BLOCK_SIZE = 64
_SCALE_DTYPE = np.float16
_QUANT_DTYPE = np.int8


def quantize_blockwise_int8(tensor: torch.Tensor) -> bytes:
    """Сериализовать тензор в activations_blob (blockwise INT8).

    Args:
        tensor: тензор произвольного shape, ожидаемый layout [batch, seq, hidden].
                Должен быть вещественным (float32/float16/bfloat16).

    Returns:
        Байты в wire-формате ADR-0009.
    """
    shape = tensor.shape
    ndim = len(shape)
    if ndim == 0:
        raise ValueError("Scalar tensors are not supported")

    # Приводим к float32 для расчётов, затем к numpy row-major
    flat: np.ndarray = tensor.detach().cpu().float().numpy().ravel()  # row-major (C-order)
    total = flat.size
    num_blocks = math.ceil(total / _BLOCK_SIZE)
    total_padded = num_blocks * _BLOCK_SIZE

    # Паддинг нулями до кратного блоку
    if total_padded > total:
        flat = np.concatenate([flat, np.zeros(total_padded - total, dtype=np.float32)])

    # Разбивка на блоки [num_blocks, BLOCK_SIZE]
    blocks = flat.reshape(num_blocks, _BLOCK_SIZE)

    # Вычисление scale per block (fp16); при scale==0 → scale=1
    abs_max = np.abs(blocks).max(axis=1)  # float32, shape [num_blocks]
    scales_f32 = abs_max / 127.0
    scales_f32 = np.where(scales_f32 == 0.0, 1.0, scales_f32)
    scales_f16 = scales_f32.astype(_SCALE_DTYPE)

    # Квантизация: q = round(clip(x / scale, -127, 127))
    # Используем fp16-конвертированный scale (как при сериализации),
    # чтобы квантизация и деквантизация использовали одно и то же значение.
    scales_for_quant = scales_f16.astype(np.float32)
    q_f32 = blocks / scales_for_quant[:, np.newaxis]
    q_f32 = np.clip(q_f32, -127.0, 127.0)
    quantized = np.round(q_f32).astype(_QUANT_DTYPE)

    # Сборка wire-формата
    # ndim: u8
    header = struct.pack(">B", ndim)
    # shape: u16[ndim] big-endian
    header += struct.pack(f">{ndim}H", *shape)
    # num_blocks: u32 big-endian
    header += struct.pack(">I", num_blocks)
    # scales: fp16[num_blocks] little-endian (tobytes даёт native; numpy float16 — IEEE 754)
    scales_bytes = scales_f16.astype("<f2").tobytes()
    # quantized: i8[total_padded] row-major
    quant_bytes = quantized.tobytes()

    return header + scales_bytes + quant_bytes


def dequantize_blockwise_int8(blob: bytes) -> torch.Tensor:
    """Десериализовать activations_blob обратно в float32-тензор.

    Деквантизованный тензор будет float32.
    Нулевой паддинг последнего блока отбрасывается по оригинальному shape.
    """
    offset = 0

    # ndim: u8
    (ndim,) = struct.unpack_from(">B", blob, offset)
    offset += 1

    # shape: u16[ndim] big-endian
    shape_vals = struct.unpack_from(f">{ndim}H", blob, offset)
    offset += 2 * ndim
    shape = tuple(shape_vals)
    total = math.prod(shape)
    num_blocks = math.ceil(total / _BLOCK_SIZE)
    total_padded = num_blocks * _BLOCK_SIZE

    # num_blocks: u32 big-endian
    (num_blocks_wire,) = struct.unpack_from(">I", blob, offset)
    offset += 4
    if num_blocks_wire != num_blocks:
        raise ValueError(f"num_blocks mismatch: wire={num_blocks_wire}, computed={num_blocks}")

    # scales: fp16[num_blocks] little-endian
    scales_size = num_blocks * 2  # 2 bytes per fp16
    scales_f16 = np.frombuffer(blob[offset : offset + scales_size], dtype="<f2").astype(np.float32)
    offset += scales_size

    # quantized: i8[total_padded]
    quant_size = total_padded
    quantized = np.frombuffer(blob[offset : offset + quant_size], dtype=_QUANT_DTYPE).copy()

    # Деквантизация: x = q * scale
    quantized_f32 = quantized.reshape(num_blocks, _BLOCK_SIZE).astype(np.float32)
    dequant = quantized_f32 * scales_f16[:, np.newaxis]

    # Обрезаем паддинг и возвращаем оригинальный shape
    flat = dequant.ravel()[:total]
    tensor = torch.from_numpy(flat.reshape(shape).astype(np.float32))
    return tensor
