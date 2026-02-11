from __future__ import annotations

from functools import lru_cache
from typing import Iterable


_POLY_REV = 0x82F63B78  # reversed Castagnoli polynomial


@lru_cache(maxsize=1)
def _crc32c_table() -> tuple[int, ...]:
    """Generate the 256-entry CRC32c table."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ _POLY_REV
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


def crc32c(data: bytes, crc: int = 0) -> int:
    """Compute CRC32c over *data*.

    Args:
        data: Bytes to checksum.
        crc: Optional starting CRC (default 0).

    Returns:
        Unsigned 32-bit CRC32c value as an int in [0, 2**32).
    """
    table = _crc32c_table()
    crc ^= 0xFFFFFFFF
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def file_checksum(path: str, *, chunk_size: int = 1 << 20) -> int:
    """Compute CRC32c of a file.

    Uses streaming reads to avoid loading large files fully into memory.
    """
    crc = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            crc = crc32c(chunk, crc)
    return crc
