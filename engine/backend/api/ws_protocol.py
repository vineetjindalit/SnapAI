"""
backend/api/ws_protocol.py — RFC 6455 WebSocket frame primitives.

Pure-stdlib helpers extracted from server.py so the protocol class stays
focused on orchestration. No external deps; tested against Chrome,
Firefox, Safari WebSocket clients.
"""
from __future__ import annotations

import base64
import hashlib
from typing import Optional, Tuple

WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_accept_key(key: str) -> str:
    """Compute the Sec-WebSocket-Accept value for the upgrade handshake."""
    combined = (key.strip() + WS_MAGIC).encode()
    return base64.b64encode(hashlib.sha1(combined).digest()).decode()


def ws_parse_frame(data: bytes) -> Tuple[Optional[int], bytes]:
    """Parse a single inbound WS frame. Returns (opcode, payload).

    Returns (None, b'') if `data` is too short for a header.
    """
    if len(data) < 2:
        return None, b""
    b0, b1 = data[0], data[1]
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    offset = 2
    if length == 126:
        if len(data) < 4: return opcode, b""
        length = int.from_bytes(data[2:4], "big")
        offset = 4
    elif length == 127:
        if len(data) < 10: return opcode, b""
        length = int.from_bytes(data[2:10], "big")
        offset = 10
    mask_key = b""
    if masked:
        mask_key = data[offset:offset + 4]
        offset += 4
    payload = bytearray(data[offset:offset + length])
    if masked:
        for i in range(len(payload)):
            payload[i] ^= mask_key[i % 4]
    return opcode, bytes(payload)


def ws_frame_consumed_bytes(buf: bytes) -> int:
    """Return how many bytes a complete frame at the head of `buf` occupies,
    or 0 if the frame is incomplete. Used by the protocol to advance its
    incoming buffer."""
    if len(buf) < 2:
        return 0
    b1 = buf[1]
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    offset = 2
    if length == 126:
        if len(buf) < 4: return 0
        offset = 4
        length = int.from_bytes(buf[2:4], "big")
    elif length == 127:
        if len(buf) < 10: return 0
        offset = 10
        length = int.from_bytes(buf[2:10], "big")
    if masked:
        offset += 4
    end = offset + length
    return end if len(buf) >= end else 0


def ws_build_frame(data: bytes, opcode: int = 0x1) -> bytes:
    """Build an unmasked outbound WebSocket frame (server → client)."""
    length = len(data)
    header = bytes([0x80 | opcode])
    if length <= 125:
        header += bytes([length])
    elif length <= 65535:
        header += bytes([126]) + length.to_bytes(2, "big")
    else:
        header += bytes([127]) + length.to_bytes(8, "big")
    return header + data
