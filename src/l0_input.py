"""L0 input parsing and normalization (OPEN-SZ-01).

Responsibilities (manual ch.7.2):
  - file format parsing, image/video decoding
  - size & colorspace unification (longest side -> 512 for model input)
  - content hashing (sha256 + perceptual hash) and basic validity checks
  - NO detection logic lives here.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

MODEL_SIDE = 512  # longest side for model input, per manual appendix C


@dataclass
class ParsedInput:
    """Normalized representation handed to L1/L2."""
    ok: bool
    error: str | None = None
    media_type: str = "image"            # image | video
    raw_bytes: int = 0
    sha256: str = ""
    phash: str = ""
    orig_width: int = 0
    orig_height: int = 0
    png_chunks: list = field(default_factory=list)   # [(type, length), ...] for PNG
    exif_present: bool = False
    bgr: np.ndarray | None = None        # normalized (<=MODEL_SIDE long side) for model input
    bgr_full: np.ndarray | None = None   # full-resolution decode (used by L3)
    resize_scale: float = 1.0


def _png_chunk_list(data: bytes) -> list:
    """Walk a PNG byte stream and record (chunk_type, length)."""
    chunks = []
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return chunks
    off, n = 8, len(data)
    while off + 8 <= n:
        length = int.from_bytes(data[off:off + 4], "big")
        ctype = data[off + 4:off + 8].decode("latin-1")
        chunks.append((ctype, length))
        off += 12 + length  # len + type + data + crc
        if ctype == "IEND":
            break
    return chunks


def perceptual_hash(gray: np.ndarray, hash_size: int = 16) -> str:
    """aHash on 16x16 downsample -> 256-bit hex string (no external deps)."""
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).astype(np.uint8).flatten()
    return "".join(f"{int(''.join(str(int(b)) for b in bits[i:i + 4]), 2):x}"
                   for i in range(0, len(bits), 4))


def parse_bytes(data: bytes, keep_full: bool = True) -> ParsedInput:
    """Parse an image already held in memory (used by the robustness matrix).

    Container features are read from these bytes, so a re-encoded input correctly
    loses its original PNG-chunk fingerprints - which is exactly the behaviour the
    robustness experiment needs to measure.
    """
    out = ParsedInput(ok=False, media_type="image")
    out.raw_bytes = len(data)
    out.sha256 = hashlib.sha256(data).hexdigest()
    out.png_chunks = _png_chunk_list(data)

    try:
        pil = Image.open(io.BytesIO(data))
        out.exif_present = bool(pil.getexif()) or b"eXIf" in data or b"iTXt" in data
        pil = pil.convert("RGB")
        rgb = np.asarray(pil)
    except Exception as e:
        out.error = f"decode failed: {e}"
        return out

    out.orig_height, out.orig_width = rgb.shape[:2]
    bgr_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    scale = 1.0
    long_side = max(out.orig_width, out.orig_height)
    if long_side > MODEL_SIDE:
        scale = MODEL_SIDE / long_side
        bgr = cv2.resize(bgr_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        bgr = bgr_full.copy()
    out.resize_scale = scale
    out.bgr = bgr
    out.bgr_full = bgr_full if keep_full else None
    out.phash = perceptual_hash(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    out.ok = True
    return out


def parse_image(path: str, keep_full: bool = True) -> ParsedInput:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        out = ParsedInput(ok=False, media_type="image")
        out.error = f"read failed: {e}"
        return out
    return parse_bytes(data, keep_full=keep_full)
