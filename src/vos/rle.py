"""COCO-compatible run-length encoding, implemented in pure NumPy.

Why not just ``pycocotools``?  It needs a C toolchain at install time and is
only used for (de)serialising masks.  Implementing the format directly keeps
the serving image small and lets us decode SA-V annotations anywhere.

Format (identical to pycocotools)
---------------------------------
* The mask is flattened in **column-major (Fortran) order**.
* ``counts`` alternates run lengths of 0s and 1s, always starting with 0s.
* *Uncompressed* RLE stores ``counts`` as a list of ints (what our API returns).
* *Compressed* RLE (SA-V / COCO json) stores ``counts`` as an ASCII string
  using a LEB128-like 5-bit variable-length code with delta encoding.
"""

from __future__ import annotations

import numpy as np


def encode(mask: np.ndarray) -> dict:
    """Encode a (H, W) bool mask into uncompressed COCO RLE ``{"size": [h, w], "counts": [...]}``."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    h, w = mask.shape
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return {"size": [h, w], "counts": []}
    # positions where the value changes
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(boundaries).tolist()
    if flat[0]:  # counts must start with a run of zeros
        runs = [0, *runs]
    return {"size": [int(h), int(w)], "counts": [int(r) for r in runs]}


def decode(rle: dict) -> np.ndarray:
    """Decode uncompressed (list) or compressed (str) COCO RLE into a (H, W) bool mask."""
    h, w = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if isinstance(counts, str):
        counts = _counts_from_string(counts)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total != h * w:
        raise ValueError(f"RLE counts sum to {total}, expected {h * w}")
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, counts)
    return flat.reshape((h, w), order="F")


def _counts_from_string(s: str) -> list[int]:
    """Port of pycocotools ``rleFrString`` (maskApi.c).

    Each count is written in 5-bit groups (char = group + 48). Bit 0x20 means
    "more groups follow"; bit 0x10 on the last group is the sign bit.  From the
    third count onwards the value is stored as a delta to the count two places
    earlier (``cnts[m-2]``), which is why long masks compress well.
    """
    counts: list[int] = []
    p = 0
    n = len(s)
    while p < n:
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def _counts_to_string(counts: list[int]) -> str:
    """Port of pycocotools ``rleToString`` (inverse of :func:`_counts_from_string`)."""
    out = []
    for i, c in enumerate(counts):
        x = int(c)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            ch = x & 0x1F
            x >>= 5
            more = not ((ch & 0x10) == 0 and x == 0) and not ((ch & 0x10) != 0 and x == -1)
            if more:
                ch |= 0x20
            out.append(chr(ch + 48))
    return "".join(out)


def encode_compressed(mask: np.ndarray) -> dict:
    """Encode into compressed COCO RLE (``counts`` is a string), as stored in SA-V / COCO json."""
    rle = encode(mask)
    return {"size": rle["size"], "counts": _counts_to_string(rle["counts"])}


def area(rle: dict) -> int:
    counts = rle["counts"]
    if isinstance(counts, (str, bytes)):
        counts = _counts_from_string(counts.decode("ascii") if isinstance(counts, bytes) else counts)
    return int(sum(counts[1::2]))
