"""Visual outputs: overlays, alpha mattes (rotoscoping deliverable), MP4 / GIF writers."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

PALETTE = [  # validated categorical palette (fixed order, never cycled past 8)
    (0x2A, 0x78, 0xD6),  # blue
    (0xEB, 0x68, 0x34),  # orange
    (0x1B, 0xAF, 0x7A),  # aqua
    (0xED, 0xA1, 0x00),  # yellow
    (0xE8, 0x7B, 0xA4),  # magenta
    (0x00, 0x83, 0x00),  # green
    (0x62, 0x50, 0xD6),  # violet
    (0xE3, 0x49, 0x48),  # red
]


def color_for(obj_id: int) -> tuple[int, int, int]:
    return PALETTE[(obj_id - 1) % len(PALETTE)]


def overlay(
    frame: np.ndarray,
    masks: dict[int, np.ndarray],
    alpha: float = 0.45,
    contour: bool = True,
    label: str | None = None,
    gt_contours: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Blend coloured masks onto an RGB frame; optional dashed-looking GT contours in white."""
    out = frame.astype(np.float32).copy()
    for oid, m in masks.items():
        if m is None or not m.any():
            continue
        c = np.array(color_for(oid), np.float32)
        out[m] = out[m] * (1 - alpha) + c * alpha
    out = out.clip(0, 255).astype(np.uint8)
    if contour:
        for oid, m in masks.items():
            if m is None or not m.any():
                continue
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color_for(oid), 2, cv2.LINE_AA)
    if gt_contours:
        for m in gt_contours.values():
            if m is None or not m.any():
                continue
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.rectangle(out, (0, 0), (min(out.shape[1], 12 + 9 * len(label)), 26), (20, 20, 20), -1)
        cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_alpha_pngs(frames: list[Path], masks: dict[int, np.ndarray], out_dir: str | Path) -> list[Path]:
    """RGBA PNG per frame: RGB = original pixels, A = mask (255 inside). Drop-in for NLEs / After Effects."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for f, m in sorted(masks.items()):
        rgb = np.array(Image.open(frames[f]).convert("RGB"))
        a = (m.astype(np.uint8) * 255)[..., None]
        p = out / f"{f:05d}.png"
        Image.fromarray(np.concatenate([rgb, a], axis=2), mode="RGBA").save(p, optimize=True)
        paths.append(p)
    return paths


def write_video(frames: list[np.ndarray], path: str | Path, fps: float = 24.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # yuv420p needs even dimensions
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=1) as wr:
        for fr in frames:
            wr.append_data(fr[:h2, :w2])
    return path


def write_gif(frames: list[np.ndarray], path: str | Path, fps: float = 12.0, width: int = 480) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    small = []
    for fr in frames:
        h, w = fr.shape[:2]
        nh = int(h * width / w)
        small.append(Image.fromarray(fr).resize((width, nh), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE))
    small[0].save(path, save_all=True, append_images=small[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 6) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    pad = np.full((h, gap, 3), 255, np.uint8)
    return np.hstack([left, pad, right])
