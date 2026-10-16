"""Procedural stress-test videos with *exact* ground truth.

Public benchmarks give an average score; they rarely tell you *why* a tracker
fails.  Each scenario here isolates one failure mode that clients care about:

occlusion     the target passes behind a wide foreground pillar and is fully hidden
              for ~12 frames → does the mask come back (re-detection)?  A second,
              never-occluded object acts as a control.
distractor    an identical-looking object crosses *in front of* the target
              → does the mask jump to the wrong object (identity switch)?
exit_reentry  the target leaves the frame for ~10 frames and comes back
              → out-of-view handling.
fast_motion   40-60 px/frame zig-zag with motion blur, rotation and scale change
              → large inter-frame displacement.

Ground truth is *modal* (visible pixels only), exactly like DAVIS, so an object
that is fully hidden has an empty GT mask and the correct prediction is "empty".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.vos.types import Sequence

H, W = 480, 854
SCENARIOS = ("occlusion", "distractor", "exit_reentry", "fast_motion")


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------


def _background(rng: np.random.Generator) -> np.ndarray:
    low = rng.uniform(60, 190, size=(6, 11, 3)).astype(np.float32)
    bg = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC)
    grain = rng.normal(0, 6, size=(H, W, 3)).astype(np.float32)
    bg = np.clip(bg + grain, 0, 255)
    # static clutter so the scene is not trivially separable by colour
    for _ in range(14):
        color = rng.uniform(40, 220, size=3).tolist()
        center = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        axes = (int(rng.uniform(15, 60)), int(rng.uniform(10, 45)))
        cv2.ellipse(bg, center, axes, float(rng.uniform(0, 180)), 0, 360, color, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(bg, (0, 0), 1.0)


@dataclass
class Sprite:
    shape: str  # "ellipse" | "roundrect"
    w: int
    h: int
    c1: tuple[int, int, int]
    c2: tuple[int, int, int]
    stripe: float = 14.0
    angle: float = 0.0


def _sprite_layers(s: Sprite, cx: float, cy: float, scale: float = 1.0, rot: float = 0.0):
    """Return (rgb float32 HxWx3, alpha float32 HxW) for one sprite placement."""
    alpha = np.zeros((H, W), np.float32)
    w, h = s.w * scale, s.h * scale
    if s.shape == "ellipse":
        cv2.ellipse(alpha, (int(cx), int(cy)), (int(w / 2), int(h / 2)), s.angle + rot, 0, 360, 1.0, -1, cv2.LINE_AA)
    else:
        box = cv2.boxPoints(((cx, cy), (w, h), s.angle + rot)).astype(np.int32)
        cv2.fillConvexPoly(alpha, box, 1.0, cv2.LINE_AA)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    theta = math.radians(35 + s.angle + rot)
    phase = ((xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)) / s.stripe
    stripes = (np.sin(phase * math.pi) > 0).astype(np.float32)[..., None]
    rgb = stripes * np.array(s.c1, np.float32) + (1 - stripes) * np.array(s.c2, np.float32)
    shade = 1.0 - 0.35 * np.clip(np.hypot(xx - cx, yy - cy) / max(w, h), 0, 1)[..., None]
    return rgb * shade, alpha


def _composite(canvas: np.ndarray, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return canvas * (1 - a) + rgb * a


@dataclass
class Placement:
    sprite: Sprite
    x: float
    y: float
    scale: float = 1.0
    rot: float = 0.0
    blur_dx: float = 0.0  # motion-blur vector (pixels) for fast motion
    blur_dy: float = 0.0


@dataclass
class Frame:
    layers: list[tuple[int | None, Placement]] = field(default_factory=list)  # (obj_id or None, placement), back→front


def _render(bg: np.ndarray, frame: Frame) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, np.ndarray]]:
    canvas = bg.copy()
    full_masks: list[tuple[int | None, np.ndarray]] = []
    for oid, p in frame.layers:
        if p.blur_dx or p.blur_dy:
            acc_rgb = np.zeros((H, W, 3), np.float32)
            acc_a = np.zeros((H, W), np.float32)
            for t in np.linspace(-0.5, 0.5, 7):
                rgb, a = _sprite_layers(p.sprite, p.x + t * p.blur_dx, p.y + t * p.blur_dy, p.scale, p.rot)
                acc_rgb += rgb * a[..., None]
                acc_a += a
            acc_a /= 7
            rgb = acc_rgb / np.maximum(acc_a[..., None] * 7, 1e-6)
            alpha = acc_a
            _, center_alpha = _sprite_layers(p.sprite, p.x, p.y, p.scale, p.rot)
            mask = center_alpha > 0.5
        else:
            rgb, alpha = _sprite_layers(p.sprite, p.x, p.y, p.scale, p.rot)
            mask = alpha > 0.5
        canvas = _composite(canvas, rgb, alpha)
        full_masks.append((oid, mask))
    # modal (visible) GT: remove pixels covered by anything drawn later
    visible: dict[int, np.ndarray] = {}
    extra: dict[str, np.ndarray] = {}
    cover = np.zeros((H, W), bool)
    for oid, m in reversed(full_masks):
        vis = m & ~cover
        if oid is not None and oid > 0:
            visible[oid] = vis
        elif oid is not None and oid < 0:  # negative id = tracked-but-unprompted distractor
            extra[f"distractor_{-oid}"] = vis
        cover |= m
    return np.clip(canvas, 0, 255).astype(np.uint8), visible, extra


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

TARGET = Sprite("ellipse", 120, 84, (240, 140, 30), (120, 50, 10), stripe=12.0, angle=10)
CONTROL = Sprite("roundrect", 80, 80, (40, 110, 230), (220, 230, 245), stripe=10.0, angle=0)
PILLAR = Sprite("roundrect", 270, 600, (95, 95, 100), (140, 135, 125), stripe=30.0, angle=0)


def _scenario_frames(name: str, n: int) -> list[Frame]:
    frames: list[Frame] = []
    for t in range(n):
        f = Frame()
        if name == "occlusion":
            # control object bounces vertically in the top-right corner, never occluded or touched
            f.layers.append((2, Placement(CONTROL, 780, 110 + 40 * math.sin(t / 4.0))))
            f.layers.append((1, Placement(TARGET, 110 + 13.0 * t, 250 + 15 * math.sin(t / 5.0))))
            f.layers.append((None, Placement(PILLAR, 430, 240)))  # foreground occluder
        elif name == "distractor":
            # target moves right, identical distractor moves left *in front of* it
            f.layers.append((1, Placement(TARGET, 120 + 12.5 * t, 250 + 10 * math.sin(t / 6.0))))
            f.layers.append((-1, Placement(TARGET, 740 - 12.5 * t, 262 - 10 * math.sin(t / 6.0))))
        elif name == "exit_reentry":
            # moves right and leaves the frame (fully out at t=13..23), then comes back from the right
            x = 520 + 32 * t if t <= 17 else 1064 - 24 * (t - 17)
            f.layers.append((1, Placement(TARGET, x, 230 + 40 * math.sin(t / 8.0), rot=4 * t)))
        elif name == "fast_motion":
            vx = 52 if (t // 8) % 2 == 0 else -52
            x = 427 + sum(52 if (k // 8) % 2 == 0 else -52 for k in range(t)) - 4 * 52
            y = 240 + 140 * math.sin(t / 2.2)
            vy = 140 * (math.sin((t + 1) / 2.2) - math.sin(t / 2.2))
            scale = 0.8 + 0.35 * (0.5 + 0.5 * math.sin(t / 4.0))
            f.layers.append((1, Placement(TARGET, x, y, scale=scale, rot=9 * t, blur_dx=vx * 0.6, blur_dy=vy * 0.6)))
        else:
            raise ValueError(f"unknown scenario {name}")
        frames.append(f)
    return frames


def generate(name: str, out_root: str | Path, num_frames: int = 48, seed: int = 0, jpeg_quality: int = 90) -> Sequence:
    """Render scenario ``name`` into ``out_root/<name>/{00000.jpg,...}`` and return a GT Sequence."""
    rng = np.random.default_rng(seed + SCENARIOS.index(name) * 101)
    bg = _background(rng)
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    gt: dict[int, dict[int, np.ndarray]] = {}
    distractors: dict[int, np.ndarray] = {}
    paths: list[Path] = []
    for t, fr in enumerate(_scenario_frames(name, num_frames)):
        img, vis, extra = _render(bg, fr)
        p = out / f"{t:05d}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        paths.append(p)
        for oid, m in vis.items():
            gt.setdefault(oid, {})[t] = m
        if "distractor_1" in extra:
            distractors[t] = extra["distractor_1"]
    return Sequence(
        name=f"synthetic/{name}",
        source="synthetic",
        frame_paths=paths,
        height=H,
        width=W,
        gt=gt,
        fps=24.0,
        meta={"scenario": name, "distractor_masks": distractors},
    )


def hidden_frames(seq: Sequence, obj_id: int = 1) -> list[int]:
    """Frames where the object is fully invisible in GT (occluded or out of view)."""
    return [f for f, m in sorted(seq.gt[obj_id].items()) if not m.any()]
