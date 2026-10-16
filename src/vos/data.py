"""Sequence loaders: DAVIS, SA-V (train-format json), synthetic and unlabeled videos.

All loaders return :class:`src.vos.types.Sequence` objects whose frames live on
disk as JPEGs, because the SAM 2 video predictor reads a directory of frames.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.vos import rle
from src.vos.types import Sequence

# ---------------------------------------------------------------------------
# DAVIS 2016 / 2017
# ---------------------------------------------------------------------------


def load_davis_sequence(root: str | Path, name: str, resolution: str = "480p") -> Sequence:
    """Load one DAVIS sequence.

    DAVIS 2017 annotations are palette PNGs where the palette index is the object id.
    DAVIS 2016 annotations are binary (0 / 255) and contain a single object (id 1).
    """
    root = Path(root)
    img_dir = root / "JPEGImages" / resolution / name
    ann_dir = root / "Annotations" / resolution / name
    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames in {img_dir}")
    w, h = Image.open(frames[0]).size

    gt: dict[int, dict[int, np.ndarray]] = {}
    for ann in sorted(ann_dir.glob("*.png")):
        idx = int(ann.stem)
        im = Image.open(ann)
        arr = np.array(im)
        if im.mode == "P":
            ids = [int(v) for v in np.unique(arr) if v not in (0, 255)]  # 255 = void in DAVIS 2017
            for oid in ids:
                gt.setdefault(oid, {})[idx] = arr == oid
        else:
            if arr.ndim == 3:
                arr = arr[..., 0]
            gt.setdefault(1, {})[idx] = arr > 127
    # make sure every object has an entry (empty mask) on every annotated frame
    ann_frames = sorted({f for per in gt.values() for f in per})
    for oid in gt:
        for f in ann_frames:
            gt[oid].setdefault(f, np.zeros((h, w), dtype=bool))
    return Sequence(name=name, source="davis", frame_paths=frames, height=h, width=w, gt=gt, fps=24.0)


def davis_split(root: str | Path, year: int = 2017, split: str = "val") -> list[str]:
    path = Path(root) / "ImageSets" / str(year) / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# SA-V (train-format: mp4 + {video_id}_manual.json with compressed RLE masklets)
# ---------------------------------------------------------------------------


def extract_frames(video_path: str | Path, out_dir: str | Path, quality: int = 95) -> list[Path]:
    """Decode a video into ``out_dir/00000.jpg ...`` (idempotent)."""
    out_dir = Path(out_dir)
    existing = sorted(out_dir.glob("*.jpg"))
    if existing:
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video_path}")
    paths, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p = out_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        paths.append(p)
        i += 1
    cap.release()
    return paths


def video_fps(video_path: str | Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return float(fps)


def load_sav_train_video(
    video_path: str | Path, manual_json: str | Path, frames_dir: str | Path, annotation_stride: int = 4
) -> Sequence:
    """Load an SA-V *train-format* video (as shipped in ``sam2/sav_dataset/example``).

    The json stores ``masklet[k][o]`` = compressed RLE of object ``o`` on the k-th
    annotated frame; annotations are at 6 fps on a 24 fps video, i.e. frame ``k * 4``.
    """
    meta = json.loads(Path(manual_json).read_text())
    frames = extract_frames(video_path, frames_dir)
    h, w = int(meta["video_height"]), int(meta["video_width"])
    gt: dict[int, dict[int, np.ndarray]] = {}
    for k, per_frame in enumerate(meta["masklet"]):
        f = k * annotation_stride
        if f >= len(frames):
            break
        for o, r in enumerate(per_frame):
            oid = int(meta["masklet_id"][o]) + 1  # keep 0 for background
            gt.setdefault(oid, {})[f] = rle.decode(r)
    return Sequence(
        name=str(meta["video_id"]),
        source="sav",
        frame_paths=frames,
        height=h,
        width=w,
        gt=gt,
        fps=video_fps(video_path),
        meta={
            "split": meta.get("video_split"),
            "masklet_size_bucket": meta.get("masklet_size_bucket"),
            "masklet_visibility_changes": meta.get("masklet_visibility_changes"),
            "annotation_stride": annotation_stride,
        },
    )


# ---------------------------------------------------------------------------
# Unlabeled video (stability metrics only)
# ---------------------------------------------------------------------------


def load_unlabeled(
    frames_or_video: str | Path, name: str | None = None, cache_dir: str | Path | None = None
) -> Sequence:
    src = Path(frames_or_video)
    if src.is_dir():
        frames = sorted(src.glob("*.jpg"), key=lambda p: int(p.stem))
        fps = 24.0
    else:
        cache_dir = Path(cache_dir or src.with_suffix(""))
        frames = extract_frames(src, cache_dir)
        fps = video_fps(src)
    w, h = Image.open(frames[0]).size
    return Sequence(name=name or src.stem, source="unlabeled", frame_paths=frames, height=h, width=w, fps=fps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def materialize(seq: Sequence, out_dir: str | Path, stride: int = 1, start: int = 0, max_frames: int | None = None):
    """Write the (sub-sampled) frames SAM 2 should see into ``out_dir`` as 00000.jpg, 00001.jpg ...

    Returns the list of *original* frame indices, so predictions can be mapped back.
    Files are hard-linked when possible (no copy cost).
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    original = list(range(start, seq.num_frames, stride))
    if max_frames is not None:
        original = original[:max_frames]
    for k, i in enumerate(original):
        dst = out_dir / f"{k:05d}.jpg"
        src = seq.frame_paths[i]
        if src.suffix.lower() in (".jpg", ".jpeg"):
            try:
                dst.hardlink_to(src)
                continue
            except OSError:
                pass
            shutil.copyfile(src, dst)
        else:
            Image.open(src).convert("RGB").save(dst, quality=95)
    return original
