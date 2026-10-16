"""Business logic for the video segmentation endpoints (SAM 2 behind the model registry).

Flow of one request
-------------------
upload (mp4/mov) ─► size guard ─► decode every ``stride``-th frame to JPEG (temp dir)
      ─► resolve model version via the registry (+ LRU cache)  ─► validate prompts
      ─► SAM 2 propagation (serialised per model instance)     ─► GT-free stability
      ─► quality alerts (MaskQualityMonitor)  ─► Prometheus metrics ─► RLE masks in JSON

The endpoint is synchronous on purpose (Phase 1).  Long videos belong in a job queue
(Celery / RQ / Arq) with the same service function as the worker body.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from src.monitoring.drift_detector import MaskQualityMonitor
from src.serving.cache import model_cache
from src.serving.config import settings
from src.serving.observability import Observability
from src.serving.schemas import VideoSegmentationRequest
from src.serving.service import ModelVersionService, TaskMismatchError
from src.vos import rle, stability
from src.vos.types import Prompt


class VideoDecodeError(Exception):
    pass


class VideoTooLargeError(Exception):
    pass


class InvalidPromptError(Exception):
    pass


_locks: dict[tuple[str, str], threading.Lock] = {}
_locks_guard = threading.Lock()
_monitor = MaskQualityMonitor()


def _model_lock(model_name: str, version: str) -> threading.Lock:
    # SAM2VideoPredictor keeps per-call state on the instance (and our feature-cache hook),
    # so one model instance must not run two videos at the same time.
    with _locks_guard:
        return _locks.setdefault((model_name, version), threading.Lock())


def decode_video(path: Path, out_dir: Path, stride: int, max_frames: int) -> dict:
    """Write every ``stride``-th frame as ``out_dir/00000.jpg …``; return video metadata."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoDecodeError("could not open the uploaded video (supported: mp4, mov, avi, mkv)")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    original: list[int] = []
    h = w = 0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                if len(original) >= max_frames:
                    raise VideoTooLargeError(
                        f"video has more than {max_frames} frames at stride {stride}; "
                        "increase 'stride', trim the clip or use the batch CLI"
                    )
                h, w = frame.shape[:2]
                cv2.imwrite(str(out_dir / f"{len(original):05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                original.append(i)
            i += 1
    finally:
        cap.release()
    if not original:
        raise VideoDecodeError("the uploaded file contains no decodable frames")
    return {"fps": fps, "total_frames": i, "height": h, "width": w, "original_indices": original}


def _to_prompts(req: VideoSegmentationRequest, meta: dict) -> dict[int, Prompt]:
    pos_of = {orig: k for k, orig in enumerate(meta["original_indices"])}
    h, w = meta["height"], meta["width"]
    prompts: dict[int, Prompt] = {}
    for o in req.objects:
        if o.frame_idx not in pos_of:
            raise InvalidPromptError(
                f"object {o.obj_id}: frame_idx {o.frame_idx} is not processed "
                f"(stride={req.stride}, processed frames are multiples of the stride)"
            )
        coords = (o.points or []) + ([o.box[:2], o.box[2:]] if o.box else [])
        for x, y in coords:
            if not (0 <= x <= w and 0 <= y <= h):
                raise InvalidPromptError(f"object {o.obj_id}: point ({x}, {y}) is outside the {w}x{h} frame")
        if o.box is not None and o.points is None:
            prompts[o.obj_id] = Prompt(kind="box", frame_idx=pos_of[o.frame_idx], box=np.array(o.box, np.float32))
        else:
            # points (+ optional box) — SAM 2 accepts both together; we send points, box as extra clicks
            pts = np.array(o.points, np.float32)
            labels = np.array(o.labels, np.int32)
            p = Prompt(kind="points", frame_idx=pos_of[o.frame_idx], points=pts, labels=labels)
            if o.box is not None:
                p.box = np.array(o.box, np.float32)
            prompts[o.obj_id] = p
    return prompts


class VideoSegmentationService:
    @staticmethod
    def get_segmenter(db: Session, model_name: str, version: str):
        if not model_cache.contains(model_name, version):
            ModelVersionService.load_model_version(db, model_name, version)
        else:
            Observability.record_cache_hit(model_name, version)
        model = model_cache.get(model_name, version)
        if model is None or not hasattr(model, "segment"):
            raise TaskMismatchError(
                f"Model '{model_name}' version '{version}' is not a video segmentation model "
                "(register it with framework='sam2', model_format='pytorch-checkpoint')."
            )
        return model

    @staticmethod
    def segment_upload(
        db: Session, model_name: str, version: str, video_path: Path, req: VideoSegmentationRequest
    ) -> dict:
        size_mb = video_path.stat().st_size / 2**20
        if size_mb > settings.vos_max_upload_mb:
            raise VideoTooLargeError(f"upload is {size_mb:.0f} MB; limit is {settings.vos_max_upload_mb} MB")
        max_frames = min(req.max_frames or settings.vos_max_frames, settings.vos_max_frames)

        segmenter = VideoSegmentationService.get_segmenter(db, model_name, version)
        workdir = Path(tempfile.mkdtemp(prefix="vos_req_"))
        try:
            t0 = time.perf_counter()
            meta = decode_video(video_path, workdir, req.stride, max_frames)
            decode_s = time.perf_counter() - t0
            prompts = _to_prompts(req, meta)

            with _model_lock(model_name, version):
                res = segmenter.segment(workdir, prompts)
            total_s = time.perf_counter() - t0

            orig = meta["original_indices"]
            frames_rgb = None
            if req.flow_stability:
                frames_rgb = {
                    k: cv2.cvtColor(cv2.imread(str(workdir / f"{k:05d}.jpg")), cv2.COLOR_BGR2RGB)
                    for k in range(len(orig))
                }
            flow_cache: dict = {}
            objects = []
            for obj_id in sorted(res.masks):
                rep = stability.object_stability(
                    obj_id,
                    res.masks[obj_id],
                    frames=frames_rgb,
                    confidence=res.confidence.get(obj_id),
                    uncertain_fraction=res.uncertain_fraction.get(obj_id),
                    object_score=res.object_score.get(obj_id),
                    flow_cache=flow_cache,
                ).as_dict()
                alerts = _monitor.evaluate(rep)
                Observability.record_mask_quality(model_name, version, rep["tiou_mean"], rep["empty_rate"], alerts)
                masks = None
                if req.return_masks:
                    masks = {orig[k]: rle.encode(m) for k, m in sorted(res.masks[obj_id].items())}
                objects.append({"obj_id": obj_id, "stability": rep, "alerts": alerts, "masks": masks})

            Observability.record_video_success(model_name, version, total_s, res.frame_latency_s)
            return {
                "model_name": model_name,
                "version": version,
                "video": {k: v for k, v in meta.items() if k != "original_indices"} | {"stride": req.stride},
                "processed_frames": [orig[k] for k in res.processed_frames],
                "objects": objects,
                "timing": {
                    "total_s": round(total_s, 4),
                    "decode_s": round(decode_s, 4),
                    "init_s": round(res.init_latency_s, 4),
                    "prompt_s": round(res.prompt_latency_s, 4),
                    "propagate_s": round(float(sum(res.frame_latency_s)), 4),
                    "fps": round(res.fps, 3),
                },
            }
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
