"""SAM 2.1 video segmenter: a thin, testable wrapper around ``SAM2VideoPredictor``.

What this wrapper adds on top of the official predictor
-------------------------------------------------------
1. **One interface for every prompt type** (mask / box / clicks) via :class:`Prompt`.
2. **Late-appearing objects** are tracked in a separate pass that starts at the
   object's first prompt frame (same policy as the official
   ``tools/vos_inference.py --track_object_appearing_later_in_video``).
3. **Persistent backbone-feature cache.**  ``SAM2VideoPredictor`` keeps the image
   features of *one* frame only (see the "we can use an LRU cache in the future"
   comment in ``_get_image_feature``).  When we run several experiments on the same
   video (mask vs. box vs. click prompts) the Hiera image encoder — ~80 % of the
   runtime — would be recomputed every time.  :class:`FeatureCache` keeps them, so
   the 2nd and 3rd prompt type cost only the memory-attention + decoder time.
   Results are bit-identical (verified in ``tests/test_sam2_integration.py``).
4. **GT-free confidence signals** per frame: mean foreground probability, share of
   uncertain pixels (0.3 < p < 0.7) and SAM 2's own object-presence logit.
5. **Timing** per frame (``time.perf_counter``) and process peak RSS.

The module imports ``torch``/``sam2`` lazily so the rest of the platform (and its
unit tests) run without a deep-learning stack installed.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.vos.types import Prompt, SegmentationResult

logger = logging.getLogger("vos.predictor")

# Hiera trunk signature → official SAM 2.1 config.  Lets us load a checkpoint without
# being told which variant it is (useful for the model registry, see loader.py).
_VARIANT_BY_TRUNK = {
    # (embed_dim, num_blocks): config
    (96, 12): ("tiny", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    (96, 16): ("small", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    (112, 24): ("base_plus", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    (144, 48): ("large", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


class VideoSegmenter(Protocol):
    """Anything that turns a folder of frames + prompts into per-frame masks."""

    name: str

    def segment(
        self,
        frame_dir: str | Path,
        prompts: dict[int, Prompt],
        video_key: str | None = None,
    ) -> SegmentationResult: ...


def peak_rss_mb() -> float:
    """Process peak resident memory (VmHWM) in MB; 0 if unavailable (non-Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def load_state_dict(source: str | Path | bytes) -> dict:
    """Load a SAM 2 checkpoint (``{"model": state_dict}``) from a path or raw bytes.

    ``weights_only=True`` refuses arbitrary pickled objects, so a tampered checkpoint
    cannot execute code at load time.
    """
    import torch

    buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    ckpt = torch.load(buf, map_location="cpu", weights_only=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(sd, dict) or not sd:
        raise ValueError("checkpoint does not contain a state dict")
    return sd


def infer_variant(state_dict: dict) -> tuple[str, str]:
    """Detect tiny/small/base_plus/large from tensor shapes (no metadata needed)."""
    try:
        embed_dim = int(state_dict["image_encoder.trunk.patch_embed.proj.weight"].shape[0])
    except KeyError as e:
        raise ValueError("not a SAM 2 checkpoint: missing Hiera patch embedding") from e
    blocks = {k.split(".")[3] for k in state_dict if k.startswith("image_encoder.trunk.blocks.")}
    key = (embed_dim, len(blocks))
    if key not in _VARIANT_BY_TRUNK:
        raise ValueError(f"unknown SAM 2 trunk signature {key}")
    return _VARIANT_BY_TRUNK[key]


class FeatureCache:
    """LRU cache of SAM 2 backbone outputs keyed by (video_key, original_frame_idx).

    Memory note: a SAM 2 backbone output is ~100 MB/frame in fp32, but ~80 MB of that
    is the sinusoidal *position encoding* (256 ch at 256×256, 128×128, 64×64), which
    depends only on the input size and is identical for every frame.  ``put`` stores
    one shared copy of it, bringing the cost down to ~20 MB/frame.
    """

    def __init__(self, max_frames: int = 64):
        self.max_frames = max_frames
        self._store: OrderedDict[tuple[str, int], Any] = OrderedDict()
        self._shared_pos: list | None = None
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, int]):
        item = self._store.get(key)
        if item is not None:
            self._store.move_to_end(key)
            self.hits += 1
        else:
            self.misses += 1
        return item

    def _dedupe_pos(self, backbone_out: dict) -> dict:
        import torch

        pos = backbone_out.get("vision_pos_enc")
        if pos is None:
            return backbone_out
        if self._shared_pos is None:
            self._shared_pos = list(pos)
            return backbone_out
        shared = []
        for new, ref in zip(pos, self._shared_pos, strict=False):
            same = new.shape == ref.shape and (new is ref or torch.equal(new, ref))
            shared.append(ref if same else new)
        out = dict(backbone_out)
        out["vision_pos_enc"] = shared
        return out

    def put(self, key: tuple[str, int], value) -> None:
        self._store[key] = self._dedupe_pos(value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_frames:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


@dataclass
class Sam2Config:
    checkpoint: str | None = None
    config: str | None = None  # auto-detected from the checkpoint when None
    device: str = "cpu"
    num_threads: int | None = None
    feature_cache_frames: int = 0  # 0 disables the persistent cache
    apply_postprocessing: bool = True


class Sam2VideoSegmenter:
    """SAM 2.1 video object segmentation with click / box / mask prompts."""

    def __init__(self, cfg: Sam2Config, state_dict: dict | None = None):
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        if cfg.num_threads:
            torch.set_num_threads(cfg.num_threads)
        if state_dict is None:
            if not cfg.checkpoint:
                raise ValueError("either cfg.checkpoint or state_dict is required")
            state_dict = load_state_dict(cfg.checkpoint)
        variant, auto_cfg = infer_variant(state_dict)
        config = cfg.config or auto_cfg
        self.variant = variant
        self.name = f"sam2.1-hiera-{variant}"
        self.device = cfg.device
        # build without a checkpoint, then load strictly (fails loudly on missing / extra keys)
        predictor = build_sam2_video_predictor(
            config, ckpt_path=None, device="cpu", apply_postprocessing=cfg.apply_postprocessing
        )
        missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.predictor = predictor.to(cfg.device).eval()
        self.num_params = sum(p.numel() for p in self.predictor.parameters())
        self.feature_cache = FeatureCache(cfg.feature_cache_frames) if cfg.feature_cache_frames else None
        self._video_key: str | None = None
        self._frame_ids: list[int] | None = None
        if self.feature_cache is not None:
            self._install_feature_cache()

    # -- feature cache --------------------------------------------------------------
    def _install_feature_cache(self) -> None:
        original = self.predictor._get_image_feature
        cache = self.feature_cache
        segmenter = self

        def cached_get_image_feature(inference_state, frame_idx, batch_size):
            ids = segmenter._frame_ids
            key = (segmenter._video_key, ids[frame_idx] if ids is not None else frame_idx)
            backbone_out = cache.get(key) if segmenter._video_key else None
            if backbone_out is not None:
                # Re-use the features but take the image view from the *current* state:
                # caching the image would pin the previous run's whole frame tensor in RAM.
                device = inference_state["device"]
                image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
                inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
            out = original(inference_state, frame_idx, batch_size)
            if backbone_out is None and segmenter._video_key:
                cache.put(key, inference_state["cached_features"][frame_idx][1])
            return out

        self.predictor._get_image_feature = cached_get_image_feature

    # -- main entry point ------------------------------------------------------------
    def segment(
        self,
        frame_dir: str | Path,
        prompts: dict[int, Prompt],
        video_key: str | None = None,
        frame_ids: list[int] | None = None,
        prob_low: float = 0.3,
        prob_high: float = 0.7,
    ) -> SegmentationResult:
        """Track every object in ``prompts`` through the frames in ``frame_dir``.

        Frame indices in prompts / results refer to positions *in frame_dir*
        (the caller maps them back to the original video when sub-sampling).
        ``video_key`` + ``frame_ids`` (original frame index of each position) are the
        feature-cache key, so runs with different strides share encoder features.
        """
        import torch

        if not prompts:
            raise ValueError("at least one prompt is required")
        self._video_key = video_key
        self._frame_ids = frame_ids
        result = SegmentationResult(masks={}, confidence={}, uncertain_fraction={}, object_score={})

        # Objects sharing the earliest prompt frame are tracked jointly; the others
        # (appearing later) get their own pass starting at their prompt frame.
        first = min(p.frame_idx for p in prompts.values())
        groups: list[dict[int, Prompt]] = [{o: p for o, p in prompts.items() if p.frame_idx == first}]
        groups += [{o: p} for o, p in prompts.items() if p.frame_idx != first]

        with torch.inference_mode():
            for group in groups:
                t0 = time.perf_counter()
                state = self.predictor.init_state(video_path=str(frame_dir), offload_video_to_cpu=True)
                result.init_latency_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                for obj_id, p in group.items():
                    self._add_prompt(state, obj_id, p)
                result.prompt_latency_s += time.perf_counter() - t0

                start = min(p.frame_idx for p in group.values())
                t_prev = time.perf_counter()
                for frame_idx, obj_ids, logits in self.predictor.propagate_in_video(state, start_frame_idx=start):
                    probs = torch.sigmoid(logits[:, 0]).float().cpu().numpy()  # (N, H, W)
                    for k, obj_id in enumerate(obj_ids):
                        if obj_id not in group:
                            continue
                        p = probs[k]
                        m = p > 0.5
                        result.masks.setdefault(obj_id, {})[frame_idx] = m
                        result.confidence.setdefault(obj_id, {})[frame_idx] = (
                            float(p[m].mean()) if m.any() else float("nan")
                        )
                        result.uncertain_fraction.setdefault(obj_id, {})[frame_idx] = float(
                            np.mean((p > prob_low) & (p < prob_high))
                        )
                        result.object_score.setdefault(obj_id, {})[frame_idx] = self._object_score(
                            state, obj_id, frame_idx
                        )
                    now = time.perf_counter()
                    result.frame_latency_s.append(now - t_prev)
                    result.processed_frames.append(frame_idx)
                    t_prev = now
                del state
        result.peak_rss_mb = peak_rss_mb()
        return result

    def _add_prompt(self, state, obj_id: int, p: Prompt) -> None:
        if p.kind == "mask":
            self.predictor.add_new_mask(state, frame_idx=p.frame_idx, obj_id=obj_id, mask=p.mask.astype(bool))
        elif p.kind == "box":
            self.predictor.add_new_points_or_box(state, frame_idx=p.frame_idx, obj_id=obj_id, box=p.box)
        elif p.kind == "points":
            # an optional box can accompany clicks (SAM 2 encodes it as two corner points)
            self.predictor.add_new_points_or_box(
                state, frame_idx=p.frame_idx, obj_id=obj_id, points=p.points, labels=p.labels, box=p.box
            )
        else:
            raise ValueError(f"unsupported prompt kind {p.kind}")

    @staticmethod
    def _object_score(state, obj_id: int, frame_idx: int) -> float | None:
        """SAM 2's object-presence logit for (object, frame); > 0 means 'object visible'."""
        try:
            obj_idx = state["obj_id_to_idx"][obj_id]
            out = state["output_dict_per_obj"][obj_idx]
            d = out["cond_frame_outputs"].get(frame_idx) or out["non_cond_frame_outputs"].get(frame_idx)
            if d is None or d.get("object_score_logits") is None:
                return None
            return float(d["object_score_logits"].flatten()[0])
        except (KeyError, IndexError, TypeError):
            return None


def segment_sequence_frames(
    segmenter: VideoSegmenter,
    frame_paths: list[Path],
    prompts: dict[int, Prompt],
    video_key: str | None = None,
    workdir: str | Path | None = None,
) -> SegmentationResult:
    """Convenience: link ``frame_paths`` into a temp dir named 00000.jpg… and segment."""
    tmp = Path(workdir or tempfile.mkdtemp(prefix="vos_frames_"))
    tmp.mkdir(parents=True, exist_ok=True)
    for k, src in enumerate(frame_paths):
        dst = tmp / f"{k:05d}.jpg"
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                dst.write_bytes(Path(src).read_bytes())
    return segmenter.segment(tmp, prompts, video_key=video_key)
