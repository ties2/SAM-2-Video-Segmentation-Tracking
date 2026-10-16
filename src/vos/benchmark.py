"""Config-driven Phase-1 benchmark: *can SAM 2 keep a mask on an object across a video?*

For every (sequence × prompt type × frame stride) the runner
  1. writes the frames SAM 2 should see (sub-sampled when stride > 1),
  2. derives the prompt from the first visible GT mask (mask / box / 1 click / 3 clicks),
  3. propagates with the segmenter,
  4. scores J, F, J&F, recall and decay (DAVIS protocol) on GT frames,
  5. computes GT-free stability metrics (temporal IoU, flow-warped IoU, drop-outs …),
  6. for sequences with visibility changes, measures *event* metrics:
       hidden_J            J on frames where the object is fully hidden (1 = correctly empty)
       redetection_delay   frames after reappearance until J ≥ 0.5
       distractor_leakage  share of the predicted mask lying on a look-alike distractor
  7. stores per-frame rows so GT-free signals can be correlated with J afterwards.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.vos import data, metrics, stability
from src.vos.predictor import VideoSegmenter
from src.vos.prompts import prompt_from_mask
from src.vos.types import Prompt, SegmentationResult, Sequence


@dataclass
class RunSpec:
    sequence: Sequence
    prompt: str = "mask"  # mask | box | click | click3
    stride: int = 1
    max_frames: int | None = None
    flow_stability: bool = True


@dataclass
class RunRecord:
    run_id: str
    sequence: str
    source: str
    prompt: str
    stride: int
    num_frames: int
    num_objects: int
    model: str
    scores: dict
    per_object: list[dict]
    stability: list[dict]
    events: list[dict]
    timing: dict
    per_frame: list[dict] = field(default_factory=list, repr=False)
    predictions: dict | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Event metrics (visibility changes / distractors)
# ---------------------------------------------------------------------------


def visibility_events(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray], frames: list[int]) -> dict:
    """Hidden-frame accuracy and re-detection delay for one object."""
    vis = [bool(gt[f].any()) for f in frames]
    js = [metrics.jaccard(gt[f], pred.get(f, np.zeros_like(gt[f]))) for f in frames]
    hidden = [j for j, v in zip(js, vis, strict=True) if not v]
    delays: list[int | None] = []
    for k in range(1, len(frames)):
        if vis[k] and not vis[k - 1]:  # reappearance
            d = None
            for m in range(k, len(frames)):
                if not vis[m]:
                    break
                if js[m] >= 0.5:
                    d = m - k
                    break
            delays.append(d)
    after = []
    relapse = 0
    if any(not v for v in vis):
        first_hidden = vis.index(False)
        after = [j for j, v in zip(js[first_hidden:], vis[first_hidden:], strict=True) if v]
        # "relapse": visible frames *after the first successful re-detection* that are lost again (J < 0.5).
        # A tracker can grab the object for one frame and drop it again when its memory is
        # dominated by "object absent" frames — the re-detection delay alone would hide that.
        first_ok = next((k for k in range(first_hidden, len(frames)) if vis[k] and js[k] >= 0.5), None)
        if first_ok is not None:
            relapse = sum(1 for k in range(first_ok + 1, len(frames)) if vis[k] and js[k] < 0.5)
    return {
        "hidden_frames": len(hidden),
        "hidden_J": float(np.mean(hidden)) if hidden else None,
        "reappearances": len(delays),
        "redetection_delay": delays,
        "recovered_all": all(d is not None for d in delays) if delays else None,
        "J_after_first_hide": float(np.mean(after)) if after else None,
        "lost_after_redetection": relapse,
    }


def distractor_leakage(pred: dict[int, np.ndarray], distractors: dict[int, np.ndarray], frames: list[int]) -> dict:
    fracs = []
    for f in frames:
        p = pred.get(f)
        d = distractors.get(f)
        if p is None or d is None or not p.any():
            continue
        fracs.append(np.count_nonzero(p & d) / np.count_nonzero(p))
    if not fracs:
        return {"distractor_leakage_mean": None, "distractor_leakage_max": None, "identity_switch": None}
    arr = np.array(fracs)
    # identity switch: ≥ 5 consecutive frames with most of the mask on the distractor
    run, best = 0, 0
    for v in arr:
        run = run + 1 if v > 0.5 else 0
        best = max(best, run)
    return {
        "distractor_leakage_mean": float(arr.mean()),
        "distractor_leakage_max": float(arr.max()),
        "identity_switch": bool(best >= 5),
    }


# ---------------------------------------------------------------------------
# Running one experiment
# ---------------------------------------------------------------------------


def build_prompts(seq: Sequence, original_idx: list[int], kind: str) -> dict[int, Prompt]:
    """Prompt each object on its first *processed* frame where GT shows it."""
    pos_of = {orig: k for k, orig in enumerate(original_idx)}
    prompts: dict[int, Prompt] = {}
    for oid in seq.object_ids:
        for f in seq.annotated_frames(oid):
            if f in pos_of and seq.gt[oid][f].any():
                prompts[oid] = prompt_from_mask(seq.gt[oid][f], kind, frame_idx=pos_of[f])
                break
    return prompts


def run_one(segmenter: VideoSegmenter, spec: RunSpec, workdir: str | Path, keep_predictions: bool = False) -> RunRecord:
    seq = spec.sequence
    frame_dir = Path(workdir) / "frames" / seq.name.replace("/", "_") / f"s{spec.stride}"
    original_idx = data.materialize(seq, frame_dir, stride=spec.stride, max_frames=spec.max_frames)
    prompts = build_prompts(seq, original_idx, spec.prompt)
    if not prompts:
        raise RuntimeError(f"{seq.name}: no visible GT object on the processed frames")

    t0 = time.perf_counter()
    res: SegmentationResult = segmenter.segment(frame_dir, prompts, video_key=seq.name, frame_ids=original_idx)
    wall = time.perf_counter() - t0

    # map positions → original frame indices
    def remap(d: dict[int, dict[int, object]]) -> dict[int, dict[int, object]]:
        return {o: {original_idx[k]: v for k, v in per.items()} for o, per in d.items()}

    pred = remap(res.masks)
    conf = remap(res.confidence)
    unc = remap(res.uncertain_fraction)
    objscore = remap(res.object_score)

    score = metrics.evaluate_sequence(seq.name, {o: seq.gt[o] for o in prompts}, pred, eval_frames=original_idx)
    summary = score.summary()

    # GT-free stability on the processed frames
    frames_rgb = {f: data.read_rgb(seq.frame_paths[f]) for f in original_idx} if spec.flow_stability else None
    flow_cache: dict = {}
    stab = [
        stability.object_stability(
            o,
            pred[o],
            frames=frames_rgb,
            confidence=conf.get(o),
            uncertain_fraction=unc.get(o),
            object_score=objscore.get(o),
            flow_cache=flow_cache,
        ).as_dict()
        for o in sorted(pred)
    ]

    # event metrics
    events = []
    for o in sorted(prompts):
        gt_frames = [f for f in seq.annotated_frames(o) if f in set(original_idx)]
        ev = {"obj_id": o, **visibility_events(seq.gt[o], pred.get(o, {}), gt_frames)}
        if seq.meta.get("distractor_masks") and o == 1:
            ev.update(distractor_leakage(pred.get(o, {}), seq.meta["distractor_masks"], gt_frames))
        events.append(ev)

    # per-frame rows (for correlation of GT-free signals with J)
    rows = []
    for obj in score.objects:
        for f, j, fm in zip(obj.frames, obj.j, obj.f, strict=True):
            p = pred.get(obj.obj_id, {}).get(f)
            rows.append(
                {
                    "sequence": seq.name,
                    "prompt": spec.prompt,
                    "stride": spec.stride,
                    "obj_id": obj.obj_id,
                    "frame": f,
                    "J": j,
                    "F": fm,
                    "gt_area": int(seq.gt[obj.obj_id][f].sum()),
                    "pred_area": int(p.sum()) if p is not None else 0,
                    "confidence": conf.get(obj.obj_id, {}).get(f),
                    "uncertain_fraction": unc.get(obj.obj_id, {}).get(f),
                    "object_score": objscore.get(obj.obj_id, {}).get(f),
                }
            )

    lat = np.array(res.frame_latency_s) if res.frame_latency_s else np.array([float("nan")])
    timing = {
        "wall_s": wall,
        "init_s": res.init_latency_s,
        "prompt_s": res.prompt_latency_s,
        "propagate_s": float(np.nansum(lat)),
        "fps": res.fps,
        "ms_per_frame": float(np.nanmean(lat) * 1000),
        "ms_per_frame_p95": float(np.nanpercentile(lat, 95) * 1000),
        "peak_rss_mb": res.peak_rss_mb,
    }
    cache = getattr(segmenter, "feature_cache", None)
    if cache is not None:
        timing["feature_cache_hits"] = cache.hits
        timing["feature_cache_misses"] = cache.misses

    run_id = f"{seq.name}|{spec.prompt}|s{spec.stride}"
    return RunRecord(
        run_id=run_id,
        sequence=seq.name,
        source=seq.source,
        prompt=spec.prompt,
        stride=spec.stride,
        num_frames=len(original_idx),
        num_objects=len(prompts),
        model=getattr(segmenter, "name", "unknown"),
        scores=summary,
        per_object=[
            {"obj_id": o.obj_id, "J": o.j_stat.as_dict(), "F": o.f_stat.as_dict(), "J&F": o.jf, "frames": len(o.frames)}
            for o in score.objects
        ],
        stability=stab,
        events=events,
        timing=timing,
        per_frame=rows,
        predictions=pred if keep_predictions else None,
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rx, ry = _rank(x[ok]), _rank(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(scores, labels) -> float:
    """AUROC of ``scores`` for predicting ``labels`` (1 = failure frame). Mann–Whitney U."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rank(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def signal_quality(rows: list[dict], fail_threshold: float = 0.5) -> dict:
    """How well do GT-free signals flag bad frames (J < threshold)?"""
    rows = [r for r in rows if r["gt_area"] > 0]  # only frames where the object is visible
    j = np.array([r["J"] for r in rows], float)
    fail = (j < fail_threshold).astype(int)
    out = {"frames": len(rows), "failure_frames": int(fail.sum())}
    for name, sign in (("confidence", -1), ("object_score", -1), ("uncertain_fraction", +1)):
        v = np.array([np.nan if r[name] is None else r[name] for r in rows], float)
        ok = ~np.isnan(v)
        out[name] = {
            "spearman_with_J": spearman(v, j),
            # sign flips the score so that "higher = more likely failure"
            "auroc_detect_failure": auroc(sign * v, fail),
            # coverage matters: confidence is undefined (NaN) when the predicted mask is empty,
            # so it never sees "object lost" frames — the most common failure.
            "frames_with_value": int(ok.sum()),
            "failures_with_value": int(fail[ok].sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _summary_from_run(d: dict) -> dict:
    sc, tm = d.get("scores", {}), d.get("timing", {})
    return {
        "run_id": d["run_id"],
        "sequence": d["sequence"],
        "source": d["source"],
        "prompt": d["prompt"],
        "stride": d["stride"],
        "frames": d["num_frames"],
        "objects": d["num_objects"],
        **{k: sc.get(k) for k in ("J&F", "J_mean", "F_mean", "J_recall", "J_decay")},
        "fps": tm.get("fps"),
        "ms_per_frame": tm.get("ms_per_frame"),
    }


def read_per_frame(path: str | Path) -> list[dict]:
    """Load per_frame.csv back with proper types (NaN for missing signals)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("J", "F", "confidence", "uncertain_fraction", "object_score"):
                r[k] = float(r[k]) if r.get(k) not in ("", None) else float("nan")
            for k in ("frame", "obj_id", "stride", "gt_area", "pred_area"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def save_results(
    records: list[RunRecord], out_dir: str | Path, extra: dict | None = None, append: bool = False
) -> None:
    """Write runs.json, per_frame.csv and summary.csv.

    ``append=True`` merges with what is already in ``out_dir`` (runs with the same
    run_id are replaced), so extra experiments can be added without re-running
    the whole benchmark.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    new_runs = []
    for r in records:
        d = asdict(r)
        d.pop("per_frame")
        d.pop("predictions")
        new_runs.append(_clean(d))
    new_ids = {r.run_id for r in records}

    old: dict = {}
    if append and (out / "runs.json").exists():
        old = json.loads((out / "runs.json").read_text())
    runs = [r for r in old.get("runs", []) if r["run_id"] not in new_ids] + new_runs
    meta = {k: v for k, v in old.items() if k != "runs"}
    meta.update({k: v for k, v in (extra or {}).items() if v})
    (out / "runs.json").write_text(json.dumps(_clean({"runs": runs, **meta}), indent=2))

    rows = [row for r in records for row in r.per_frame]
    if append and (out / "per_frame.csv").exists():
        kept = [
            r
            for r in read_per_frame(out / "per_frame.csv")
            if f"{r['sequence']}|{r['prompt']}|s{r['stride']}" not in new_ids
        ]
        rows = kept + rows
    if rows:
        with open(out / "per_frame.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(_clean(rows))

    summ = [_summary_from_run(d) for d in runs]
    if summ:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
            w.writeheader()
            w.writerows(_clean(summ))
