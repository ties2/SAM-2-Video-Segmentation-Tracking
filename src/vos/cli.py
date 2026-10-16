"""Command-line entry point: ``python -m src.vos.cli <command>``.

Commands
--------
fetch-weights   download a SAM 2.1 checkpoint (official Meta URL, GitHub mirror as fallback)
fetch-samples   get the small public samples used by the Phase-1 benchmark
synth           render the synthetic stress-test scenarios
benchmark       run a YAML-configured benchmark and write results/
segment         rotoscope a video: clicks/box in, alpha PNGs + overlay MP4 + masks + stability out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from src.vos import benchmark, data, stability, synthetic, visualize
from src.vos.types import Prompt

# Official checkpoints (Meta) and a public mirror that ships the same architecture in fp16.
WEIGHTS = {
    "tiny": [
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt",
    ],
    "small": [
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_s.pt",
    ],
    "base_plus": [
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt",
    ],
    "large": [
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt",
    ],
}

SAMPLE_REPOS = {
    # DAVIS 2016/2017 val sequence "car-shadow" (frames + full GT), shipped with OSVOS for its demo
    "davis": ("https://github.com/scaelles/OSVOS-TensorFlow.git", ["DAVIS"]),
    # SA-V example video + manual masklets, the 'bedroom' clip and demo videos from the SAM 2 repo
    "sam2": (
        "https://github.com/facebookresearch/sam2.git",
        ["sav_dataset/example", "notebooks/videos/bedroom", "demo/data/gallery"],
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# fetch-weights / fetch-samples
# ---------------------------------------------------------------------------


def cmd_fetch_weights(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for url in WEIGHTS[args.variant]:
        dst = out / Path(url).name
        if dst.exists():
            print(f"exists: {dst} sha256={sha256(dst)}")
            return
        try:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dst)
            print(f"saved {dst} sha256={sha256(dst)}")
            return
        except Exception as e:  # network policy, 404 …
            print(f"  failed: {e}")
            dst.unlink(missing_ok=True)
    sys.exit("could not download weights from any source")


def cmd_fetch_samples(args) -> None:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, paths) in SAMPLE_REPOS.items():
        tmp = root / f".clone_{name}"
        if not tmp.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(tmp)], check=True
            )
            subprocess.run(["git", "-C", str(tmp), "sparse-checkout", "set", "--no-cone", *paths], check=True)
        dst = root / name
        dst.mkdir(exist_ok=True)
        for p in paths:
            src = tmp / p
            target = dst / Path(p).name
            if src.exists() and not target.exists():
                shutil.copytree(src, target) if src.is_dir() else shutil.copy2(src, target)
        print(f"{name}: {sorted(x.name for x in dst.iterdir())}")
    print(
        "\nFull DAVIS 2017 (not redistributed here): "
        "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
    )


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------


def cmd_synth(args) -> None:
    for s in args.scenarios or synthetic.SCENARIOS:
        seq = synthetic.generate(s, args.out, num_frames=args.frames, seed=args.seed)
        print(f"{seq.name}: {seq.num_frames} frames, objects={seq.object_ids}, hidden={synthetic.hidden_frames(seq)}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def _load_sequence(spec: dict, cache: dict) -> data.Sequence:
    key = json.dumps(spec, sort_keys=True)
    if key in cache:
        return cache[key]
    src = spec["source"]
    if src == "davis":
        seq = data.load_davis_sequence(spec["root"], spec["name"], spec.get("resolution", "480p"))
    elif src == "sav":
        seq = data.load_sav_train_video(spec["video"], spec["annotations"], spec["frames_dir"])
    elif src == "synthetic":
        seq = synthetic.generate(spec["scenario"], spec["root"], num_frames=spec.get("num_frames", 48))
    elif src == "unlabeled":
        seq = data.load_unlabeled(spec["frames"], name=spec.get("name"))
    else:
        raise ValueError(f"unknown source {src}")
    cache[key] = seq
    return seq


def _environment(cfg: dict, segmenter) -> dict:
    import torch

    from src.vos.predictor import load_state_dict

    ck = Path(cfg["model"]["checkpoint"])
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": cfg["model"].get("device", "cpu"),
        "model": segmenter.name,
        "num_params": segmenter.num_params,
        "checkpoint": ck.name,
        "checkpoint_sha256": sha256(ck),
        "checkpoint_dtypes": sorted({str(v.dtype) for v in load_state_dict(ck).values()}),
        "checkpoint_source": cfg["model"].get("source"),
    }


def cmd_benchmark(args) -> None:
    from src.vos.predictor import Sam2Config, Sam2VideoSegmenter

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out or cfg["out_dir"])
    workdir = Path(cfg.get("workdir", ".cache/vos"))
    m = cfg["model"]
    segmenter = Sam2VideoSegmenter(
        Sam2Config(
            checkpoint=m["checkpoint"],
            device=m.get("device", "cpu"),
            num_threads=m.get("num_threads"),
            feature_cache_frames=m.get("feature_cache_frames", 0),
        )
    )
    env = _environment(cfg, segmenter)
    print(json.dumps(env, indent=2))

    seq_cache: dict = {}
    records: list[benchmark.RunRecord] = []
    only = set(args.only.split(",")) if args.only else None
    pred_dir = out_dir / "predictions"
    current_video: str | None = None
    for exp in cfg["experiments"]:
        for sname in exp["sequences"]:
            if only and sname not in only:
                continue
            seq = _load_sequence(cfg["sequences"][sname], seq_cache)
            if segmenter.feature_cache is not None and seq.name != current_video:
                segmenter.feature_cache.clear()  # one video at a time keeps RAM bounded
            current_video = seq.name
            for stride in exp.get("strides", [1]):
                for prompt in exp.get("prompts", ["mask"]):
                    spec = benchmark.RunSpec(
                        seq,
                        prompt=prompt,
                        stride=stride,
                        max_frames=exp.get("max_frames"),
                        flow_stability=exp.get("flow_stability", True),
                    )
                    t0 = time.perf_counter()
                    rec = benchmark.run_one(segmenter, spec, workdir, keep_predictions=True)
                    print(
                        f"[{time.perf_counter() - t0:7.1f}s] {rec.run_id:42s} J&F={rec.scores['J&F']:.3f} "
                        f"J={rec.scores['J_mean']:.3f} decay={rec.scores['J_decay']:+.3f} "
                        f"{rec.timing['ms_per_frame']:.0f} ms/frame",
                        flush=True,
                    )
                    # keep compact predictions for the report / demo videos
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        pred_dir / (rec.run_id.replace("/", "_").replace("|", "__") + ".npz"),
                        **{
                            f"o{o}_f{f}": np.packbits(mk) for o, per in rec.predictions.items() for f, mk in per.items()
                        },
                        shape=np.array([seq.height, seq.width]),
                    )
                    rec.predictions = None  # free RAM; predictions live in the .npz
                    records.append(rec)
                    benchmark.save_results(records, out_dir, extra={"environment": env}, append=args.append)

    # unlabeled clips: stability only
    unl = []
    for u in cfg.get("unlabeled", []):
        if only and u["sequence"] not in only:
            continue
        seq = _load_sequence(cfg["sequences"][u["sequence"]], seq_cache)
        unl.append(run_unlabeled(segmenter, seq, u, workdir, out_dir))
    benchmark.save_results(records, out_dir, extra={"environment": env, "unlabeled": unl}, append=args.append)
    # failure-detection quality of the GT-free signals, over *all* runs in out_dir
    analysis = benchmark.signal_quality(benchmark.read_per_frame(out_dir / "per_frame.csv"))
    benchmark.save_results(records, out_dir, extra={"signal_quality": analysis}, append=True)
    print(json.dumps(benchmark._clean(analysis), indent=2))


def _prompts_from_spec(objs: dict, pos_of: dict[int, int]) -> dict[int, Prompt]:
    prompts = {}
    for oid, o in objs.items():
        f = int(o.get("frame", 0))
        if "box" in o:
            prompts[int(oid)] = Prompt(kind="box", frame_idx=pos_of[f], box=np.array(o["box"], np.float32))
        else:
            pts = np.array(o["points"], np.float32)
            labels = np.array(o.get("labels", [1] * len(pts)), np.int32)
            prompts[int(oid)] = Prompt(kind="points", frame_idx=pos_of[f], points=pts, labels=labels)
    return prompts


def run_unlabeled(segmenter, seq, u: dict, workdir: Path, out_dir: Path) -> dict:
    stride = u.get("stride", 1)
    frame_dir = workdir / "frames" / f"{seq.name}_unl" / f"s{stride}"
    orig = data.materialize(seq, frame_dir, stride=stride, max_frames=u.get("max_frames"))
    pos_of = {o: k for k, o in enumerate(orig)}
    prompts = _prompts_from_spec(u["objects"], pos_of)
    res = segmenter.segment(frame_dir, prompts, video_key=f"{seq.name}_unl", frame_ids=orig)
    frames = {orig[k]: data.read_rgb(seq.frame_paths[orig[k]]) for k in range(len(orig))}
    flow_cache: dict = {}
    reps = []
    for o in sorted(res.masks):
        masks = {orig[k]: m for k, m in res.masks[o].items()}
        reps.append(
            stability.object_stability(
                o,
                masks,
                frames=frames,
                confidence={orig[k]: v for k, v in res.confidence[o].items()},
                uncertain_fraction={orig[k]: v for k, v in res.uncertain_fraction[o].items()},
                object_score={orig[k]: v for k, v in res.object_score[o].items()},
                flow_cache=flow_cache,
            ).as_dict()
        )
    vids = []
    for k in range(len(orig)):
        f = orig[k]
        vids.append(
            visualize.overlay(frames[f], {o: res.masks[o].get(k) for o in res.masks}, label=f"{seq.name}  frame {f}")
        )
    vpath = visualize.write_video(vids, out_dir / "videos" / f"{seq.name}_tracking.mp4", fps=max(1.0, seq.fps / stride))
    print(f"unlabeled {seq.name}: {len(orig)} frames, {res.fps:.2f} fps -> {vpath}")
    return {
        "sequence": seq.name,
        "stride": stride,
        "frames": len(orig),
        "objects": len(prompts),
        "stability": reps,
        "fps": res.fps,
        "ms_per_frame": 1000 * float(np.mean(res.frame_latency_s)),
        "video": str(vpath),
    }


# ---------------------------------------------------------------------------
# segment (the rotoscoping deliverable)
# ---------------------------------------------------------------------------


def _parse_obj_args(points: list[str], boxes: list[str]) -> dict:
    objs: dict[int, dict] = {}
    for p in points or []:  # "1:x,y" or "1:x,y:0" (label 0 = background click)
        parts = p.split(":")
        oid = int(parts[0])
        x, y = (float(v) for v in parts[1].split(","))
        label = int(parts[2]) if len(parts) > 2 else 1
        o = objs.setdefault(oid, {"points": [], "labels": []})
        o["points"].append([x, y])
        o["labels"].append(label)
    for b in boxes or []:  # "2:x0,y0,x1,y1"
        oid, coords = b.split(":")
        objs.setdefault(int(oid), {})["box"] = [float(v) for v in coords.split(",")]
    return objs


def cmd_segment(args) -> None:
    from src.vos.predictor import Sam2Config, Sam2VideoSegmenter

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq = data.load_unlabeled(args.video, cache_dir=out / "frames_in")
    objs = _parse_obj_args(args.point, args.box)
    if not objs:
        sys.exit("give at least one --point OBJ:X,Y or --box OBJ:X0,Y0,X1,Y1")
    seg = Sam2VideoSegmenter(Sam2Config(checkpoint=args.checkpoint, device=args.device, num_threads=args.threads))
    frame_dir = out / "frames_model"
    orig = data.materialize(seq, frame_dir, stride=args.stride, max_frames=args.max_frames)
    prompts = _prompts_from_spec(
        {k: {**v, "frame": args.frame} for k, v in objs.items()}, {o: k for k, o in enumerate(orig)}
    )
    t0 = time.perf_counter()
    res = seg.segment(frame_dir, prompts)
    elapsed = time.perf_counter() - t0
    masks = {o: {orig[k]: m for k, m in per.items()} for o, per in res.masks.items()}

    np.savez_compressed(
        out / "masks.npz",
        **{f"o{o}_f{f}": np.packbits(m) for o, per in masks.items() for f, m in per.items()},
        shape=np.array([seq.height, seq.width]),
    )
    for o, per in masks.items():
        visualize.write_alpha_pngs(seq.frame_paths, per, out / f"alpha_obj{o}")
    frames = [visualize.overlay(data.read_rgb(seq.frame_paths[f]), {o: masks[o].get(f) for o in masks}) for f in orig]
    visualize.write_video(frames, out / "overlay.mp4", fps=max(1.0, seq.fps / args.stride))
    reps = [
        stability.object_stability(
            o,
            per,
            confidence={orig[k]: v for k, v in res.confidence[o].items()},
            uncertain_fraction={orig[k]: v for k, v in res.uncertain_fraction[o].items()},
            object_score={orig[k]: v for k, v in res.object_score[o].items()},
        ).as_dict()
        for o, per in masks.items()
    ]
    report = {"frames": len(orig), "seconds": elapsed, "fps": res.fps, "objects": reps}
    (out / "report.json").write_text(json.dumps(benchmark._clean(report), indent=2))
    print(json.dumps(benchmark._clean(report), indent=2))


# ---------------------------------------------------------------------------
# rescore: recompute event metrics from saved predictions (no inference)
# ---------------------------------------------------------------------------


def cmd_rescore(args) -> None:
    from src.monitoring.drift_detector import MaskQualityMonitor
    from src.vos.report import load_predictions

    monitor = MaskQualityMonitor()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.results)
    doc = json.loads((out_dir / "runs.json").read_text())
    cache: dict = {}
    by_name = {}
    for key, spec in cfg["sequences"].items():
        if spec["source"] == "unlabeled":
            continue
        by_name[key] = spec
    for r in doc["runs"]:
        key = next(
            (k for k in by_name if r["sequence"] in (k, f"synthetic/{k}") or by_name[k].get("name") == r["sequence"]),
            None,
        )
        if key is None:
            print(f"skip {r['run_id']}: sequence not in config")
            continue
        seq = _load_sequence(by_name[key], cache)
        pred = load_predictions(out_dir, r["run_id"])
        frames = set(range(0, seq.num_frames, r["stride"]))
        events = []
        for o in sorted(pred):
            gt_frames = [f for f in seq.annotated_frames(o) if f in frames]
            ev = {"obj_id": o, **benchmark.visibility_events(seq.gt[o], pred.get(o, {}), gt_frames)}
            if seq.meta.get("distractor_masks") and o == 1:
                ev.update(benchmark.distractor_leakage(pred.get(o, {}), seq.meta["distractor_masks"], gt_frames))
            events.append(ev)
        r["events"] = benchmark._clean(events)
        # add metrics introduced after the run (no flow needed) and the monitor's verdict
        for st in r["stability"]:
            per = pred.get(st["obj_id"], {})
            idx = sorted(per)
            al = [
                stability.aligned_iou(per[a], per[b])
                for a, b in zip(idx, idx[1:], strict=False)
                if per[a].any() and per[b].any()
            ]
            al = [v for v in al if v == v]
            st["aligned_iou_mean"] = float(np.mean(al)) if al else None
            st["aligned_iou_p05"] = float(np.percentile(al, 5)) if al else None
        r["alerts"] = {str(st["obj_id"]): monitor.evaluate(st) for st in r["stability"]}
        print(f"rescored {r['run_id']}: alerts={r['alerts']}")
    for u in doc.get("unlabeled", []):
        u["alerts"] = {str(st["obj_id"]): monitor.evaluate(st) for st in u["stability"]}
    (out_dir / "runs.json").write_text(json.dumps(doc, indent=2))


# ---------------------------------------------------------------------------
# profile: cost model  t_frame ≈ t_encoder + N_objects · t_memory
# ---------------------------------------------------------------------------


def cmd_profile(args) -> None:
    from src.vos.predictor import Sam2Config, Sam2VideoSegmenter

    seg = Sam2VideoSegmenter(
        Sam2Config(checkpoint=args.checkpoint, device=args.device, num_threads=args.threads, feature_cache_frames=256)
    )
    seq = synthetic.generate("fast_motion", Path(args.workdir) / "profile_src", num_frames=args.frames)
    frame_dir = Path(args.workdir) / "profile_frames"
    orig = data.materialize(seq, frame_dir)
    rng = np.random.default_rng(0)

    def prompts(n: int) -> dict[int, Prompt]:
        pts = rng.uniform([60, 60], [seq.width - 60, seq.height - 60], size=(n, 2)).astype(np.float32)
        return {i + 1: Prompt(kind="points", points=pts[i : i + 1], labels=np.ones(1, np.int32)) for i in range(n)}

    def per_frame_ms(res) -> float:
        lat = np.array(res.frame_latency_s[1:])  # skip the prompted frame
        return float(np.median(lat) * 1000)

    cold = seg.segment(frame_dir, prompts(1), video_key="profile", frame_ids=orig)
    cold_ms = per_frame_ms(cold)
    warm = {}
    for n in args.objects:
        warm[n] = per_frame_ms(seg.segment(frame_dir, prompts(n), video_key="profile", frame_ids=orig))
        print(f"objects={n}: {warm[n]:.0f} ms/frame (encoder features cached)", flush=True)
    ns = np.array(sorted(warm), float)
    ts = np.array([warm[int(n)] for n in ns])
    slope, intercept = np.polyfit(ns, ts, 1) if len(ns) > 1 else (ts[0], 0.0)
    enc_ms = cold_ms - warm.get(1, ts[0])
    out = {
        "frames": args.frames,
        "threads": args.threads,
        "cold_ms_per_frame_1obj": cold_ms,
        "encoder_ms_per_frame": enc_ms,
        "warm_ms_per_frame": {str(k): v for k, v in warm.items()},
        "memory_ms_per_object_per_frame": float(slope),
        "fixed_overhead_ms": float(intercept),
        "model": seg.name,
        "projection_fps": {str(n): 1000.0 / (enc_ms + intercept + slope * n) for n in (1, 2, 4, 8, 16)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m src.vos.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch-weights")
    p.add_argument("--variant", default="tiny", choices=sorted(WEIGHTS))
    p.add_argument("--out", default="weights")
    p.set_defaults(func=cmd_fetch_weights)

    p = sub.add_parser("fetch-samples")
    p.add_argument("--out", default="data/samples")
    p.set_defaults(func=cmd_fetch_samples)

    p = sub.add_parser("synth")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenarios", nargs="*", choices=synthetic.SCENARIOS)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("benchmark")
    p.add_argument("--config", default="configs/phase1.yaml")
    p.add_argument("--out")
    p.add_argument("--only", help="comma-separated sequence keys to run")
    p.add_argument("--append", action="store_true", help="merge into existing results instead of overwriting")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("segment", help="rotoscope a video with clicks / boxes")
    p.add_argument("--video", required=True, help="mp4/mov file or folder of JPEG frames")
    p.add_argument("--point", action="append", help="OBJ:X,Y[:LABEL]  (repeatable)")
    p.add_argument("--box", action="append", help="OBJ:X0,Y0,X1,Y1")
    p.add_argument("--frame", type=int, default=0, help="frame where the prompts are given")
    p.add_argument("--checkpoint", default="weights/sam2.1_t.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--out", default="outputs/segment")
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("rescore", help="recompute event metrics from saved predictions")
    p.add_argument("--config", default="configs/phase1.yaml")
    p.add_argument("--results", default="results/phase1")
    p.set_defaults(func=cmd_rescore)

    p = sub.add_parser("profile", help="encoder vs per-object memory cost on this machine")
    p.add_argument("--checkpoint", default="weights/sam2.1_t.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--objects", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--workdir", default=".cache/vos_profile")
    p.add_argument("--out", default="results/phase1/profile.json")
    p.set_defaults(func=cmd_profile)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
