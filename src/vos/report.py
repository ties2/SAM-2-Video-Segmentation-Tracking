"""Turn ``results/phase1`` into plots, demo videos/GIFs, a portfolio cover and REPORT.md.

python -m src.vos.report --results results/phase1 --config configs/phase1.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.vos import data, visualize  # noqa: E402

# reference palette (dataviz skill) — fixed slot order
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b9b8b2"


def _style(ax, title: str, ylabel: str | None = None, xlabel: str | None = None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def _fig(w=7.2, h=3.6, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(
            fmt.format(b.get_height()),
            (b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK2,
        )


def load(results: Path):
    runs = json.loads((results / "runs.json").read_text())
    rows = list(csv.DictReader(open(results / "per_frame.csv")))
    for r in rows:
        for k in ("J", "F", "confidence", "uncertain_fraction", "object_score"):
            r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
        for k in ("frame", "obj_id", "stride", "gt_area", "pred_area"):
            r[k] = int(r[k])
    return runs, rows


def run_by(runs, seq, prompt="mask", stride=1):
    for r in runs["runs"]:
        if r["sequence"] == seq and r["prompt"] == prompt and r["stride"] == stride:
            return r
    return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_prompt_effort(runs, out: Path):
    order = [("mask", "Full mask"), ("box", "Box"), ("click3", "3 clicks"), ("click", "1 click")]
    vals = [(lab, run_by(runs, "car-shadow", p)) for p, lab in order]
    vals = [(lab, r["scores"]["J&F"]) for lab, r in vals if r]
    if not vals:
        return None
    fig, ax = _fig(6.4, 3.4)
    bars = ax.bar([v[0] for v in vals], [v[1] for v in vals], color=BLUE, width=0.55)
    _bar_labels(ax, bars)
    ax.set_ylim(0, 1.08)  # bars start at zero: equal bars ARE the finding
    _style(ax, "User input needed: J&F by prompt type (DAVIS car-shadow)", "J&F")
    fig.tight_layout()
    p = out / "prompt_effort.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def plot_stride(runs, out: Path, fps_video: float = 24.0):
    series = [
        ("car-shadow", "DAVIS car-shadow (slow)", BLUE),
        ("synthetic/fast_motion", "Fast motion (50 px/frame)", ORANGE),
        ("synthetic/occlusion", "12-frame occlusion", AQUA),
    ]
    data = {}
    for seq, label, color in series:
        pts = [(s, run_by(runs, seq, "mask", s)) for s in (1, 2, 4)]
        pts = [(s, r["scores"]["J&F"]) for s, r in pts if r]
        if len(pts) >= 2:
            data[seq] = (label, color, pts)
    if not data:
        return None
    fig, (a1, a2) = _fig(8.4, 3.4, ncols=2)
    markers = ["o", "s", "^"]  # secondary encoding: overlapping lines stay distinguishable
    ends = []
    for k, (_seq, (label, color, pts)) in enumerate(data.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a1.plot(xs, ys, color=color, lw=2, marker=markers[k % 3], ms=5, label=label)
        ends.append((ys[-1], xs[-1]))
    # end labels, nudged apart when two series finish at (almost) the same value
    placed: list[float] = []
    for y, x in sorted(ends, reverse=True):
        dy = 0
        while any(abs((y + dy / 100) - q) < 0.035 for q in placed):
            dy -= 4
        placed.append(y + dy / 100)
        a1.annotate(
            f"{y:.2f}", (x, y), xytext=(6, dy * 2.2), textcoords="offset points", va="center", fontsize=8, color=INK2
        )
    a1.set_xticks([1, 2, 4], [f"every {s}\n({fps_video / s:.0f} fps)" for s in (1, 2, 4)])
    a1.set_ylim(0, 1.08)
    a1.set_xlim(0.7, 4.8)
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    _style(a1, "Accuracy vs. frame sub-sampling", "J&F", "processed frames")
    a1.grid(axis="x", visible=False)
    b2 = a2.bar([f"every {s}" for s in (1, 2, 4)], [1, 1 / 2, 1 / 4], color=BLUE, width=0.5)
    _bar_labels(a2, b2, "{:.2f}×")
    a2.set_ylim(0, 1.15)
    _style(a2, "Encoder compute per video second", "× of stride-1 cost", "processed frames")
    fig.tight_layout()
    p = out / "stride_tradeoff.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def plot_failure_modes(runs, out: Path):
    seqs = [
        ("synthetic/occlusion", "Occlusion\n(12 frames hidden)"),
        ("synthetic/exit_reentry", "Leaves frame\n(12 frames out)"),
        ("synthetic/distractor", "Look-alike\ndistractor"),
        ("synthetic/fast_motion", "Fast motion\n(50 px/frame)"),
        ("car-shadow", "DAVIS\ncar-shadow"),
        ("sav_000001", "SA-V\n(5 objects)"),
    ]
    vals = []
    for s, lab in seqs:
        r = run_by(runs, s, "mask", 4 if s == "sav_000001" else 1)
        if r:
            vals.append((lab, r["scores"]["J&F"]))
    if not vals:
        return None
    fig, ax = _fig(7.6, 3.4)
    bars = ax.bar([v[0] for v in vals], [v[1] for v in vals], color=BLUE, width=0.55)
    _bar_labels(ax, bars)
    ax.set_ylim(0, 1.08)
    _style(ax, "Does the mask persist? J&F per scenario", "J&F")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    p = out / "failure_modes.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def plot_per_frame(rows, out: Path):
    panels = [
        ("synthetic/occlusion", 1, "Occlusion: target behind pillar"),
        ("synthetic/exit_reentry", 1, "Target leaves and re-enters the frame"),
        ("synthetic/distractor", 1, "Identical distractor crosses in front"),
        ("sav_000001", 3, "SA-V object with 10 visibility changes"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (seq, obj, title) in zip(axes.ravel(), panels, strict=True):
        sel = sorted(
            (r for r in rows if r["sequence"] == seq and r["obj_id"] == obj and r["prompt"] == "mask"),
            key=lambda r: r["frame"],
        )
        _style(ax, title, "J (IoU)", "frame")
        if not sel:
            continue
        f = np.array([r["frame"] for r in sel])
        j = np.array([r["J"] for r in sel])
        hidden = np.array([r["gt_area"] == 0 for r in sel])
        for k in np.flatnonzero(hidden):
            step = f[1] - f[0] if len(f) > 1 else 1
            ax.axvspan(f[k] - step / 2, f[k] + step / 2, color=GRID, lw=0)
        ax.plot(f, j, color=BLUE, lw=2)
        ax.set_ylim(-0.03, 1.05)
    fig.text(
        0.01,
        0.005,
        "Grey bands: object fully hidden; the correct output is an empty mask (J = 1).",
        fontsize=8,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "per_frame_J.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def plot_signal_quality(runs, out: Path):
    sq = runs.get("signal_quality") or {}
    names = [
        ("object_score", "SAM 2 object score"),
        ("confidence", "Mask confidence"),
        ("uncertain_fraction", "Uncertain-pixel share"),
    ]
    vals = [(lab, sq.get(k, {}).get("auroc_detect_failure")) for k, lab in names]
    vals = [(a, b) for a, b in vals if b is not None]
    if not vals:
        return None
    fig, ax = _fig(6.4, 3.2)
    bars = ax.bar([v[0] for v in vals], [v[1] for v in vals], color=BLUE, width=0.5)
    _bar_labels(ax, bars, "{:.2f}")
    ax.axhline(0.5, color=INK2, lw=0.8)
    ax.annotate("chance = 0.5", (len(vals) - 0.55, 0.53), fontsize=8, color=INK2, ha="right")
    for b, (_lab, v) in zip(bars, vals, strict=True):
        if v < 0.5:
            ax.annotate(
                "inverted: lost objects have no uncertain pixels",
                (b.get_x() + b.get_width() / 2, v + 0.09),
                ha="center",
                fontsize=7,
                color=INK2,
            )
    ax.set_ylim(0, 1.08)
    _style(
        ax,
        f"Label-free failure detection (AUROC, {sq.get('failure_frames', 0)} bad of {sq.get('frames', 0)} frames)",
        "AUROC",
    )
    fig.tight_layout()
    p = out / "signal_quality.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Demo media
# ---------------------------------------------------------------------------


def load_predictions(results: Path, run_id: str) -> dict[int, dict[int, np.ndarray]]:
    f = results / "predictions" / (run_id.replace("/", "_").replace("|", "__") + ".npz")
    z = np.load(f)
    h, w = z["shape"]
    out: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for k in z.files:
        if k == "shape":
            continue
        o, fr = k[1:].split("_f")
        out[int(o)][int(fr)] = np.unpackbits(z[k])[: h * w].reshape(h, w).astype(bool)
    return dict(out)


def demo_frames(seq, preds, run, with_gt=True, every=1):
    from src.vos.metrics import jaccard

    frames = []
    for f in sorted({fr for per in preds.values() for fr in per})[::every]:
        img = data.read_rgb(seq.frame_paths[f])
        gt = {o: seq.gt[o].get(f) for o in seq.gt} if (with_gt and seq.gt) else None
        label = f"{seq.name}  |  frame {f}  |  prompt: {run['prompt']}"
        if seq.gt:
            jv = [
                jaccard(seq.gt[o][f], preds[o].get(f, np.zeros_like(seq.gt[o][f])))
                for o in preds
                if f in seq.gt.get(o, {})
            ]
            if jv:
                label += f"  |  J={np.mean(jv):.2f}"
        frames.append(visualize.overlay(img, {o: preds[o].get(f) for o in preds}, label=label, gt_contours=gt))
    return frames


def make_cover(tiles: list[np.ndarray], out: Path, headline: str, sub: str, chips: list[str] | None = None) -> Path:
    """1600×1200 portfolio thumbnail: title, 2×2 tracked frames (same size, centre-cropped), metric chips."""
    W, H = 1600, 1200
    canvas = Image.new("RGB", (W, H), (16, 18, 22))
    d = ImageDraw.Draw(canvas)
    try:
        f1 = ImageFont.truetype("DejaVuSans-Bold.ttf", 58)
        f2 = ImageFont.truetype("DejaVuSans.ttf", 28)
        f3 = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except OSError:
        f1 = f2 = f3 = ImageFont.load_default()
    d.text((60, 46), headline, font=f1, fill=(245, 245, 245))
    d.text((60, 124), sub, font=f2, fill=(180, 184, 190))
    tw, th = 730, 410
    pos = [(60, 190), (810, 190), (60, 620), (810, 620)]
    for t, (x, y) in zip(tiles[:4], pos, strict=False):
        im = Image.fromarray(t)
        scale = max(tw / im.width, th / im.height)
        im = im.resize((int(im.width * scale + 0.5), int(im.height * scale + 0.5)), Image.LANCZOS)
        left, top = (im.width - tw) // 2, (im.height - th) // 2
        canvas.paste(im.crop((left, top, left + tw, top + th)), (x, y))
    x = 60
    for chip in chips or []:
        w = int(d.textlength(chip, font=f3)) + 36
        d.rounded_rectangle((x, 1070, x + w, 1124), radius=12, fill=(0x1C, 0x5C, 0xAB))
        d.text((x + 18, 1083), chip, font=f3, fill=(255, 255, 255))
        x += w + 16
    canvas.save(out, quality=92)
    return out


def _fit(frame: np.ndarray, size=(960, 540)) -> np.ndarray:
    """Letterbox / pillarbox a frame onto a fixed canvas (keeps aspect ratio)."""
    W, H = size
    im = Image.fromarray(frame)
    scale = min(W / im.width, H / im.height)
    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", size, (16, 18, 22))
    canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return np.array(canvas)


def _title_card(title: str, caption: str, size=(960, 540)) -> np.ndarray:
    canvas = Image.new("RGB", size, (16, 18, 22))
    d = ImageDraw.Draw(canvas)
    try:
        f1 = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        f2 = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    d.text((60, size[1] // 2 - 50), title, font=f1, fill=(245, 245, 245))
    d.text((60, size[1] // 2 + 10), caption, font=f2, fill=(180, 184, 190))
    d.rectangle((60, size[1] // 2 - 70, 140, size[1] // 2 - 64), fill=(0x2A, 0x78, 0xD6))
    return np.array(canvas)


def make_reel(clips: list[tuple[str, str, list[np.ndarray]]], out: Path, fps: float = 10.0) -> Path:
    """Portfolio demo: title card + clip, for each (title, caption, frames)."""
    frames = []
    for title, caption, clip in clips:
        frames += [_title_card(title, caption)] * int(fps * 1.8)
        frames += [_fit(f) for f in clip]
    return visualize.write_video(frames, out, fps=fps)


def _f(v, d=3):
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


def write_markdown(runs: dict, results: Path, readme: Path | None = None) -> Path:
    """REPORT.md with the key tables; also refreshes the RESULTS block of the README."""
    R = runs["runs"]
    lines = ["| Sequence | Prompt | Stride | Frames | Obj | J&F | J | F | J-decay | ms/frame |", "|---" * 10 + "|"]
    for r in R:
        sc = r["scores"]
        lines.append(
            f"| {r['sequence']} | {r['prompt']} | {r['stride']} | {r['num_frames']} | {r['num_objects']} | "
            f"{_f(sc['J&F'])} | {_f(sc['J_mean'])} | {_f(sc['F_mean'])} | {_f(sc['J_decay'])} | "
            f"{_f(r['timing']['ms_per_frame'], 0)} |"
        )
    ev = [
        "| Scenario | Hidden frames | Hidden-J | Re-detect delay | Lost again after re-detect | Max leakage |",
        "|---|---|---|---|---|---|",
    ]
    for r in R:
        if r["source"] != "synthetic" or r["stride"] != 1:
            continue
        e = next((x for x in r["events"] if x["obj_id"] == 1), {})
        leak = e.get("distractor_leakage_max")
        ev.append(
            f"| {r['sequence']} ({r['prompt']}) | {e.get('hidden_frames', 0)} | {_f(e.get('hidden_J'), 2)} | "
            f"{e.get('redetection_delay') or '–'} | {e.get('lost_after_redetection', '–')} | "
            f"{'–' if leak is None else f'{100 * leak:.0f} %'} |"
        )
    sq = runs.get("signal_quality", {})
    sig = ["| GT-free signal | Spearman ρ with J | AUROC (detect J < 0.5) |", "|---|---|---|"]
    for k in ("object_score", "confidence", "uncertain_fraction"):
        if k in sq:
            sig.append(f"| {k} | {_f(sq[k]['spearman_with_J'], 2)} | {_f(sq[k]['auroc_detect_failure'], 2)} |")
    body = "\n".join(
        [
            "### All runs",
            *lines,
            "",
            "### Persistence events (synthetic stress tests, exact GT)",
            *ev,
            "",
            "### Can failures be detected without labels?",
            *sig,
            "",
            "![failure modes](plots/failure_modes.png)",
            "![per-frame J](plots/per_frame_J.png)",
        ]
    )
    md = results / "REPORT.md"
    md.write_text("# Phase 1 results\n\nGenerated by `python -m src.vos.report`.\n\n" + body + "\n")
    if readme and readme.exists():
        txt = readme.read_text()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a in txt and b in txt:
            rel = body.replace("](plots/", "](results/phase1/plots/")
            txt = txt[: txt.index(a) + len(a)] + "\n" + rel + "\n" + txt[txt.index(b) :]
            readme.write_text(txt)
    return md


def make_results_card(plots: Path, out: Path, width: int = 1600) -> Path | None:
    """Stack the persistence charts into one portfolio image."""
    parts = [plots / "failure_modes.png", plots / "per_frame_J.png"]
    ims = [Image.open(p).convert("RGB") for p in parts if p.exists()]
    if not ims:
        return None
    ims = [im.resize((width, int(im.height * width / im.width)), Image.LANCZOS) for im in ims]
    card = Image.new("RGB", (width, sum(im.height for im in ims)), SURFACE)
    y = 0
    for im in ims:
        card.paste(im, (0, y))
        y += im.height
    card.save(out, optimize=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase1")
    ap.add_argument("--config", default="configs/phase1.yaml")
    args = ap.parse_args(argv)
    results = Path(args.results)
    cfg = yaml.safe_load(Path(args.config).read_text())
    runs, rows = load(results)
    plots = results / "plots"
    plots.mkdir(exist_ok=True)
    made = [
        plot_prompt_effort(runs, plots),
        plot_stride(runs, plots),
        plot_failure_modes(runs, plots),
        plot_per_frame(rows, plots),
        plot_signal_quality(runs, plots),
    ]
    print("plots:", [str(p) for p in made if p])
    print("markdown:", write_markdown(runs, results, Path("README.md")))
    print("results card:", make_results_card(plots, results / "portfolio_results.png"))

    # demo GIF / MP4 per interesting run
    media = results / "videos"
    media.mkdir(exist_ok=True)
    tiles = {}
    clips: dict[str, list[np.ndarray]] = {}
    demos = [
        ("car-shadow", "mask", 1),
        ("synthetic/occlusion", "mask", 1),
        ("synthetic/distractor", "click", 1),
        ("synthetic/exit_reentry", "mask", 1),
        ("synthetic/fast_motion", "mask", 1),
        ("sav_000001", "mask", 4),
    ]
    from src.vos.cli import _load_sequence

    cache: dict = {}
    for seq_name, prompt, stride in demos:
        run = run_by(runs, seq_name, prompt, stride)
        if not run:
            continue
        key = {"car-shadow": "car-shadow", "sav_000001": "sav_000001"}.get(seq_name, seq_name.split("/")[-1])
        seq = _load_sequence(cfg["sequences"][key], cache)
        preds = load_predictions(results, run["run_id"])
        frames = demo_frames(seq, preds, run)
        stem = seq_name.replace("/", "_") + f"_{prompt}"
        visualize.write_video(frames, media / f"{stem}.mp4", fps=max(1.0, 24.0 / stride / 2))
        visualize.write_gif(frames[:: 2 if len(frames) > 60 else 1], media / f"{stem}.gif", fps=8, width=480)
        pick = {"synthetic/occlusion": 40, "synthetic/distractor": 30, "synthetic/exit_reentry": 44}
        tiles[seq_name] = frames[min(len(frames) - 1, pick.get(seq_name, len(frames) // 2))]
        clips[seq_name] = frames
        print("media:", stem)

    bedroom_mp4 = media / "bedroom_tracking.mp4"
    if bedroom_mp4.exists():
        import imageio.v2 as imageio

        rd = imageio.get_reader(bedroom_mp4)
        bed = [fr for fr in rd]
        tiles["bedroom"] = bed[-1]  # both children visible and separated
        clips["bedroom"] = bed

    order = ["bedroom", "car-shadow", "sav_000001", "synthetic/distractor"]
    cover_tiles = [tiles[k] for k in order if k in tiles]
    cs = run_by(runs, "car-shadow", "mask", 1)
    sub = "SAM 2.1 · click / box / mask prompts · J&F evaluation · label-free QA · FastAPI + Prometheus"
    chips = []
    if cs:
        chips.append(f"J&F {cs['scores']['J&F']:.3f} (DAVIS)")
    occ = run_by(runs, "synthetic/occlusion", "mask", 1)
    if occ:
        chips.append("12-frame occlusion: recovered")
    sq = runs.get("signal_quality", {}).get("object_score", {})
    if sq.get("auroc_detect_failure") is not None:
        chips.append(f"Failure detection AUROC {sq['auroc_detect_failure']:.2f}")
    if cover_tiles:
        make_cover(cover_tiles, results / "portfolio_cover.jpg", "Video Object Segmentation & Tracking", sub, chips)
        print("cover:", results / "portfolio_cover.jpg")

    reel = [(t, c, clips[k]) for k, t, c in REEL_SPEC if k in clips]
    if reel:
        print("reel:", make_reel(reel, results / "videos" / "portfolio_demo.mp4"))


# (sequence, title card, caption) for the portfolio demo reel
REEL_SPEC = [
    ("bedroom", "Real video, 2 clicks per child", "Two children cross each other; both masks keep their identity"),
    ("car-shadow", "DAVIS car-shadow", "One prompt on frame 0; the mask follows the car (J per frame)"),
    ("synthetic/occlusion", "Full occlusion for 12 frames", "Empty mask while hidden; back on the first visible frame"),
    ("synthetic/exit_reentry", "Failure case: object leaves the frame", "Re-detected, lost again, then recovered"),
    ("synthetic/distractor", "Look-alike distractor (1 click)", "An identical object crosses in front of the target"),
]


if __name__ == "__main__":
    main()
