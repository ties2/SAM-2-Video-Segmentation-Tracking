"""Real SAM 2.1 inference on a few synthetic frames (skipped without torch / weights).

Run with:  pytest -m slow
"""

import os
from pathlib import Path

import numpy as np
import pytest

CKPT = Path(os.environ.get("SAM2_CHECKPOINT", "weights/sam2.1_t.pt"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not CKPT.exists(), reason=f"SAM 2 checkpoint not found at {CKPT}"),
]
torch = pytest.importorskip("torch")
pytest.importorskip("sam2")


@pytest.fixture(scope="module")
def segmenter():
    from src.vos.predictor import Sam2Config, Sam2VideoSegmenter

    return Sam2VideoSegmenter(Sam2Config(checkpoint=str(CKPT), num_threads=2, feature_cache_frames=8))


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory):
    from src.vos import data, synthetic

    root = tmp_path_factory.mktemp("synth")
    seq = synthetic.generate("fast_motion", root, num_frames=4)
    d = root / "frames"
    orig = data.materialize(seq, d)
    return seq, d, orig


def test_variant_is_inferred_from_weights():
    from src.vos.predictor import infer_variant, load_state_dict

    variant, cfg = infer_variant(load_state_dict(CKPT))
    assert variant in {"tiny", "small", "base_plus", "large"}
    assert cfg.endswith(".yaml")


def test_mask_prompt_tracks_object(segmenter, short_clip):
    from src.vos import metrics
    from src.vos.prompts import prompt_from_mask

    seq, d, orig = short_clip
    res = segmenter.segment(d, {1: prompt_from_mask(seq.gt[1][0], "mask")}, video_key="clip", frame_ids=orig)
    assert sorted(res.masks[1]) == [0, 1, 2, 3]
    score = metrics.evaluate_sequence("clip", seq.gt, res.masks, skip_first_last=False)
    assert score.summary()["J_mean"] > 0.7
    assert all(v is not None for v in res.object_score[1].values())


def test_feature_cache_gives_identical_masks(segmenter, short_clip):
    from src.vos.prompts import prompt_from_mask

    seq, d, orig = short_clip
    p = {1: prompt_from_mask(seq.gt[1][0], "click")}
    segmenter.feature_cache.clear()
    a = segmenter.segment(d, p, video_key="clip2", frame_ids=orig)
    hits_before = segmenter.feature_cache.hits
    b = segmenter.segment(d, p, video_key="clip2", frame_ids=orig)
    assert segmenter.feature_cache.hits > hits_before
    for k in a.masks[1]:
        assert np.array_equal(a.masks[1][k], b.masks[1][k])


def test_registry_loader_builds_segmenter_from_bytes():
    from src.serving.loader import Sam2CheckpointLoader

    seg = Sam2CheckpointLoader().load(CKPT.read_bytes())
    assert hasattr(seg, "segment") and seg.num_params > 1_000_000


def test_registry_loader_rejects_garbage():
    from src.serving.loader import InvalidModelArtifact, Sam2CheckpointLoader

    with pytest.raises(InvalidModelArtifact):
        Sam2CheckpointLoader().load(b"definitely not a checkpoint")


def test_video_api_end_to_end_with_real_sam2(tmp_path):
    """Registry → storage → Sam2CheckpointLoader → LRU cache → /segment-video, with real inference."""
    import io
    import json
    import time

    import cv2
    from fastapi.testclient import TestClient

    from src.serving.app import app
    from src.serving.cache import model_cache
    from src.serving.dependency import get_artifact_storage
    from src.vos import synthetic

    seq = synthetic.generate("occlusion", tmp_path / "synth", num_frames=6)
    video = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (seq.width, seq.height))
    for p in seq.frame_paths:
        vw.write(cv2.imread(str(p)))
    vw.release()

    client = TestClient(app)
    name = f"sam2-e2e-{int(time.time())}"
    uri = f"local://{name}/2.1-tiny/model.pt"
    get_artifact_storage().save(uri, CKPT.read_bytes())
    assert client.post("/models", json={"name": name}).status_code == 200
    r = client.post(
        f"/models/{name}/versions",
        json={"version": "2.1-tiny", "framework": "sam2", "model_format": "pytorch-checkpoint", "artifact_uri": uri},
    )
    assert r.status_code == 200, r.text

    ys, xs = np.nonzero(seq.gt[2][0])  # the never-occluded control object
    prompts = {"objects": [{"obj_id": 2, "points": [[float(xs.mean()), float(ys.mean())]]}]}
    r = client.post(
        f"/models/{name}/versions/2.1-tiny/segment-video",
        files={"video": ("clip.mp4", io.BytesIO(video.read_bytes()), "video/mp4")},
        data={"prompts": json.dumps(prompts)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["processed_frames"] == list(range(6))
    obj = body["objects"][0]
    assert obj["stability"]["empty_frames"] == 0
    from src.vos import metrics, rle

    j = [metrics.jaccard(seq.gt[2][int(f)], rle.decode(m)) for f, m in obj["masks"].items()]
    assert np.mean(j) > 0.8
    model_cache.clear()
