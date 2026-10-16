"""Video segmentation API, end-to-end through registry → storage → loader → cache → service.

A fake segmenter is registered in the loader registry so these tests run in
milliseconds without torch; the real SAM 2 path is covered by test_sam2_integration.py.
"""

import io
import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving import loader as loader_module
from src.serving.app import app
from src.serving.cache import InMemoryModelCache, model_cache
from src.serving.dependency import get_artifact_storage
from src.serving.observability import MASK_ALERTS, VIDEO_FRAMES, VIDEO_REQUESTS
from src.vos.types import SegmentationResult

client = TestClient(app)


class FakeSegmenter:
    """Puts a 20-px disk around the first click (or box centre) on every frame."""

    name = "fake-vos"

    def segment(self, frame_dir, prompts, video_key=None, frame_ids=None):
        frames = sorted(p for p in __import__("pathlib").Path(frame_dir).glob("*.jpg"))
        h, w = cv2.imread(str(frames[0])).shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        res = SegmentationResult(masks={}, confidence={}, uncertain_fraction={}, object_score={})
        for oid, p in prompts.items():
            if p.points is not None:
                cx, cy = p.points[0]
            else:
                cx, cy = (p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= 20**2
            for k in range(p.frame_idx, len(frames)):
                res.masks.setdefault(oid, {})[k] = m
                res.confidence.setdefault(oid, {})[k] = 0.95
                res.uncertain_fraction.setdefault(oid, {})[k] = 0.01
                res.object_score.setdefault(oid, {})[k] = 5.0
        for k in range(len(frames)):
            res.frame_latency_s.append(0.001)
            res.processed_frames.append(k)
        return res


class FakeLoader:
    def load(self, artifact_bytes):
        assert artifact_bytes == b"fake-checkpoint"
        return FakeSegmenter()


def _video_bytes(n=12, w=160, h=96) -> bytes:
    import tempfile

    path = tempfile.mktemp(suffix=".mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
    for t in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(img, (30 + 5 * t, 48), 15, (0, 140, 255), -1)
        vw.write(img)
    vw.release()
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def video_model():
    loader_module.LOADER_REGISTRY[("fake-vos", "test")] = FakeLoader
    name = f"vos-model-{int(time.time() * 1000)}"
    uri = f"local://{name}/1.0.0/fake.pt"
    get_artifact_storage().save(uri, b"fake-checkpoint")
    client.post("/models", json={"name": name, "description": "fake video segmenter"})
    r = client.post(
        f"/models/{name}/versions",
        json={"version": "1.0.0", "framework": "fake-vos", "model_format": "test", "artifact_uri": uri},
    )
    assert r.status_code == 200, r.text
    yield name
    loader_module.LOADER_REGISTRY.pop(("fake-vos", "test"), None)
    model_cache.clear()


def _post(name, prompts, version="1.0.0", data=None, video=None, filename="clip.mp4"):
    return client.post(
        f"/models/{name}/versions/{version}/segment-video",
        files={"video": (filename, io.BytesIO(video if video is not None else _video_bytes()), "video/mp4")},
        data={"prompts": json.dumps(prompts), **(data or {})},
    )


def test_segment_video_happy_path(video_model):
    before = VIDEO_FRAMES.labels(model=video_model, version="1.0.0")._value.get()
    r = _post(video_model, {"objects": [{"obj_id": 1, "points": [[30, 48]]}, {"obj_id": 2, "box": [60, 20, 100, 70]}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["video"]["total_frames"] == 12
    assert body["processed_frames"] == list(range(12))
    assert [o["obj_id"] for o in body["objects"]] == [1, 2]
    obj = body["objects"][0]
    assert obj["stability"]["tiou_mean"] == pytest.approx(1.0)
    assert obj["alerts"] == []
    rle0 = obj["masks"]["0"]
    assert rle0["size"] == [96, 160] and sum(rle0["counts"]) == 96 * 160
    assert VIDEO_FRAMES.labels(model=video_model, version="1.0.0")._value.get() == before + 12


def test_stride_and_no_masks(video_model):
    r = _post(
        video_model, {"objects": [{"obj_id": 1, "points": [[30, 48]]}]}, data={"stride": "3", "return_masks": "false"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["processed_frames"] == [0, 3, 6, 9]
    assert body["objects"][0]["masks"] is None


def test_prompt_on_unprocessed_frame_is_422(video_model):
    r = _post(video_model, {"objects": [{"obj_id": 1, "frame_idx": 1, "points": [[30, 48]]}]}, data={"stride": "2"})
    assert r.status_code == 422
    assert "not processed" in r.json()["detail"]


def test_point_outside_frame_is_422(video_model):
    r = _post(video_model, {"objects": [{"obj_id": 1, "points": [[999, 48]]}]})
    assert r.status_code == 422


@pytest.mark.parametrize(
    "prompts",
    [
        {"objects": []},
        {"objects": [{"obj_id": 0, "points": [[1, 1]]}]},
        {"objects": [{"obj_id": 1}]},
        {"objects": [{"obj_id": 1, "points": [[1, 1]], "labels": [1, 0]}]},
        {"objects": [{"obj_id": 1, "box": [10, 10, 5, 20]}]},
        {"objects": [{"obj_id": 1, "points": [[1, 1]]}, {"obj_id": 1, "points": [[2, 2]]}]},
    ],
)
def test_invalid_prompts_are_422(video_model, prompts):
    assert _post(video_model, prompts).status_code == 422


def test_bad_json_and_bad_container(video_model):
    r = client.post(
        f"/models/{video_model}/versions/1.0.0/segment-video",
        files={"video": ("clip.mp4", io.BytesIO(_video_bytes()), "video/mp4")},
        data={"prompts": "{not json"},
    )
    assert r.status_code == 422
    assert _post(video_model, {"objects": [{"obj_id": 1, "points": [[1, 1]]}]}, filename="clip.gif").status_code == 415
    assert _post(video_model, {"objects": [{"obj_id": 1, "points": [[1, 1]]}]}, video=b"not a video").status_code == 400


def test_too_many_frames_is_413(video_model, monkeypatch):
    from src.serving.config import settings

    monkeypatch.setattr(settings, "vos_max_frames", 5)
    before = VIDEO_REQUESTS.labels(model=video_model, version="1.0.0", status="error")._value.get()
    r = _post(video_model, {"objects": [{"obj_id": 1, "points": [[30, 48]]}]})
    assert r.status_code == 413
    assert VIDEO_REQUESTS.labels(model=video_model, version="1.0.0", status="error")._value.get() == before + 1


def test_unknown_model_is_404():
    assert _post("does-not-exist", {"objects": [{"obj_id": 1, "points": [[1, 1]]}]}).status_code == 404


def test_tabular_model_rejects_video_and_video_model_rejects_predict(video_model):
    # /predict on a video model → 400 (task mismatch), not a 500
    r = client.post(f"/models/{video_model}/versions/1.0.0/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 400


def test_production_routing(video_model):
    assert client.post(f"/models/{video_model}/versions/1.0.0/promote").status_code == 200
    r = client.post(
        f"/models/{video_model}/segment-video",
        files={"video": ("clip.mp4", io.BytesIO(_video_bytes()), "video/mp4")},
        data={"prompts": json.dumps({"objects": [{"obj_id": 1, "points": [[30, 48]]}]})},
    )
    assert r.status_code == 200, r.text
    assert r.json()["version"] == "1.0.0"


def test_quality_alert_metric_increments(video_model, monkeypatch):
    from src.monitoring.drift_detector import MaskQualityMonitor, QualityThresholds
    from src.serving import video_service

    # the fake segmenter reports confidence 0.95; demand 0.99 so the monitor raises an alert
    monkeypatch.setattr(video_service, "_monitor", MaskQualityMonitor(QualityThresholds(min_confidence=0.99)))
    before = MASK_ALERTS.labels(model=video_model, version="1.0.0", reason="low_confidence")._value.get()
    assert _post(video_model, {"objects": [{"obj_id": 1, "points": [[30, 48]]}]}).status_code == 200
    assert MASK_ALERTS.labels(model=video_model, version="1.0.0", reason="low_confidence")._value.get() == before + 1


def test_capabilities_lists_sam2():
    loaders = client.get("/capabilities").json()["loaders"]
    assert {"framework": "sam2", "model_format": "pytorch-checkpoint", "task": "video-segmentation"} in loaders


# ---------------- LRU model cache ----------------------------------------------------


def test_lru_cache_evicts_least_recently_used():
    c = InMemoryModelCache(max_items=2)
    c.set("a", "1", "A")
    c.set("b", "1", "B")
    assert c.get("a", "1") == "A"  # touch a → b becomes LRU
    c.set("c", "1", "C")
    assert c.contains("a", "1") and c.contains("c", "1") and not c.contains("b", "1")
    assert c.evictions == 1


def test_unbounded_cache_keeps_everything():
    c = InMemoryModelCache(max_items=None)
    for i in range(50):
        c.set("m", str(i), i)
    assert len(c) == 50
