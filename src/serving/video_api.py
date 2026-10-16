"""HTTP layer for video object segmentation (SAM 2 models served from the registry)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session

from src.serving.database import get_db
from src.serving.loader import InvalidModelArtifact, ModelLoadError, UnsupportedModelFormat
from src.serving.observability import Observability
from src.serving.schemas import VideoSegmentationRequest, VideoSegmentationResponse
from src.serving.service import ArchivedModelError, LifecycleService, TaskMismatchError
from src.serving.storage import ArtifactNotFoundError
from src.serving.video_service import (
    InvalidPromptError,
    VideoDecodeError,
    VideoSegmentationService,
    VideoTooLargeError,
)

router = APIRouter(prefix="/models", tags=["Video Segmentation"])

_ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _parse_request(prompts: str, stride: int, max_frames: int | None, return_masks: bool, flow: bool):
    try:
        payload = json.loads(prompts)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"'prompts' is not valid JSON: {e.msg}") from e
    objects = payload.get("objects") if isinstance(payload, dict) else payload
    try:
        return VideoSegmentationRequest(
            objects=objects, stride=stride, max_frames=max_frames, return_masks=return_masks, flow_stability=flow
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json())) from e


def _run(db: Session, model_name: str, version: str, video: UploadFile, req: VideoSegmentationRequest):
    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        Observability.record_video_error(model_name, version, "unsupported_container")
        raise HTTPException(status_code=415, detail=f"unsupported video type '{suffix}'")
    tmp = Path(tempfile.mkdtemp(prefix="vos_upload_"))
    path = tmp / f"input{suffix}"
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(video.file, f, length=1024 * 1024)
        return VideoSegmentationService.segment_upload(db, model_name, version, path, req)
    except VideoTooLargeError as e:
        Observability.record_video_error(model_name, version, "too_large")
        raise HTTPException(status_code=413, detail=str(e)) from e
    except VideoDecodeError as e:
        Observability.record_video_error(model_name, version, "decode_error")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except InvalidPromptError as e:
        Observability.record_video_error(model_name, version, "invalid_prompt")
        raise HTTPException(status_code=422, detail=str(e)) from e
    except TaskMismatchError as e:
        Observability.record_video_error(model_name, version, "task_mismatch")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ArchivedModelError as e:
        Observability.record_video_error(model_name, version, "archived_model")
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ArtifactNotFoundError as e:
        Observability.record_video_error(model_name, version, "artifact_not_found")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (UnsupportedModelFormat, InvalidModelArtifact) as e:
        Observability.record_video_error(model_name, version, "invalid_model_artifact")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ModelLoadError as e:
        Observability.record_video_error(model_name, version, "model_load_error")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except HTTPException as e:
        if e.status_code == 404:
            Observability.record_video_error(model_name, version, "not_found")
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@router.post("/{model_name}/versions/{version}/segment-video", response_model=VideoSegmentationResponse)
def segment_video(
    model_name: str,
    version: str,
    video: UploadFile = File(..., description="mp4 / mov / avi / mkv / webm"),
    prompts: str = Form(
        ...,
        description='JSON: {"objects": [{"obj_id": 1, "frame_idx": 0, "points": [[x, y]], "labels": [1]}, '
        '{"obj_id": 2, "box": [x0, y0, x1, y1]}]}',
    ),
    stride: int = Form(1, ge=1, description="process every N-th frame"),
    max_frames: int | None = Form(None, ge=1),
    return_masks: bool = Form(True, description="include per-frame COCO RLE masks"),
    flow_stability: bool = Form(False, description="also compute optical-flow-compensated stability"),
    db: Session = Depends(get_db),
):
    """Track objects through an uploaded video with a SAM 2 model from the registry."""
    req = _parse_request(prompts, stride, max_frames, return_masks, flow_stability)
    return _run(db, model_name, version, video, req)


@router.post("/{model_name}/segment-video", response_model=VideoSegmentationResponse)
def segment_video_production(
    model_name: str,
    video: UploadFile = File(...),
    prompts: str = Form(...),
    stride: int = Form(1, ge=1),
    max_frames: int | None = Form(None, ge=1),
    return_masks: bool = Form(True),
    flow_stability: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Same as above, routed to the model's current *production* version."""
    prod = LifecycleService.get_production(db, model_name)
    req = _parse_request(prompts, stride, max_frames, return_masks, flow_stability)
    return _run(db, model_name, prod.version, video, req)
