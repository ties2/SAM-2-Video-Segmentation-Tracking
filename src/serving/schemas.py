# validation data

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# Schemas for Model Version
class ModelVersionCreate(BaseModel):
    version: str
    framework: str
    model_format: str
    artifact_uri: str
    status: str = "staging"


class ModelVersionResponse(ModelVersionCreate):
    id: int
    model_id: int
    created_at: datetime

    # allo to pydantic that read data directly from SQLAlchemy
    model_config = ConfigDict(from_attributes=True)


# Schemas for Model Registry
class ModelCreate(BaseModel):
    name: str
    description: str | None = None


class ModelResponse(ModelCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_empty_feature(cls, v):
        if not v:
            raise ValueError("Feature cannot be empty")
        return v


class PredictionResponse(BaseModel):
    model_name: str
    version: str
    prediction: Any


# ---------------------------------------------------------------------------
# Video segmentation (SAM 2)
# ---------------------------------------------------------------------------
class VideoObjectPrompt(BaseModel):
    """One object to track. Give ``points`` (+ ``labels``) and/or a ``box`` on ``frame_idx``."""

    obj_id: int
    frame_idx: int = 0
    points: list[list[float]] | None = None  # [[x, y], ...] in original-video pixels
    labels: list[int] | None = None  # 1 = foreground click, 0 = background click
    box: list[float] | None = None  # [x0, y0, x1, y1]

    @field_validator("obj_id")
    @classmethod
    def positive_id(cls, v):
        if v <= 0:
            raise ValueError("obj_id must be a positive integer (0 is background)")
        return v

    @field_validator("frame_idx")
    @classmethod
    def non_negative_frame(cls, v):
        if v < 0:
            raise ValueError("frame_idx must be >= 0")
        return v

    @field_validator("points")
    @classmethod
    def points_are_xy(cls, v):
        if v is not None and (not v or any(len(p) != 2 for p in v)):
            raise ValueError("points must be a non-empty list of [x, y] pairs")
        return v

    @field_validator("box")
    @classmethod
    def box_is_xyxy(cls, v):
        if v is not None and (len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]):
            raise ValueError("box must be [x0, y0, x1, y1] with x1 > x0 and y1 > y0")
        return v

    @model_validator(mode="after")
    def points_or_box(self):
        if self.points is None and self.box is None:
            raise ValueError("each object needs 'points' or 'box'")
        if self.points is not None:
            if self.labels is None:
                self.labels = [1] * len(self.points)
            if len(self.labels) != len(self.points):
                raise ValueError("'labels' must have one entry per point")
        return self


class VideoSegmentationRequest(BaseModel):
    objects: list[VideoObjectPrompt]
    stride: int = 1
    max_frames: int | None = None
    return_masks: bool = True
    flow_stability: bool = False

    @field_validator("objects")
    @classmethod
    def unique_objects(cls, v):
        if not v:
            raise ValueError("at least one object is required")
        ids = [o.obj_id for o in v]
        if len(ids) != len(set(ids)):
            raise ValueError("obj_id values must be unique")
        return v

    @field_validator("stride")
    @classmethod
    def stride_positive(cls, v):
        if v < 1:
            raise ValueError("stride must be >= 1")
        return v


class TrackedObject(BaseModel):
    obj_id: int
    stability: dict[str, Any]
    alerts: list[str]
    masks: dict[int, dict] | None = None  # frame_idx -> COCO RLE {"size": [h, w], "counts": [...]}


class VideoSegmentationResponse(BaseModel):
    model_name: str
    version: str
    video: dict[str, Any]
    processed_frames: list[int]
    objects: list[TrackedObject]
    timing: dict[str, float]
