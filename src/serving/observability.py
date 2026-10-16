import json
import logging
from contextvars import ContextVar

from prometheus_client import Counter, Histogram

# Simple logger configuration (in real projects, libraries like structlog can be used)
logger = logging.getLogger("msp_observability")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)

# Save Request ID in the current execution context
# request_id_var = contextvars.ContextVar("request_id", default="unknown")
request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="")

# Metrics Definition
PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total prediction requests", ["model", "version", "status"])

PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction errors", ["model", "version", "error_type"])

PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds", ["model", "version"])

MODEL_LOAD_TOTAL = Counter(
    "model_load_total", "Total model loads from artifact storage", ["model", "version", "status"]
)

MODEL_LOAD_LATENCY = Histogram("model_load_latency_seconds", "Model loading latency in seconds", ["model", "version"])

CACHE_HITS = Counter("model_cache_hits_total", "Total model cache hits", ["model", "version"])

CACHE_MISSES = Counter("model_cache_misses_total", "Total model cache misses", ["model", "version"])


MODEL_PROMOTIONS = Counter("model_promotions_total", "Total model promotions", ["model", "version", "status"])

# --- Video segmentation (SAM 2) metrics -------------------------------------------------
# Video jobs take seconds to minutes, so the default Prometheus buckets (5 ms … 10 s) would
# put almost every request in +Inf; explicit buckets keep the p95 panel meaningful.
VIDEO_REQUESTS = Counter(
    "video_segmentation_requests_total", "Video segmentation requests", ["model", "version", "status"]
)
VIDEO_ERRORS = Counter(
    "video_segmentation_errors_total", "Video segmentation errors", ["model", "version", "error_type"]
)
VIDEO_LATENCY = Histogram(
    "video_segmentation_latency_seconds",
    "End-to-end latency of a video segmentation request",
    ["model", "version"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
)
VIDEO_FRAME_LATENCY = Histogram(
    "video_frame_latency_seconds",
    "Propagation latency per processed frame",
    ["model", "version"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8),
)
VIDEO_FRAMES = Counter("video_frames_processed_total", "Frames processed by the video segmenter", ["model", "version"])
# Ground-truth-free mask quality, observed once per tracked object per request
MASK_TEMPORAL_IOU = Histogram(
    "mask_temporal_iou",
    "Mean IoU between consecutive masks of a tracked object (1 = perfectly stable)",
    ["model", "version"],
    buckets=(0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0),
)
MASK_EMPTY_RATIO = Histogram(
    "mask_empty_frame_ratio",
    "Share of processed frames where a tracked object has an empty mask",
    ["model", "version"],
    buckets=(0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0),
)
MASK_ALERTS = Counter(
    "mask_quality_alerts_total", "Mask quality alerts raised by the GT-free monitor", ["model", "version", "reason"]
)


# Abstraction Layer
class Observability:
    @staticmethod
    def set_request_id(request_id: str):
        request_id_ctx_var.set(request_id)

    @staticmethod
    def get_request_id() -> str:
        return request_id_ctx_var.get()

    @staticmethod
    def log_prediction_event(model: str, version: str, status: str, latency_ms: float = None, error_type: str = None):
        """Generate structured JSON log without PII"""
        log_data = {
            "event": "prediction",
            "request_id": Observability.get_request_id(),
            "model_name": model,
            "version": version,
            "status": status,
        }
        if latency_ms is not None:
            log_data["latency_ms"] = round(latency_ms, 2)
        if error_type is not None:
            log_data["error_type"] = error_type

        logger.info(json.dumps(log_data))

    # --- Metrics Wrapper Methods ---
    @staticmethod
    def record_prediction_success(model: str, version: str, latency_sec: float):
        PREDICTION_REQUESTS.labels(model=model, version=version, status="success").inc()
        PREDICTION_LATENCY.labels(model=model, version=version).observe(latency_sec)
        Observability.log_prediction_event(model, version, "success", latency_ms=latency_sec * 1000)

    @staticmethod
    def record_prediction_error(model: str, version: str, error_type: str):
        PREDICTION_REQUESTS.labels(model=model, version=version, status="error").inc()
        PREDICTION_ERRORS.labels(model=model, version=version, error_type=error_type).inc()
        Observability.log_prediction_event(model, version, "error", error_type=error_type)

    @staticmethod
    def record_cache_hit(model: str, version: str):
        CACHE_HITS.labels(model=model, version=version).inc()

    @staticmethod
    def record_cache_miss(model: str, version: str):
        CACHE_MISSES.labels(model=model, version=version).inc()

    @staticmethod
    def record_model_load(model: str, version: str, status: str, latency_sec: float = 0.0):
        MODEL_LOAD_TOTAL.labels(model=model, version=version, status=status).inc()
        if status == "success":
            MODEL_LOAD_LATENCY.labels(model=model, version=version).observe(latency_sec)

    # --- Video segmentation ------------------------------------------------------------
    @staticmethod
    def record_video_success(model: str, version: str, latency_sec: float, frame_latencies: list[float]):
        VIDEO_REQUESTS.labels(model=model, version=version, status="success").inc()
        VIDEO_LATENCY.labels(model=model, version=version).observe(latency_sec)
        VIDEO_FRAMES.labels(model=model, version=version).inc(len(frame_latencies))
        hist = VIDEO_FRAME_LATENCY.labels(model=model, version=version)
        for t in frame_latencies:
            hist.observe(t)
        logger.info(
            json.dumps(
                {
                    "event": "video_segmentation",
                    "request_id": Observability.get_request_id(),
                    "model_name": model,
                    "version": version,
                    "status": "success",
                    "frames": len(frame_latencies),
                    "latency_ms": round(latency_sec * 1000, 1),
                }
            )
        )

    @staticmethod
    def record_video_error(model: str, version: str, error_type: str):
        VIDEO_REQUESTS.labels(model=model, version=version, status="error").inc()
        VIDEO_ERRORS.labels(model=model, version=version, error_type=error_type).inc()
        logger.info(
            json.dumps(
                {
                    "event": "video_segmentation",
                    "request_id": Observability.get_request_id(),
                    "model_name": model,
                    "version": version,
                    "status": "error",
                    "error_type": error_type,
                }
            )
        )

    @staticmethod
    def record_mask_quality(model: str, version: str, tiou_mean: float | None, empty_rate: float | None, alerts):
        if tiou_mean is not None:
            MASK_TEMPORAL_IOU.labels(model=model, version=version).observe(tiou_mean)
        if empty_rate is not None:
            MASK_EMPTY_RATIO.labels(model=model, version=version).observe(empty_rate)
        for reason in alerts:
            MASK_ALERTS.labels(model=model, version=version, reason=reason).inc()

    @staticmethod
    def record_promotion(model_name: str, version: str, status: str, error_type: str = None):
        MODEL_PROMOTIONS.labels(model=model_name, version=version, status=status).inc()
        log_data = {
            "event": "model_promotion",
            "request_id": request_id_ctx_var.get(),
            "model_name": model_name,
            "version": version,
            "status": status,
        }
        if error_type:
            log_data["error_type"] = error_type
        logger.info(json.dumps(log_data))
