import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.serving.api import router as models_router
from src.serving.database import Base, engine
from src.serving.loader import LOADER_REGISTRY, TASK_BY_FRAMEWORK
from src.serving.observability import Observability
from src.serving.video_api import router as video_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables when the application starts (not during import)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="ML Model Serving Platform — Video Segmentation & Tracking",
    description="Model registry + serving for scikit-learn models and SAM 2 video object segmentation.",
    lifespan=lifespan,
)


# Middleware to generate and set Request ID
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    Observability.set_request_id(req_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.get("/metrics", tags=["Observability"])
def get_metrics():
    """Standard and Prometheus-compatible output"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


@app.get("/capabilities", tags=["Health"])
def capabilities():
    """Which (framework, model_format) pairs this deployment can load, and what they do."""
    return {
        "loaders": [
            {"framework": f, "model_format": m, "task": TASK_BY_FRAMEWORK.get(f, "unknown")} for f, m in LOADER_REGISTRY
        ]
    }


app.include_router(models_router)
app.include_router(video_router)
