from src.serving.config import settings
from src.serving.storage import ArtifactStorage, LocalArtifactStorage


def get_artifact_storage() -> ArtifactStorage:
    if settings.ARTIFACT_STORAGE_BACKEND == "local":  # or change config to use "s3"
        return LocalArtifactStorage(base_dir=settings.LOCAL_ARTIFACT_DIR)
    raise ValueError(f"Unsupported storage backend: {settings.ARTIFACT_STORAGE_BACKEND}")
