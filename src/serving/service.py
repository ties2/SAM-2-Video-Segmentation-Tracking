import time

from fastapi import HTTPException
from sqlalchemy.orm import Session
from starlette import status

from src.serving import schemas
from src.serving.cache import model_cache
from src.serving.dependency import get_artifact_storage
from src.serving.loader import get_loader
from src.serving.observability import Observability
from src.serving.repository import ModelRepository, ModelVersionRepository
from src.serving.storage import ArtifactNotFoundError


# Custom Exceptions
class ArchivedModelError(Exception):
    pass


class InvalidFeatures(Exception):
    pass


class PredictionError(Exception):
    pass


class TaskMismatchError(Exception):
    """The requested operation does not match what the model can do (e.g. /predict on SAM 2)."""


# Services
class ModelService:
    @staticmethod
    def create_model(db: Session, model_data: schemas.ModelCreate):
        existing_model = ModelRepository.get_model_by_name(db, model_data.name)
        if existing_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model with name '{model_data.name}' already exists."
            )
        return ModelRepository.create_model(db, model_data)

    @staticmethod
    def get_all_models(db: Session):
        return ModelRepository.get_all_models(db)

    @staticmethod
    def get_model_by_name(db: Session, model_name: str):
        model = ModelRepository.get_model_by_name(db, model_name)
        if not model:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_name}' not found.")
        return model


class ModelVersionService:
    @staticmethod
    def create_version(db: Session, model_name: str, version_data: schemas.ModelVersionCreate):
        model = ModelService.get_model_by_name(db, model_name)

        existing_version = ModelVersionRepository.get_version_by_model_and_name(db, model.id, version_data.version)
        if existing_version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Version '{version_data.version}' already exists for model '{model_name}'.",
            )

        return ModelVersionRepository.create_model_version(db, model.id, version_data)

    @staticmethod
    def get_versions(db: Session, model_name: str):
        model = ModelService.get_model_by_name(db, model_name)
        return ModelVersionRepository.get_versions_by_model(db, model.id)

    @staticmethod
    def load_model_version(db: Session, model_name: str, version: str) -> dict:
        # 1. Check the cache
        if model_cache.contains(model_name, version):
            Observability.record_cache_hit(model_name, version)
            return {"model_name": model_name, "version": version, "status": "already_loaded"}

        Observability.record_cache_miss(model_name, version)

        # 2. Check if the model exists
        model = ModelService.get_model_by_name(db, model_name)
        model_version = ModelVersionRepository.get_version_by_model_and_name(db, model.id, version)

        if not model_version:
            raise HTTPException(status_code=404, detail="Version not found")

        if model_version.status == "archived":
            raise ArchivedModelError("Cannot load archived model version")

        # Pick the loader from the registry (raises UnsupportedModelFormat for unknown pairs)
        loader = get_loader(model_version.framework, model_version.model_format)

        # 3. Load the model and record load time (perf_counter: monotonic, high resolution)
        storage = get_artifact_storage()
        try:
            start_time = time.perf_counter()
            artifact_bytes = storage.get(model_version.artifact_uri)
            loaded_model = loader.load(artifact_bytes)
            load_latency = time.perf_counter() - start_time

            Observability.record_model_load(model_name, version, "success", load_latency)
        except ArtifactNotFoundError:
            Observability.record_model_load(model_name, version, "error")
            raise ArtifactNotFoundError(f"Artifact missing for URI: {model_version.artifact_uri}") from None
        except Exception as e:
            Observability.record_model_load(model_name, version, "error")
            raise e

        # 4. Store in cache
        model_cache.set(model_name, version, loaded_model)
        return {"model_name": model_name, "version": version, "status": "loaded"}

    @staticmethod
    def get_model_status(model_name: str, version: str) -> dict:
        is_loaded = model_cache.contains(model_name, version)
        return {"model_name": model_name, "version": version, "loaded": is_loaded}


# New: Lifecycle Service (MSP-008)
class LifecycleService:
    @staticmethod
    def promote_version(db: Session, model_name: str, version: str):
        # 1. Get the model (the method itself raises a 404 if it doesn't exist)
        model = ModelService.get_model_by_name(db, model_name)

        # 2. Get the version and perform validation
        target_version = ModelVersionRepository.get_version_by_model_and_name(db, model.id, version)
        if not target_version:
            Observability.record_promotion(model_name, version, "failed", "version_not_found")
            raise HTTPException(status_code=404, detail="Version not found")

        if target_version.status == "archived":
            Observability.record_promotion(model_name, version, "failed", "cannot_promote_archived")
            raise HTTPException(status_code=400, detail="Cannot promote an archived version")

        # 3. Check artifact integrity and loadability
        storage = get_artifact_storage()
        try:
            artifact_bytes = storage.get(target_version.artifact_uri)
            loader = get_loader(target_version.framework, target_version.model_format)
            _ = loader.load(artifact_bytes)  # Just testing if it loads
        except ArtifactNotFoundError:
            Observability.record_promotion(model_name, version, "failed", "artifact_missing")
            raise HTTPException(status_code=400, detail="Artifact does not exist") from None
        except Exception as e:
            Observability.record_promotion(model_name, version, "failed", "model_corrupted")
            raise HTTPException(status_code=400, detail=f"Model artifact is corrupted: {e!s}") from e

        # 4. Perform promotion via safe database transaction
        try:
            new_prod, old_prod = ModelVersionRepository.promote_transactionally(db, model.id, target_version)

            # 5. Invalidate the old version cache
            if old_prod:
                model_cache.invalidate(model_name, old_prod.version)

            Observability.record_promotion(model_name, version, "success")
            return new_prod
        except Exception:
            Observability.record_promotion(model_name, version, "failed", "transaction_failed")
            raise HTTPException(status_code=500, detail="Promotion transaction failed") from None

    @staticmethod
    def get_production(db: Session, model_name: str):
        model = ModelService.get_model_by_name(db, model_name)
        prod_version = ModelVersionRepository.get_production_version(db, model.id)
        if not prod_version:
            raise HTTPException(status_code=404, detail="No production version found")
        return prod_version


class PredictionService:
    @staticmethod
    def predict(db: Session, model_name: str, version: str, features: list[float]) -> dict:
        if model_cache.contains(model_name, version):
            Observability.record_cache_hit(model_name, version)
        else:
            ModelVersionService.load_model_version(db, model_name, version)

        model = model_cache.get(model_name, version)
        if not model:
            raise PredictionError("Model is not available in memory")

        if not hasattr(model, "predict"):
            raise TaskMismatchError(
                f"Model '{model_name}' version '{version}' does not support tabular prediction; "
                "use the /segment-video endpoint for video segmentation models."
            )

        if hasattr(model, "n_features_in_") and len(features) != model.n_features_in_:
            raise InvalidFeatures(f"Expected {model.n_features_in_} features, got {len(features)}.")

        try:
            start_pred_time = time.perf_counter()
            prediction_result = model.predict([features])
            prediction_latency = time.perf_counter() - start_pred_time

            val = prediction_result[0].item() if hasattr(prediction_result[0], "item") else prediction_result[0]

            Observability.record_prediction_success(model_name, version, prediction_latency)
            return {"model_name": model_name, "version": version, "prediction": val}

        except Exception as e:
            raise PredictionError(f"Prediction failed: {e!s}") from e

    # Predict Production (MSP-008)
    @staticmethod
    def predict_production(db: Session, model_name: str, features: list[float]) -> dict:
        """Automatically route prediction to the Production version of the model"""
        model = ModelService.get_model_by_name(db, model_name)
        prod_version = ModelVersionRepository.get_production_version(db, model.id)
        if not prod_version:
            raise HTTPException(status_code=404, detail="No production version found")

        return PredictionService.predict(db, model_name, prod_version.version, features)
