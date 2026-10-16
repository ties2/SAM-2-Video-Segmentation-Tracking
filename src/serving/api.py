from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.serving import schemas
from src.serving.database import get_db
from src.serving.loader import (
    InvalidModelArtifact,
    ModelLoadError,
    UnsupportedModelFormat,
)
from src.serving.observability import Observability
from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.service import (
    ArchivedModelError,
    InvalidFeatures,
    LifecycleService,
    ModelService,
    ModelVersionService,
    PredictionError,
    PredictionService,
    TaskMismatchError,
)
from src.serving.storage import ArtifactNotFoundError

router = APIRouter(prefix="/models", tags=["Model Registry"])


# Endpoints for Models
@router.post("", response_model=schemas.ModelResponse)
def create_model(model_data: schemas.ModelCreate, db: Session = Depends(get_db)):
    return ModelService.create_model(db, model_data)


@router.get("", response_model=list[schemas.ModelResponse])
def get_all_models(db: Session = Depends(get_db)):
    return ModelService.get_all_models(db)


@router.get("/{model_name}", response_model=schemas.ModelResponse)
def get_model(model_name: str, db: Session = Depends(get_db)):
    return ModelService.get_model_by_name(db, model_name)


# Endpoints for Model Versions
@router.post("/{model_name}/versions", response_model=schemas.ModelVersionResponse)
def create_model_version(model_name: str, version_data: schemas.ModelVersionCreate, db: Session = Depends(get_db)):
    return ModelVersionService.create_version(db, model_name, version_data)


@router.get("/{model_name}/versions", response_model=list[schemas.ModelVersionResponse])
def get_model_versions(model_name: str, db: Session = Depends(get_db)):
    return ModelVersionService.get_versions(db, model_name)


#  Endpoints for Model Loader
@router.post("/{model_name}/versions/{version}/load")
def load_model(model_name: str, version: str, db: Session = Depends(get_db)):
    """
    Endpoint to load a model from Storage into Memory
    """
    try:
        result = ModelVersionService.load_model_version(db, model_name, version)
        return result

    except ArchivedModelError as e:
        # 409 Conflict status code for when operation conflicts with current state of archived model
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e

    except ArtifactNotFoundError as e:
        # 404 Not Found status code
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    except (UnsupportedModelFormat, InvalidModelArtifact) as e:
        # 400 Bad Request status code for when the file format is incorrect or the file is invalid
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    except ModelLoadError as e:
        # 500 Internal Server Error status code for unexpected errors during model loading
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e

    # Errors related to the model or version not being found are already handled in the service with a 404


@router.get("/{model_name}/versions/{version}/status")
def get_model_status(model_name: str, version: str):
    """
    Endpoint to view the current status of the model in the Cache
    """
    # This method doesn't need the database because it reads directly from RAM (Cache)
    result = ModelVersionService.get_model_status(model_name, version)
    return result


@router.post("/{model_name}/versions/{version}/predict", response_model=PredictionResponse)
def predict_model(model_name: str, version: str, request: PredictionRequest, db: Session = Depends(get_db)):
    try:
        return PredictionService.predict(db, model_name, version, request.features)

    except ArchivedModelError as e:
        Observability.record_prediction_error(model_name, version, "archived_model")
        raise HTTPException(status_code=409, detail=str(e)) from e

    except ArtifactNotFoundError as e:
        Observability.record_prediction_error(model_name, version, "artifact_not_found")
        raise HTTPException(status_code=404, detail=str(e)) from e

    except InvalidFeatures as e:
        Observability.record_prediction_error(model_name, version, "invalid_features")
        raise HTTPException(status_code=422, detail=str(e)) from e

    except TaskMismatchError as e:
        Observability.record_prediction_error(model_name, version, "task_mismatch")
        raise HTTPException(status_code=400, detail=str(e)) from e

    except (UnsupportedModelFormat, InvalidModelArtifact) as e:
        Observability.record_prediction_error(model_name, version, "invalid_model_artifact")
        raise HTTPException(status_code=400, detail=str(e)) from e

    except HTTPException as e:
        if e.status_code == 404:
            err_type = "model_not_found" if "Model" in str(e.detail) else "version_not_found"
            Observability.record_prediction_error(model_name, version, err_type)
        raise e

    except (PredictionError, ModelLoadError) as e:
        Observability.record_prediction_error(model_name, version, "prediction_error")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Lifecycle & Promotion Endpoints (MSP-008)
@router.post("/{model_name}/versions/{version}/promote")
def promote_model_version(model_name: str, version: str, db: Session = Depends(get_db)):
    """
    Promote a specific version of a model to the Production environment.
    The previous Production version is automatically archived.
    """
    updated_version = LifecycleService.promote_version(db, model_name, version)
    return {"model_name": model_name, "version": updated_version.version, "status": updated_version.status}


@router.get("/{model_name}/production")
def get_production_version(model_name: str, db: Session = Depends(get_db)):
    """
    Get the version of the model that is currently in the Production status.
    """
    prod_version = LifecycleService.get_production(db, model_name)
    return {"model_name": model_name, "version": prod_version.version, "status": prod_version.status}


@router.post("/{model_name}/predict")
def predict_production(model_name: str, payload: dict, db: Session = Depends(get_db)):
    """
    Perform prediction on the Production version without needing to know the version number.
    """
    features = payload.get("features")
    if not features or not isinstance(features, list):
        raise HTTPException(status_code=422, detail="Invalid request body. 'features' list is required.")

    try:
        return PredictionService.predict_production(db, model_name, features)
    except InvalidFeatures as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except TaskMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
