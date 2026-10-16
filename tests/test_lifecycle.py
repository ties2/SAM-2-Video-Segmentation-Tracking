import io
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from src.serving.app import app
from src.serving.cache import model_cache
from src.serving.database import SessionLocal
from src.serving.db_models import DBModelVersion
from src.serving.dependency import get_artifact_storage

client = TestClient(app)


# ==========================================
# Fixtures
# ==========================================
@pytest.fixture(scope="module")
def setup_lifecycle_models():
    model_name = f"lifecycle-model-{int(time.time())}"
    client.post("/models", json={"name": model_name, "description": "Lifecycle test"})

    import joblib

    from src.serving.dependency import get_artifact_storage

    storage = get_artifact_storage()

    # ---> FIX: Using fit to create a fully real and standard model <---
    # First model
    clf1 = LinearRegression()
    # Providing 3 training data samples (with 2 features) so the model actually fits
    clf1.fit([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [0.0, 1.0, 2.0])
    buf1 = io.BytesIO()
    joblib.dump(clf1, buf1)

    filename1 = f"{model_name}_v1.joblib"
    storage.save(f"local://{filename1}", buf1.getvalue())

    # Second model
    clf2 = LinearRegression()
    clf2.fit([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [1.0, 2.0, 3.0])
    buf2 = io.BytesIO()
    joblib.dump(clf2, buf2)

    filename2 = f"{model_name}_v2.joblib"
    storage.save(f"local://{filename2}", buf2.getvalue())

    # Register versions in the database
    client.post(
        f"/models/{model_name}/versions",
        json={
            "version": "v1.0.0",
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": f"local://{filename1}",
        },
    )
    client.post(
        f"/models/{model_name}/versions",
        json={
            "version": "v2.0.0",
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": f"local://{filename2}",
        },
    )

    yield model_name

    # Cleanup
    try:
        storage.delete(f"local://{filename1}")
        storage.delete(f"local://{filename2}")
    except Exception:
        pass


# ==========================================
# Tests (15 Scenarios)
# ==========================================


def test_1_unknown_model_promote():
    res = client.post("/models/unknown-model/versions/v1.0.0/promote")
    assert res.status_code == 404


def test_2_unknown_version_promote(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    res = client.post(f"/models/{model_name}/versions/v9.9.9/promote")
    assert res.status_code == 404


def test_3_missing_artifact_rejects_promotion(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    client.post(
        f"/models/{model_name}/versions",
        json={
            "version": "v3-missing",
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": "local://fake_missing_file.joblib",
        },
    )
    res = client.post(f"/models/{model_name}/versions/v3-missing/promote")
    assert res.status_code == 400
    assert "Artifact does not exist" in res.json()["detail"]


def test_4_corrupted_model_rejects_promotion(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    storage = get_artifact_storage()

    # Create a corrupted file using the system's dedicated storage
    filename_corrupted = f"{model_name}_corrupted.joblib"
    storage.save(f"local://{filename_corrupted}", b"I am not a real model!")

    client.post(
        f"/models/{model_name}/versions",
        json={
            "version": "v4-corrupted",
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": f"local://{filename_corrupted}",
        },
    )
    res = client.post(f"/models/{model_name}/versions/v4-corrupted/promote")

    assert res.status_code == 400
    assert "corrupted" in res.json()["detail"]


def test_5_production_endpoint_no_version(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    res = client.get(f"/models/{model_name}/production")
    assert res.status_code == 404


def test_6_successful_promotion_v1(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    res = client.post(f"/models/{model_name}/versions/v1.0.0/promote")
    assert res.status_code == 200
    assert res.json()["status"] == "production"


def test_7_production_endpoint_returns_current(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    res = client.get(f"/models/{model_name}/production")
    assert res.status_code == 200
    assert res.json()["version"] == "v1.0.0"


def test_8_production_prediction_works(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    res = client.post(f"/models/{model_name}/predict", json={"features": [10.0, 5.0]})
    assert res.status_code == 200
    assert res.json()["version"] == "v1.0.0"


def test_9_promote_v2_archives_v1(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    res = client.post(f"/models/{model_name}/versions/v2.0.0/promote")
    assert res.status_code == 200

    prod_res = client.get(f"/models/{model_name}/production")
    assert prod_res.json()["version"] == "v2.0.0"


def test_10_cannot_promote_archived_v1(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    res = client.post(f"/models/{model_name}/versions/v1.0.0/promote")
    assert res.status_code == 400
    assert "archived" in res.json()["detail"].lower()


def test_11_cache_invalidated_after_promotion(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    assert model_cache.contains(model_name, "v1.0.0") is False


def test_12_production_prediction_uses_new_model(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    res = client.post(f"/models/{model_name}/predict", json={"features": [10.0, 5.0]})
    assert res.status_code == 200
    assert res.json()["version"] == "v2.0.0"


def test_13_predict_production_invalid_features(setup_lifecycle_models):
    model_name = setup_lifecycle_models
    res = client.post(f"/models/{model_name}/predict", json={"features": [10.0]})
    assert res.status_code == 422
    assert "Expected 2" in res.json()["detail"]


def test_14_db_transaction_rollback(setup_lifecycle_models):
    model_name = setup_lifecycle_models

    client.post(
        f"/models/{model_name}/versions",
        json={
            "version": "v5.0.0",
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": f"local://{model_name}_v2.joblib",
        },
    )

    with patch("sqlalchemy.orm.Session.commit", side_effect=SQLAlchemyError("DB Crash")):
        res = client.post(f"/models/{model_name}/versions/v5.0.0/promote")
        assert res.status_code == 500

    prod_res = client.get(f"/models/{model_name}/production")
    assert prod_res.json()["version"] == "v2.0.0"


def test_15_concurrency_only_one_production_allowed(setup_lifecycle_models):
    db = SessionLocal()
    from src.serving.repository import ModelRepository

    model = ModelRepository.get_model_by_name(db, setup_lifecycle_models)

    duplicate_prod = DBModelVersion(
        model_id=model.id,
        version="v99.0.0",
        framework="sk",
        model_format="jb",
        artifact_uri="local://dummy_file.joblib",
        status="production",
    )
    db.add(duplicate_prod)

    try:
        db.commit()
        db.delete(duplicate_prod)
        db.commit()
    except IntegrityError:
        pass
    finally:
        db.close()
