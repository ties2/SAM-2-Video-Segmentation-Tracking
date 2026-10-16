import io

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

# Using the main app for Integration tests
from src.serving.app import app
from src.serving.cache import model_cache
from src.serving.dependency import get_artifact_storage
from src.serving.loader import InvalidModelArtifact, ModelLoadError, SklearnJoblibLoader

client = TestClient(app)


# ==========================================
# Fixtures (Base data for running tests)
# ==========================================
@pytest.fixture
def valid_model_bytes():
    """Create a real scikit-learn model and convert it to bytes"""
    model = LogisticRegression()
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


@pytest.fixture
def invalid_model_bytes():
    """Create a valid joblib file that is not a machine learning model (it is a simple dictionary)"""
    buf = io.BytesIO()
    joblib.dump({"not_a_model": True}, buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clear_cache_before_tests():
    """Clear the Cache before running each test to prevent interference"""
    model_cache.clear()


# ==========================================
# Cache Tests
# ==========================================
def test_cache_set_and_get():
    """Test 1: model is cached after load"""
    model_cache.set("fraud-model", "v1", "dummy_model_object")
    assert model_cache.get("fraud-model", "v1") == "dummy_model_object"


def test_cache_contains_and_remove():
    """Test 2 & 3: remove cached model"""
    model_cache.set("fraud-model", "v2", "dummy")
    assert model_cache.contains("fraud-model", "v2") is True

    model_cache.remove("fraud-model", "v2")
    assert model_cache.contains("fraud-model", "v2") is False


def test_cache_clear():
    """Test 4: clear cache"""
    model_cache.set("model1", "v1", "dummy")
    model_cache.set("model2", "v1", "dummy")
    model_cache.clear()
    assert model_cache.contains("model1", "v1") is False
    assert model_cache.contains("model2", "v1") is False


# ==========================================
# Loader Tests (Tests for the loader and validation)
# ==========================================
def test_loader_valid_model(valid_model_bytes):
    """Test 5: load valid sklearn joblib"""
    loader = SklearnJoblibLoader()
    model = loader.load(valid_model_bytes)
    # Check for the existence of the predict method (standard sklearn interface)
    assert hasattr(model, "predict")


def test_loader_invalid_object(invalid_model_bytes):
    """Test 6: invalid artifact (not a model)"""
    loader = SklearnJoblibLoader()
    with pytest.raises(InvalidModelArtifact):
        loader.load(invalid_model_bytes)


def test_loader_corrupt_bytes():
    """Test 7: corrupt artifact bytes"""
    loader = SklearnJoblibLoader()
    with pytest.raises(ModelLoadError):
        loader.load(b"this is just some random text, not a joblib file")


# ==========================================
# Integration & API Tests
# ==========================================
def test_api_load_success_and_cache(valid_model_bytes):
    """Test 8 & 9: registry -> storage -> loader AND second load uses cache"""
    model_name = "test-load-model"
    version = "1.0.0"
    uri = f"local://{model_name}/{version}/model.joblib"

    # 1. Create the model in the registry
    client.post("/models", json={"name": model_name, "description": "Test"})
    client.post(
        f"/models/{model_name}/versions",
        json={
            "version": version,
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": uri,
            "status": "staging",
        },
    )

    # 2. Physically save the file (simulate model upload)
    storage = get_artifact_storage()
    storage.save(uri, valid_model_bytes)

    # 3. Load the model via API (first time)
    response1 = client.post(f"/models/{model_name}/versions/{version}/load")
    assert response1.status_code == 200
    assert response1.json()["status"] == "loaded"

    # 4. Reload (should be read from Cache)
    response2 = client.post(f"/models/{model_name}/versions/{version}/load")
    assert response2.status_code == 200
    assert response2.json()["status"] == "already_loaded"

    # 5. Check the status Endpoint
    status_resp = client.get(f"/models/{model_name}/versions/{version}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["loaded"] is True


def test_api_load_missing_artifact():
    """Test 10: missing artifact fails correctly (404)"""
    model_name = "missing-model"
    client.post("/models", json={"name": model_name, "description": "Test"})
    client.post(
        f"/models/{model_name}/versions",
        json={
            "version": "1.0.0",
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": "local://does/not/exist.joblib",
            "status": "staging",
        },
    )

    response = client.post(f"/models/{model_name}/versions/1.0.0/load")
    assert response.status_code == 404


def test_api_load_archived_model():
    """Test 11: archived model cannot load (409)"""
    model_name = "archived-model"
    client.post("/models", json={"name": model_name, "description": "Test"})
    client.post(
        f"/models/{model_name}/versions",
        json={
            "version": "1.0.0",
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": "local://dummy.joblib",
            "status": "archived",  # Archived status
        },
    )

    response = client.post(f"/models/{model_name}/versions/1.0.0/load")
    assert response.status_code == 409
