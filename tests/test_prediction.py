import io
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.serving.app import app
from src.serving.cache import model_cache
from src.serving.dependency import get_artifact_storage

client = TestClient(app)


# ==========================================
# Fixtures (Initial setup)
# ==========================================
@pytest.fixture
def trained_model_bytes():
    """Create a real trained model with 4 features"""
    model = LogisticRegression()
    # Dummy dataset with 4 features
    X = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    y = np.array([0, 1])
    model.fit(X, y)  # Train the model to generate the n_features_in_ attribute

    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def setup_environment(trained_model_bytes):
    """Clear the Cache and register a valid model before running each test"""
    model_cache.clear()
    model_name = "fraud-model"
    version = "1.0.0"
    uri = f"local://{model_name}/{version}/model.joblib"

    # Register the model and version in the database
    client.post("/models", json={"name": model_name, "description": "Fraud Detection Model"})
    client.post(
        f"/models/{model_name}/versions",
        json={
            "version": version,
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": uri,
            "status": "production",
        },
    )

    # Save the file in storage
    storage = get_artifact_storage()
    storage.save(uri, trained_model_bytes)

    return {"model": model_name, "version": version, "uri": uri}


# ==========================================
# 1. Successful Prediction
# ==========================================
def test_valid_prediction(setup_environment):
    """Test 1: valid prediction"""
    response = client.post(
        f"/models/{setup_environment['model']}/versions/{setup_environment['version']}/predict",
        json={"features": [0.1, 0.5, 1.2, 3.4]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == setup_environment["model"]
    assert "prediction" in data


def test_another_valid_prediction(setup_environment):
    """Test 2: prediction with another valid input"""
    response = client.post(
        f"/models/{setup_environment['model']}/versions/{setup_environment['version']}/predict",
        json={"features": [9.9, 8.8, 7.7, 6.6]},
    )
    assert response.status_code == 200
    assert "prediction" in response.json()


# ==========================================
# 2. Validation Tests
# ==========================================
def test_empty_features(setup_environment):
    """Test 3: empty features (422)"""
    response = client.post(
        f"/models/{setup_environment['model']}/versions/{setup_environment['version']}/predict", json={"features": []}
    )
    assert response.status_code == 422


def test_non_numeric_features(setup_environment):
    """Test 4: non-numeric feature (422)"""
    response = client.post(
        f"/models/{setup_environment['model']}/versions/{setup_environment['version']}/predict",
        json={"features": [1.2, "invalid_string", 3.4, 5.6]},
    )
    assert response.status_code == 422


def test_wrong_feature_count(setup_environment):
    """Test 5: wrong feature count (422)"""
    # Send 3 features instead of the expected 4 features
    response = client.post(
        f"/models/{setup_environment['model']}/versions/{setup_environment['version']}/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )
    assert response.status_code == 422
    assert "Expected" in response.text and "features" in response.text


# ==========================================
# 3. Registry & Status Tests
# ==========================================
def test_unknown_model():
    """Test 6: unknown model (404)"""
    response = client.post("/models/unknown-model/versions/1.0.0/predict", json={"features": [1, 2, 3, 4]})
    assert response.status_code == 404


def test_unknown_version(setup_environment):
    """Test 7: unknown version (404)"""
    response = client.post(
        f"/models/{setup_environment['model']}/versions/99.9.9/predict", json={"features": [1, 2, 3, 4]}
    )
    assert response.status_code == 404


def test_archived_model(setup_environment):
    """Test 8: archived model cannot be used for prediction (409)"""
    archived_version = "2.0.0"
    client.post(
        f"/models/{setup_environment['model']}/versions",
        json={
            "version": archived_version,
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": f"local://{setup_environment['model']}/{archived_version}/model.joblib",
            "status": "archived",
        },
    )

    response = client.post(
        f"/models/{setup_environment['model']}/versions/{archived_version}/predict", json={"features": [1, 2, 3, 4]}
    )
    assert response.status_code == 409


# ==========================================
# 4. Loading & Caching Tests
# ==========================================
@patch("src.serving.loader.SklearnJoblibLoader.load")  # loaders now live in the registry
def test_prediction_cache_behavior(mock_load, setup_environment, trained_model_bytes):
    """Test 9 & 10: Auto-loading and Caching verification"""
    # Setup Mock behavior to return a real model
    import io

    import joblib

    real_model = joblib.load(io.BytesIO(trained_model_bytes))
    mock_load.return_value = real_model

    url = f"/models/{setup_environment['model']}/versions/{setup_environment['version']}/predict"
    payload = {"features": [1.0, 2.0, 3.0, 4.0]}

    # First request: it should load the model
    response1 = client.post(url, json=payload)
    assert response1.status_code == 200
    assert mock_load.call_count == 1  # The load method was called exactly once

    # Second request: it should use the cache
    response2 = client.post(url, json=payload)
    assert response2.status_code == 200
    assert mock_load.call_count == 1  # The load method was not called again!


# ==========================================
# 5. Error Tests
# ==========================================
def test_corrupted_model(setup_environment):
    """Test 11: corrupted model artifact (500)"""
    corrupt_version = "3.0.0"
    uri = f"local://{setup_environment['model']}/{corrupt_version}/model.joblib"

    client.post(
        f"/models/{setup_environment['model']}/versions",
        json={
            "version": corrupt_version,
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": uri,
            "status": "staging",
        },
    )

    storage = get_artifact_storage()
    storage.save(uri, b"corrupted random bytes string")

    response = client.post(
        f"/models/{setup_environment['model']}/versions/{corrupt_version}/predict", json={"features": [1, 2, 3, 4]}
    )
    assert response.status_code == 500


@patch("src.serving.service.model_cache.get")
def test_prediction_failure(mock_cache_get, setup_environment):
    """Test 12: prediction internal failure (500)"""
    # Create a fake model that raises an error during prediction
    mock_model = MagicMock()
    mock_model.n_features_in_ = 4
    mock_model.predict.side_effect = Exception("Simulated internal ML error")
    mock_cache_get.return_value = mock_model

    # To skip Auto-load and use the Mock model directly
    with patch("src.serving.service.model_cache.contains", return_value=True):
        response = client.post(
            f"/models/{setup_environment['model']}/versions/{setup_environment['version']}/predict",
            json={"features": [1, 2, 3, 4]},
        )

    assert response.status_code == 500
    assert "Simulated internal ML error" in response.text
