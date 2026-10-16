import uuid

from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)

# Generate a unique name for the model in this test run
test_model_name = f"fraud-model-test-{uuid.uuid4().hex[:6]}"


def test_create_model():
    """Test 1: Successfully create a new model"""
    response = client.post("/models", json={"name": test_model_name, "description": "Test model for pytest"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_model_name
    assert "id" in data


def test_create_duplicate_model():
    """Test 2: Prevent creating a model with a duplicate name"""
    response = client.post("/models", json={"name": test_model_name, "description": "Duplicate model"})
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_get_all_models():
    """Test 3: Retrieve a list of all models"""
    response = client.get("/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0


def test_get_model_by_name():
    """Test 4: Get a specific model's details by its name"""
    response = client.get(f"/models/{test_model_name}")
    assert response.status_code == 200
    assert response.json()["name"] == test_model_name


def test_create_model_version():
    """Test 5: Register a new version for the model"""
    response = client.post(
        f"/models/{test_model_name}/versions",
        json={
            "version": "1.0.0",
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": "models/test/1.0.0/model.joblib",
            "status": "staging",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0.0"
    assert "id" in data


def test_create_duplicate_model_version():
    """Test 6: Prevent registering a duplicate version for a model"""
    response = client.post(
        f"/models/{test_model_name}/versions",
        json={
            "version": "1.0.0",
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": "models/test/1.0.0/model.joblib",
            "status": "staging",
        },
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]
