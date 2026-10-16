from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    # 1. Check if the HTTP request was successful
    assert response.status_code == 200

    # 2. Parse the JSON response
    data = response.json()

    # 3. Assert that the status is 'ok' (ignoring other fields)
    assert data["status"] == "ok"

    # Optional: You can also assert that important ML-specific keys exist
    # assert "static_model_loaded" in data
    # assert "pipeline_mode" in data
