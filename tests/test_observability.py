import io
import json
import logging

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.cache import model_cache
from src.serving.dependency import get_artifact_storage

# Importing metrics to check their in-memory values
from src.serving.observability import CACHE_HITS, CACHE_MISSES, MODEL_LOAD_TOTAL, PREDICTION_ERRORS, PREDICTION_REQUESTS

client = TestClient(app)


# Fixtures
@pytest.fixture
def trained_model_bytes():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    # Dataset with 2 features
    X = np.array([[0.1, 0.2], [0.5, 0.6]])
    y = np.array([0, 1])
    model.fit(X, y)

    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def setup_env(trained_model_bytes):
    """Clear the cache and register a test model for observability"""
    model_cache.clear()
    model_name = "obs-model"
    version = "1.0.0"
    uri = f"local://{model_name}/{version}/model.joblib"

    client.post("/models", json={"name": model_name, "description": "Observability Test"})
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

    storage = get_artifact_storage()
    storage.save(uri, trained_model_bytes)

    return {"model": model_name, "version": version}


# Helper function to read Prometheus metric values in tests
def get_counter_value(metric_obj, **labels):
    try:
        return metric_obj.labels(**labels)._value.get()
    except KeyError:
        return 0.0


# 1. Metrics Tests (Metrics and Loading tests)
def test_prediction_metrics_success(setup_env):
    """Test 1, 3, 4, 5: prediction success increments counter and records latency & load metrics"""
    initial_reqs = get_counter_value(
        PREDICTION_REQUESTS, model=setup_env["model"], version=setup_env["version"], status="success"
    )
    initial_loads = get_counter_value(
        MODEL_LOAD_TOTAL, model=setup_env["model"], version=setup_env["version"], status="success"
    )

    client.post(f"/models/{setup_env['model']}/versions/{setup_env['version']}/predict", json={"features": [1.0, 2.0]})

    # Test 1: Increment success counter
    assert (
        get_counter_value(PREDICTION_REQUESTS, model=setup_env["model"], version=setup_env["version"], status="success")
        == initial_reqs + 1
    )
    # Test 4 and 5: Increment loading counter (since the cache was empty)
    assert (
        get_counter_value(MODEL_LOAD_TOTAL, model=setup_env["model"], version=setup_env["version"], status="success")
        == initial_loads + 1
    )

    # Test 3 (Latency) is evaluated by checking /metrics (at the end of the file)


def test_prediction_metrics_failure(setup_env):
    """Test 2: prediction failure increments error counter"""
    initial_errs = get_counter_value(
        PREDICTION_ERRORS, model=setup_env["model"], version=setup_env["version"], error_type="invalid_features"
    )

    # Send 3 features instead of 2 features (generates invalid_features error)
    client.post(
        f"/models/{setup_env['model']}/versions/{setup_env['version']}/predict", json={"features": [1.0, 2.0, 3.0]}
    )

    assert (
        get_counter_value(
            PREDICTION_ERRORS, model=setup_env["model"], version=setup_env["version"], error_type="invalid_features"
        )
        == initial_errs + 1
    )


# ==========================================
# 2. Cache Metrics Tests
# ==========================================
def test_cache_metrics_flow(setup_env):
    """Test 6, 7, 8: cache hit/miss counters and load avoidance"""
    init_miss = get_counter_value(CACHE_MISSES, model=setup_env["model"], version=setup_env["version"])
    init_hit = get_counter_value(CACHE_HITS, model=setup_env["model"], version=setup_env["version"])
    init_loads = get_counter_value(
        MODEL_LOAD_TOTAL, model=setup_env["model"], version=setup_env["version"], status="success"
    )

    url = f"/models/{setup_env['model']}/versions/{setup_env['version']}/predict"
    payload = {"features": [1.0, 2.0]}

    # First request: Generate Cache Miss
    client.post(url, json=payload)
    assert get_counter_value(CACHE_MISSES, model=setup_env["model"], version=setup_env["version"]) == init_miss + 1
    # Note: We expect the load to increase as well, but it might have changed in previous tests,
    # so we check that it has increased by exactly 1 unit compared to the initial value
    assert (
        get_counter_value(MODEL_LOAD_TOTAL, model=setup_env["model"], version=setup_env["version"], status="success")
        == init_loads + 1
    )

    # Second request: Generate Cache Hit and prevent reloading
    client.post(url, json=payload)
    assert get_counter_value(CACHE_HITS, model=setup_env["model"], version=setup_env["version"]) == init_hit + 1
    # The load should not increase, so it must remain equal to the previous value
    assert (
        get_counter_value(MODEL_LOAD_TOTAL, model=setup_env["model"], version=setup_env["version"], status="success")
        == init_loads + 1
    )


# 3. Logging Tests (Structured Logger Tests)
def test_structured_logging_success(setup_env, caplog):
    """Test 9: successful prediction creates structured log without raw features"""
    caplog.set_level(logging.INFO, logger="msp_observability")

    client.post(f"/models/{setup_env['model']}/versions/{setup_env['version']}/predict", json={"features": [5.5, 6.6]})

    # Check if a log was generated
    assert len(caplog.records) > 0
    log_record = caplog.records[-1]

    # Parse the JSON log
    log_data = json.loads(log_record.message)

    assert log_data["event"] == "prediction"
    assert log_data["status"] == "success"
    assert "latency_ms" in log_data
    assert "request_id" in log_data
    assert "features" not in log_data  # Features should not be logged (privacy preservation)


def test_structured_logging_error(setup_env, caplog):
    """Test 10: failed prediction creates structured log"""
    caplog.set_level(logging.INFO, logger="msp_observability")

    client.post(f"/models/{setup_env['model']}/versions/999.0/predict", json={"features": [1.0, 2.0]})

    log_record = caplog.records[-1]
    log_data = json.loads(log_record.message)

    assert log_data["event"] == "prediction"
    assert log_data["status"] == "error"
    assert log_data["error_type"] == "version_not_found"
    assert "request_id" in log_data


# ==========================================
# 4. Request ID Tests
# ==========================================
def test_request_id_generated(setup_env):
    """Test 11: request ID generated when missing"""
    response = client.post(
        f"/models/{setup_env['model']}/versions/{setup_env['version']}/predict", json={"features": [1.0, 2.0]}
    )
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_request_id_preserved(setup_env):
    """Test 12: existing request ID preserved"""
    custom_id = "my-custom-tracing-id-123"
    response = client.post(
        f"/models/{setup_env['model']}/versions/{setup_env['version']}/predict",
        json={"features": [1.0, 2.0]},
        headers={"X-Request-ID": custom_id},
    )
    assert response.headers["X-Request-ID"] == custom_id


# ==========================================
# 5. Prometheus /metrics Endpoint Tests
# ==========================================
def test_metrics_endpoint_returns_200():
    """Test 13: /metrics returns 200"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_metrics_endpoint_exposes_data(setup_env):
    """Test 14: /metrics exposes prediction metrics"""
    # First, send a successful request to generate data
    client.post(f"/models/{setup_env['model']}/versions/{setup_env['version']}/predict", json={"features": [1.0, 2.0]})

    response = client.get("/metrics")
    metrics_text = response.text

    # Check for the existence of main metrics and labels with Prometheus values
    assert "prediction_requests_total" in metrics_text
    assert "prediction_latency_seconds_bucket" in metrics_text
    assert "model_load_total" in metrics_text
    assert "model_cache_misses_total" in metrics_text
    assert f'model="{setup_env["model"]}"' in metrics_text
