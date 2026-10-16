"""Seed the running platform with a real trained model, ready for load testing.

End-to-end flow (mirrors how a real deployment would register a model):
  1. Train the fraud model (in-memory).
  2. Save the artifact into the platform's storage backend.
  3. Register the model + version via the API.
  4. Promote the version to production.

Run it against a running stack (from the host, or `docker compose exec api`):
    python -m src.models.seed
"""

from __future__ import annotations

import io
import os

import joblib
import requests

from src.models.train import train
from src.serving.dependency import get_artifact_storage

BASE_URL = os.environ.get("MSP_BASE_URL", "http://localhost:8000")
MODEL_NAME = "fraud-model"
VERSION = "1.0.0"
ARTIFACT_URI = f"local://{MODEL_NAME}/{VERSION}/model.joblib"


def main() -> None:
    print("1. Training model...")
    pipeline, metrics = train(n_samples=20_000)
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")

    print("2. Saving artifact to storage...")
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    get_artifact_storage().save(ARTIFACT_URI, buf.getvalue())

    print("3. Registering model + version...")
    requests.post(f"{BASE_URL}/models", json={"name": MODEL_NAME, "description": "Fraud risk model"})
    requests.post(
        f"{BASE_URL}/models/{MODEL_NAME}/versions",
        json={
            "version": VERSION,
            "framework": "scikit-learn",
            "model_format": "joblib",
            "artifact_uri": ARTIFACT_URI,
            "status": "staging",
        },
    )

    print("4. Promoting to production...")
    res = requests.post(f"{BASE_URL}/models/{MODEL_NAME}/versions/{VERSION}/promote")
    print(f"   -> {res.status_code}: {res.json()}")

    print("\nSeed complete. You can now run:  locust -f locustfile.py --host", BASE_URL)


if __name__ == "__main__":
    main()
