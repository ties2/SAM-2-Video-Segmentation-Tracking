"""Load test for the ML Model Serving Platform.

Measures real prediction latency and throughput under concurrent load, so the
README can report honest p50/p95/p99 numbers instead of claims.

Prerequisites
-------------
1. The stack is running:            docker compose up -d
2. A model is trained + promoted.   python -m src.models.seed   (see below)

Run (headless, 50 users, 30s, CSV output)
-----------------------------------------
    locust -f locustfile.py --headless \
        -u 50 -r 10 -t 30s \
        --host http://localhost:8000 \
        --csv results/loadtest

Or open the web UI:
    locust -f locustfile.py --host http://localhost:8000
    # then browse http://localhost:8089
"""

import random

from locust import HttpUser, between, task

MODEL_NAME = "fraud-model"
VERSION = "1.0.0"


def random_transaction() -> list[float]:
    """Six features matching the training feature order."""
    return [
        round(random.uniform(5, 500), 2),  # amount
        random.randint(0, 23),  # transaction_hour
        random.randint(18, 90),  # customer_age
        random.randint(0, 12),  # transactions_last_24h
        round(random.uniform(20, 300), 2),  # avg_amount_last_30d
        round(random.uniform(0, 2000), 1),  # distance_from_home_km
    ]


class PredictionUser(HttpUser):
    """Simulates a client hitting the versioned prediction endpoint."""

    # Each simulated user waits 0.1-0.5s between requests.
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict_versioned(self):
        """The hot path: predict against a specific version (auto-loads + caches)."""
        self.client.post(
            f"/models/{MODEL_NAME}/versions/{VERSION}/predict",
            json={"features": random_transaction()},
            name="/predict [versioned]",
        )

    @task(2)
    def predict_production(self):
        """Predict against whatever version is currently in production."""
        self.client.post(
            f"/models/{MODEL_NAME}/predict",
            json={"features": random_transaction()},
            name="/predict [production]",
        )

    @task(1)
    def health(self):
        """Cheap liveness call to establish a latency baseline."""
        self.client.get("/health", name="/health")
