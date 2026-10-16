"""Train a fraud-detection model and save it as a joblib artifact.

This is intentionally a *boring* baseline (LogisticRegression). The point of
the platform is reliable infrastructure around a model, not an exotic model.
The same pipeline can later be swapped for XGBoost behind the identical
artifact contract.

Usage
-----
    python -m src.models.train
    python -m src.models.train --n-samples 20000 --out models/fraud-model.joblib

The script:
  1. Generates a synthetic but realistic fraud dataset (imbalanced).
  2. Splits, scales, and trains a LogisticRegression inside a Pipeline.
  3. Evaluates on a held-out set (ROC-AUC, precision/recall, confusion matrix).
  4. Persists the fitted Pipeline to disk with joblib.

The saved object is a scikit-learn Pipeline, so it exposes `.predict()` and
`.n_features_in_` exactly like the serving layer's loader expects.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The six features the serving contract documents, in a fixed order.
FEATURE_NAMES = [
    "amount",
    "transaction_hour",
    "customer_age",
    "transactions_last_24h",
    "avg_amount_last_30d",
    "distance_from_home_km",
]
RANDOM_STATE = 42


def generate_dataset(n_samples: int, fraud_rate: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic, imbalanced fraud dataset.

    Fraudulent transactions are drawn from shifted distributions: larger
    amounts, later hours, more recent transactions, and greater distance from
    home. This gives the model a real (but learnable) signal.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # --- Legitimate transactions ---
    legit = np.column_stack(
        [
            rng.gamma(shape=2.0, scale=40.0, size=n_legit),  # amount (~$80)
            rng.normal(14, 4, n_legit).clip(0, 23),  # transaction_hour (daytime)
            rng.normal(42, 14, n_legit).clip(18, 90),  # customer_age
            rng.poisson(2, n_legit),  # transactions_last_24h
            rng.gamma(shape=2.0, scale=45.0, size=n_legit),  # avg_amount_last_30d
            rng.exponential(15, n_legit).clip(0, 500),  # distance_from_home_km
        ]
    )

    # --- Fraudulent transactions (shifted distributions) ---
    fraud = np.column_stack(
        [
            rng.gamma(shape=3.0, scale=120.0, size=n_fraud),  # larger amounts
            rng.normal(2, 3, n_fraud).clip(0, 23),  # odd hours (late night)
            rng.normal(38, 16, n_fraud).clip(18, 90),  # customer_age
            rng.poisson(6, n_fraud),  # bursty activity
            rng.gamma(shape=2.0, scale=50.0, size=n_fraud),  # avg_amount_last_30d
            rng.exponential(120, n_fraud).clip(0, 3000),  # far from home
        ]
    )

    x = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # Shuffle so classes aren't ordered.
    idx = rng.permutation(len(y))
    return x[idx], y[idx]


def train(n_samples: int) -> tuple[Pipeline, dict]:
    """Train the pipeline and return it alongside evaluation metrics."""
    x, y = generate_dataset(n_samples)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Scaling matters for LogisticRegression; bundling it in the Pipeline means
    # the serving layer receives raw features and never has to know about it.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # handle the 3% fraud imbalance
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, target_names=["legit", "fraud"]),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    return pipeline, metrics


def save_model(pipeline: Pipeline, out_path: Path) -> int:
    """Persist the pipeline to disk; return the artifact size in bytes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    data = buf.getvalue()
    out_path.write_bytes(data)
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud-detection model.")
    parser.add_argument("--n-samples", type=int, default=20_000, help="dataset size")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/fraud-model.joblib"),
        help="output path for the joblib artifact",
    )
    args = parser.parse_args()

    print(f"Generating {args.n_samples:,} samples and training...")
    pipeline, metrics = train(args.n_samples)

    print("\n=== Evaluation (held-out test set) ===")
    print(f"ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"Train / test size: {metrics['n_train']:,} / {metrics['n_test']:,}")
    print(f"Confusion matrix:  {metrics['confusion_matrix']}")
    print("\n" + metrics["report"])

    size = save_model(pipeline, args.out)
    print(f"Saved model to {args.out} ({size:,} bytes)")
    print(f"Model expects {pipeline.n_features_in_} features: {FEATURE_NAMES}")


if __name__ == "__main__":
    main()
