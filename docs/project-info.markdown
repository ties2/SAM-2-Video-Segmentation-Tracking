# ML Model Serving Platform (MSP)

> Production-grade infrastructure for **registering, versioning, and serving machine learning models** — built to demonstrate MLOps / ML-infrastructure / backend engineering, not just another ML classifier.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Status](https://img.shields.io/badge/status-WIP%20(MSP--001%20%26%20MSP--002%20done)-yellow)

---

## Table of Contents

1. [What this project is](#1-what-this-project-is)
2. [The mental model: model vs. platform](#2-the-mental-model-model-vs-platform)
3. [Architecture](#3-architecture)
4. [Tech stack](#4-tech-stack)
5. [Project structure](#5-project-structure)
6. [Data model (Model Registry)](#6-data-model-model-registry)
7. [API reference](#7-api-reference)
8. [Design decisions & trade-offs](#8-design-decisions--trade-offs)
9. [Getting started](#9-getting-started)
10. [Testing](#10-testing)
11. [Current status](#11-current-status)
12. [Known limitations & design notes](#12-known-limitations--design-notes)
13. [Roadmap / future improvements](#13-roadmap--future-improvements)
14. [Development workflow](#14-development-workflow)
15. [Author](#15-author)

---

## 1. What this project is

### The problem

In most ML projects, a trained model file (`model.joblib`) is committed straight into a Git repo and loaded by whatever app happens to need it. That works for a demo, but it breaks down in production:

- You cannot answer *"which model version is live right now?"*
- You cannot roll back a bad model without editing code and rebuilding.
- You have no record of *how* a model was produced (data, params, metrics).
- Two models with different feature contracts silently collide.

### The goal

This platform is the **infrastructure that sits around a model** and answers those questions. The ML model itself is intentionally a small component inside a larger system. The engineering value is everything around it: the registry, the versioning, the serving contract, the deployment story, and the observability.

The first concrete use case is **real-time fraud-risk scoring for a transaction**, but the platform is deliberately decoupled from any single algorithm — a `scikit-learn` model today can be swapped for `XGBoost` tomorrow behind the same API.

> **Design principle:** the serving infrastructure must not be tightly coupled to one specific ML algorithm.

---

## 2. The mental model: model vs. platform

There are two distinct concepts in this repository. Keeping them separate is the whole point.

**A) The ML model** — a pure function:

```
transaction features  ─►  Logistic Regression  ─►  fraud probability
```

**B) The ML Model Serving Platform** — the system this repo actually builds:

```
Client
  │
  ▼
API (FastAPI)
  │
  ▼
Authentication            (planned)
  │
  ▼
Model Registry            ◄── implemented
  │
  ▼
Model Version resolution  ◄── implemented (metadata)
  │
  ▼
Model Loader              (planned)
  │
  ▼
ML Model  ─►  Prediction  (planned)
  │
  ▼
Monitoring                (planned)
```

Your project is **primarily (B)**. That is what makes it an MLOps / ML-infrastructure / distributed-systems project rather than a data-science notebook.

### The intended serving contract

This is the eventual public contract. **It is not implemented yet** — the registry (metadata) is built first, inference comes later (see [roadmap](#13-roadmap--future-improvements)).

Request:

```http
POST /predict
```
```json
{
  "model": "fraud-model",
  "version": "1.0.0",
  "features": {
    "amount": 125.50,
    "transaction_hour": 23,
    "customer_age": 31,
    "transactions_last_24h": 7,
    "avg_amount_last_30d": 85.20,
    "distance_from_home_km": 240.5
  }
}
```

Response:

```json
{
  "model": "fraud-model",
  "version": "1.0.0",
  "prediction": 1,
  "score": 0.91
}
```

A raw **probability/score** is returned rather than a bare `true/false`, so the business threshold (e.g. `score >= 0.80 → fraud`) can be tuned independently of the model. That separation is what later makes **A/B testing, canary deployment, and rollback** meaningful.

---

## 3. Architecture

### Layered application architecture (implemented)

Business logic never lives in a route handler. Every request flows through clean, testable layers:

```
        HTTP request
             │
             ▼
     ┌───────────────┐
     │   API layer   │   src/serving/api.py      (routing, request/response shape)
     └───────┬───────┘
             │
             ▼
     ┌───────────────┐
     │ Service layer │   src/serving/service.py  (business rules, HTTP errors)
     └───────┬───────┘
             │
             ▼
     ┌───────────────┐
     │  Repository   │   src/serving/repository.py (the ONLY layer that talks to the DB)
     └───────┬───────┘
             │
             ▼
     ┌───────────────┐
     │  PostgreSQL   │   via SQLAlchemy ORM
     └───────────────┘
```

- **API** — declares endpoints, validates I/O with Pydantic, injects a DB session. No SQL, no business logic.
- **Service** — enforces rules (e.g. "a model name must be unique", "a version can't be registered twice") and raises the correct `HTTPException`.
- **Repository** — the single place that issues SQLAlchemy queries. Swappable without touching business logic.

### Runtime topology (implemented)

```
              ┌──────────────────────────┐
              │        docker compose     │
              │                           │
   :8000 ───► │  api   (FastAPI/uvicorn)  │
              │    │                      │
              │    ▼  depends_on: healthy │
   :5432 ───► │  db    (PostgreSQL 15)    │
              │        volume: postgres_data
              └──────────────────────────┘
```

### Target production topology (roadmap)

```
                 Client
                   │  POST /predict
                   ▼
            FastAPI (API Gateway)
                   │
                   ▼
            Model Registry ──► Model Server × N   (Kubernetes, horizontal scaling)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
                  Redis        PostgreSQL       Metrics
                                                   │
                                                   ▼
                                              Prometheus ─► Grafana
```

---

## 4. Tech stack

| Concern                | Technology                        | Status         |
| ---------------------- | --------------------------------- | -------------- |
| API framework          | FastAPI + Uvicorn                 | ✅ Implemented |
| Validation / settings  | Pydantic v2, pydantic-settings    | ✅ Implemented |
| ORM                    | SQLAlchemy 2.x                    | ✅ Implemented |
| Database               | PostgreSQL 15                     | ✅ Implemented |
| DB driver              | psycopg2-binary                   | ✅ Implemented |
| Containerization       | Docker, Docker Compose            | ✅ Implemented |
| Testing                | pytest, httpx (TestClient)        | ✅ Implemented |
| Env / task runner      | Makefile, `pyproject.toml`        | ✅ Implemented |
| ML model               | scikit-learn (LogisticRegression) | 🚧 Planned     |
| Model format           | joblib                            | 🚧 Planned     |
| Experiment tracking    | MLflow                            | 🚧 Planned     |
| Data versioning        | DVC                               | 🚧 Planned     |
| Caching                | Redis                             | 🚧 Planned     |
| Metrics / dashboards   | Prometheus, Grafana               | 🚧 Planned     |
| Orchestration          | Kubernetes                        | 🚧 Planned     |
| CI/CD & linting        | GitHub Actions, Ruff, pre-commit  | 🚧 Planned     |

> Items marked 🚧 exist as **scaffolding** (empty placeholder files / commented dependencies) so the structure is ready, but they are **not yet implemented**. This README states current reality first and roadmap second, on purpose.

---

## 5. Project structure

```text
ML-Model-Serving-Platform/
│
├── src/
│   ├── __init__.py
│   ├── serving/                     # API domain — the serving platform itself
│   │   ├── app.py                   # FastAPI app entrypoint (+ /health)
│   │   ├── config.py                # Settings loaded from environment (.env)
│   │   ├── database.py              # SQLAlchemy engine, session, get_db()
│   │   ├── db_models.py             # ORM tables: DBModel, DBModelVersion
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── repository.py            # DB access layer (queries only)
│   │   ├── service.py               # Business logic layer
│   │   └── api.py                   # /models routes
│   │
│   ├── models/                      # ML domain (scaffolding — currently empty)
│   │   ├── train.py                 # (planned) training pipeline
│   │   └── evaluate.py              # (planned) offline evaluation
│   │
│   ├── monitoring/                  # (planned) post-deployment checks
│   │   └── drift_detector.py        # (planned) data/concept drift
│   │
│   └── utils/
│       └── logger.py                # (planned) structured logging
│
├── tests/
│   ├── test_health.py               # health endpoint test
│   └── test_registry.py             # 6 model-registry tests
│
├── deployment/
│   └── Dockerfile                   # Python 3.12-slim image + healthcheck
│
├── data/                            # tracked by DVC, ignored by Git (planned)
│   ├── raw.dvc
│   └── processed.dvc
│
├── .github/workflows/ci-cd.yaml     # (planned) CI/CD pipeline
├── docker-compose.yml               # api + postgres services
├── requirements.txt                 # Python dependencies
├── Makefile                         # venv bootstrap / cleanup
├── pyproject.toml                   # pytest config, pythonpath
├── .pre-commit-config.yaml          # (planned) Ruff hooks
├── env.example.                     # environment variable template
└── .gitignore
```

---

## 6. Data model (Model Registry)

The registry stores **metadata only** — it does not yet load or execute models. Two tables with a one-to-many relationship:

```
┌────────────────────────────┐          ┌──────────────────────────────────┐
│ models                     │  1     N │ model_versions                    │
├────────────────────────────┤ ───────► ├──────────────────────────────────┤
│ id           PK            │          │ id            PK                  │
│ name         UNIQUE, idx   │          │ model_id      FK → models.id      │
│ description  nullable      │          │ version                           │
│ created_at                 │          │ framework                         │
└────────────────────────────┘          │ model_format                      │
                                         │ artifact_path                     │
   cascade: delete-orphan                │ status  (default 'staging')       │
                                         │ created_at                        │
                                         │ UNIQUE(model_id, version)         │
                                         └──────────────────────────────────┘
```

Constraints enforced at the DB level:

- `models.name` is **unique** — no two models share a name.
- `(model_id, version)` is **unique** — a model can't register the same version twice.
- Deleting a model cascades to delete its versions (`delete-orphan`).

Lifecycle status is currently a free-text field with three intended values. Setting `status = "production"` is metadata only — no deployment happens yet:

```
fraud-model
 ├── 1.0.0 → staging
 ├── 1.1.0 → production
 └── 2.0.0 → archived
```

---

## 7. API reference

Interactive docs (Swagger UI) are auto-generated at **`http://localhost:8000/docs`** when the app is running.

### Health

| Method | Path      | Description        | Response            |
| ------ | --------- | ------------------ | ------------------- |
| `GET`  | `/health` | Liveness check     | `{"status": "ok"}`  |

### Models

| Method | Path                   | Description                          |
| ------ | ---------------------- | ------------------------------------ |
| `POST` | `/models`              | Register a new model                 |
| `GET`  | `/models`              | List all models                      |
| `GET`  | `/models/{model_name}` | Get one model by name (404 if absent)|

**Create a model**

```http
POST /models
Content-Type: application/json
```
```json
{ "name": "fraud-model", "description": "Fraud risk prediction model" }
```
Response `200`:
```json
{ "id": 1, "name": "fraud-model", "description": "Fraud risk prediction model", "created_at": "..." }
```
Duplicate name → `400 Bad Request` — `"Model with name 'fraud-model' already exists."`

### Model versions

| Method | Path                             | Description                     |
| ------ | -------------------------------- | ------------------------------- |
| `POST` | `/models/{model_name}/versions`  | Register a version for a model  |
| `GET`  | `/models/{model_name}/versions`  | List versions of a model        |

**Register a version**

```http
POST /models/fraud-model/versions
```
```json
{
  "version": "1.0.0",
  "framework": "scikit-learn",
  "model_format": "joblib",
  "artifact_path": "models/fraud-model/1.0.0/model.joblib",
  "status": "staging"
}
```
Duplicate version for the same model → `400 Bad Request`.
Unknown model → `404 Not Found`.

### End-to-end flow that works today

```
POST /models                       → fraud-model created
POST /models/fraud-model/versions  → 1.0.0 registered
GET  /models/fraud-model/versions  → [1.0.0]
```

> **Not implemented yet:** `POST /predict`. It appears in the serving contract above as the target design, but inference is a later ticket.

---

## 8. Design decisions & trade-offs

**Why a Model Registry instead of committing `model.joblib` to Git?**
Git is built for text/code, not large binaries — every retrain bloats history and slows `clone`/`pull`. A lone `.joblib` is a black box: no accuracy, no dataset version, no hyperparameters, no lineage. A registry stores that metadata, supports lifecycle stages (`staging`/`production`/`archived`), and lets deployment pick the "production" model **without changing application code**. In short: *Git manages code; a registry manages artifacts.*

**Why the three-layer (API → Service → Repository) split?**
Separation of concerns makes each layer independently testable and swappable. You can change the database (Repository) without touching business rules (Service), and change routing (API) without touching either.

**Why separate `Create` and `Response` Pydantic schemas?**
`Create` schemas carry only what a client may send; `Response` schemas add server-generated fields (`id`, `created_at`) and use `from_attributes=True` so they serialize straight from SQLAlchemy objects. This prevents leaking internal fields and keeps input validation strict.

**Why `scikit-learn` + `LogisticRegression` as the first model?**
The point isn't an exotic network — it's *reliable infrastructure around a model*. A boring baseline is ideal and can later be replaced by XGBoost behind the same API.

**Why config from environment variables?**
Twelve-factor style: `.env` is git-ignored, `env.example.` documents the required keys, and `pydantic-settings` validates them at startup. Secrets never enter version control.

**Trade-off accepted for now:** tables are created at app startup via `Base.metadata.create_all()` rather than migrations. Simple to run, but not how schema changes should be managed long-term (see below).

---

## 9. Getting started

### Prerequisites

- Docker & Docker Compose
- (Optional, for running outside containers) Python 3.12

### 1. Clone

```bash
git clone <your-repo-url>
cd ML-Model-Serving-Platform
```

### 2. Configure environment

```bash
cp env.example. .env
```

`.env` keys:

```dotenv
POSTGRES_USER=admin
POSTGRES_PASSWORD=secretpassword
POSTGRES_DB=ml_serving_db
POSTGRES_HOST=db          # 'db' inside Docker; 'localhost' if running uvicorn on the host
POSTGRES_PORT=5432
APP_ENV=development
```

### 3. Start the stack

```bash
docker compose up --build -d
```

This launches PostgreSQL, waits for it to be healthy, then starts the API on port 8000.

### 4. Verify

```bash
curl http://localhost:8000/health          # {"status":"ok"}
# open the interactive docs:
#   http://localhost:8000/docs
```

### Running the API on the host (optional)

```bash
make                                        # create .venv + install requirements
source .venv/bin/activate
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```
> When running on the host, set `POSTGRES_HOST=localhost` in `.env`.

### Troubleshooting

```bash
docker compose down
docker compose build --no-cache api
docker compose up -d
```

### Clean up

```bash
make clean            # remove the local virtual environment
docker compose down -v   # stop containers and drop the DB volume
```

---

## 10. Testing

Tests run **inside the container**, against a **real PostgreSQL** instance, for environment parity. The registry tests use a random UUID suffix per run so they don't collide across runs.

```bash
docker compose run --rm api pytest -v
```

Current coverage:

- `tests/test_health.py` — `GET /health` returns `200` and `{"status": "ok"}`.
- `tests/test_registry.py` — 6 tests: create model, reject duplicate model, list models, get model by name, register version, reject duplicate version.

> Note: `src/serving/app.py` calls `Base.metadata.create_all()` on import, so the test process needs a reachable database — which is why tests run through Docker Compose rather than bare `pytest` on the host.

---

## 11. Current status

Delivered as two Jira-style tickets:

| Ticket    | Title                              | Status  |
| --------- | ---------------------------------- | ------- |
| MSP-001   | Bootstrap ML Serving Platform      | ✅ Done |
| MSP-002   | Model Registry                     | ✅ Done |

**Implemented**

- ✅ FastAPI application + `/health`
- ✅ PostgreSQL via Docker Compose (with healthcheck & dependency ordering)
- ✅ Model Registry: `models` + `model_versions` tables with constraints
- ✅ Full CRUD-ish registry API (create/list/get models, register/list versions)
- ✅ Clean 3-layer architecture (API / Service / Repository)
- ✅ Pydantic v2 validation with separate Create/Response schemas
- ✅ Environment-based configuration (secrets kept out of Git)
- ✅ Dockerfile with layer caching + container healthcheck
- ✅ 7 automated tests passing against a real database

**Scaffolding present but empty** (ready for future tickets): `src/models/`, `src/monitoring/`, `src/utils/logger.py`, `.github/workflows/ci-cd.yaml`, `.pre-commit-config.yaml`, `data/*.dvc`.

---

## 12. Known limitations & design notes

Honest engineering notes — these are intentional simplifications for the current stage, not accidental gaps. Each maps to a future ticket.

- **No schema migrations.** Tables are created with `create_all()` at startup. Production should adopt **Alembic** so schema changes are versioned and reversible.
- **`status` is free text.** `staging` / `production` / `archived` are conventions, not enforced. A DB `Enum` (or a `CHECK` constraint) would prevent typos.
- **POST returns `200`, not `201 Created`.** Fine functionally; tightening to REST conventions is a small future polish.
- **No pagination** on `GET /models`. Acceptable at small scale; add limit/offset before the table grows.
- **No authentication or rate limiting yet.** These are explicitly scheduled (JWT + rate limiting) in Week 2 of the roadmap.
- **`datetime.utcnow` is used for timestamps.** Works, but is deprecated in newer Python; prefer timezone-aware `datetime.now(timezone.utc)`.
- **The registry stores `artifact_path` but nothing loads it yet.** Model loading and inference are deliberately out of scope until the registry is solid.

---

## 13. Roadmap / future improvements

The platform follows a phased plan. The registry is the foundation; production concerns build on top.

### Phase 1 — Backend foundation ✅ (done)
Bootstrap, Model Registry, layered architecture, testing.

### Phase 2 — Model lifecycle & production API 🚧
- Model **loading** from `artifact_path` into memory for inference
- **`POST /predict`** implementing the serving contract
- Version **status transitions** (active / staging) and pre-deploy **validation** (features/metadata compatibility)
- **Redis** caching for hot models/requests
- **JWT authentication** and **rate limiting**

### Phase 3 — Kubernetes & scaling 🚧
- Multi-stage Docker build
- Kubernetes `Deployment` / `Service` / `ConfigMap` / `Secret`
- **Horizontal scaling** of the model server
- `/liveness` + `/readiness` probes, CPU/memory requests & limits
- **Load testing** with Locust; capture **p50 / p95 / p99** latency and throughput

### Phase 4 — Professional MLOps 🚧
- **Prometheus** metrics (`request_count`, `request_latency`, `prediction_count`, `error_count`, `model_version`)
- **Grafana** dashboards
- **Canary deployment** (e.g. 95% → v1, 5% → v2) and **A/B testing**
- **Automatic rollback** on error-rate / latency regression
- **Failure simulation** (kill model server, kill DB, Redis outage, bad model, traffic spikes)
- **CI/CD** with GitHub Actions (lint → test → build → deploy) and **Ruff** pre-commit hooks
- Model/data **drift detection**

### Beyond — portfolio arc
This repo is Project 1 of a planned three-project MLOps portfolio:

```
ML Application            ML Infrastructure          Security ML
(Fraud Detection)   ─►    (this: Model Serving)  ─►  (AI Security Detection)
```

The next project (AI Security Detection Platform: Kafka streaming → feature engineering → anomaly detection → alerting) is designed to **reuse this platform's serving infrastructure** for inference.

---

## 14. Development workflow

The project is built ticket-by-ticket, mirroring a real team:

```
Jira Ticket → design → implement → pytest → git commit → code review → next ticket
```

Golden rules carried across tickets:

- **Commit every day** — even 30 lines.
- **Every feature runs in Docker** — no "works on my machine".
- **Every endpoint has a test** — `pytest` must be green before a ticket is Done.
- **Report artifacts, not hours** — *"Feature X merged"*, not *"studied X hours"*.
- **One ticket at a time** — don't start the next until the current one passes review.

A ticket is **Done** only when:

```bash
docker compose up --build   # succeeds
curl /health                # returns 200
pytest                      # no failures
```

---

## 15. Author

**Nirvana Fanaelahi**

Built as a hands-on MLOps / ML-infrastructure portfolio project. The emphasis is deliberately on **production engineering around a model** — versioning, serving, deployment, and observability — rather than on the model itself.

---

*This README reflects the actual state of the codebase (MSP-001 and MSP-002 complete). Features labelled "planned" / 🚧 are scaffolded but not yet implemented, and are tracked in the roadmap above.*
