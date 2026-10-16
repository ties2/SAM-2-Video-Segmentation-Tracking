"""Shared pytest setup.

* Defaults to a throw-away SQLite database when no DATABASE_URL / POSTGRES_* is set,
  so `pytest` works on a laptop and in CI without Docker.  CI can still point
  DATABASE_URL at PostgreSQL for environment parity.
* Creates the tables once per session.  The original suite relied on the API
  container having created them already (TestClient was never used as a context
  manager, so the FastAPI lifespan never ran) — on a fresh database every test failed.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="msp_tests_")
if not os.environ.get("DATABASE_URL") and not os.environ.get("POSTGRES_HOST"):
    os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/msp_test.db"
os.environ.setdefault("LOCAL_ARTIFACT_DIR", os.path.join(_tmp, "artifacts"))

import pytest  # noqa: E402

from src.serving.app import app  # noqa: E402,F401  (imports every ORM model)
from src.serving.database import Base, engine  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    Base.metadata.create_all(bind=engine)
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs torch + SAM 2 weights (deselect with -m 'not slow')")
