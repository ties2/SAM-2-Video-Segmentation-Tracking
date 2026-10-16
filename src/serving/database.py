# script for connect to database

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.serving.config import settings

SQLALCHEMY_DATABASE_URL = settings.sqlalchemy_url()

# SQLite (local runs / unit tests) needs check_same_thread=False because FastAPI
# serves sync endpoints from a thread pool; PostgreSQL needs nothing extra.
_connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
