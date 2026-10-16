# script for define tables of database

from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import relationship

from src.serving.database import Base


class DBModel(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    versions = relationship("DBModelVersion", back_populates="model", cascade="all, delete-orphan")
    deployments = relationship("DBModelDeployment", back_populates="model", cascade="all, delete-orphan")


class DBModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    version = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    model_format = Column(String, nullable=False)
    artifact_uri = Column(String, nullable=False)
    status = Column(String, nullable=False, default="staging")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    model = relationship("DBModel", back_populates="versions")

    __table_args__ = (
        UniqueConstraint("model_id", "version", name="_model_version_uc"),
        # "at most one production version per model", enforced by the database itself.
        # sqlite_where keeps the same guarantee when running locally / in unit tests on SQLite.
        Index(
            "uix_model_production",
            "model_id",
            unique=True,
            postgresql_where=text("status = 'production'"),
            sqlite_where=text("status = 'production'"),
        ),
    )


class DBModelDeployment(Base):
    __tablename__ = "model_deployments"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)

    # Version Tracking
    stable_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    canary_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)

    # Traffic Splitting Weights
    stable_weight = Column(Integer, nullable=False, default=100)
    canary_weight = Column(Integer, nullable=False, default=0)

    # Statuses: ACTIVE_CANARY, PROMOTED, ROLLED_BACK
    status = Column(String, nullable=False, default="ACTIVE_CANARY")

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    # Relationships
    model = relationship("DBModel", back_populates="deployments")
    stable_version = relationship("DBModelVersion", foreign_keys=[stable_version_id])
    canary_version = relationship("DBModelVersion", foreign_keys=[canary_version_id])

    __table_args__ = (
        # Ensure only one ACTIVE deployment exists per model at any given time
        Index(
            "uix_active_model_deployment",
            "model_id",
            unique=True,
            postgresql_where=text("status = 'ACTIVE_CANARY'"),
            sqlite_where=text("status = 'ACTIVE_CANARY'"),
        ),
    )
