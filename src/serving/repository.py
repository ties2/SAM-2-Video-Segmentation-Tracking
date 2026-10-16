from sqlalchemy.orm import Session

from src.serving import db_models, schemas


class ModelRepository:
    @staticmethod
    def get_model_by_name(db: Session, name: str):
        return db.query(db_models.DBModel).filter(db_models.DBModel.name == name).first()

    @staticmethod
    def get_all_models(db: Session):
        return db.query(db_models.DBModel).all()

    @staticmethod
    def create_model(db: Session, model_data: schemas.ModelCreate):
        db_model = db_models.DBModel(name=model_data.name, description=model_data.description)
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model


class ModelVersionRepository:
    @staticmethod
    def get_version_by_model_and_name(db: Session, model_id: int, version: str):
        return (
            db.query(db_models.DBModelVersion)
            .filter(db_models.DBModelVersion.model_id == model_id, db_models.DBModelVersion.version == version)
            .first()
        )

    @staticmethod
    def get_versions_by_model(db: Session, model_id: int):
        return db.query(db_models.DBModelVersion).filter(db_models.DBModelVersion.model_id == model_id).all()

    @staticmethod
    def create_model_version(db: Session, model_id: int, version_data: schemas.ModelVersionCreate):
        db_version = db_models.DBModelVersion(
            model_id=model_id,
            version=version_data.version,
            framework=version_data.framework,
            model_format=version_data.model_format,
            artifact_uri=version_data.artifact_uri,
            status=version_data.status,
        )
        db.add(db_version)
        db.commit()
        db.refresh(db_version)
        return db_version

    # =========================================================
    # --- New Methods for MSP-008 (Lifecycle & Promotion) ---
    # =========================================================

    @staticmethod
    def get_production_version(db: Session, model_id: int):
        """Find the version that currently has the production status"""
        return (
            db.query(db_models.DBModelVersion)
            .filter(db_models.DBModelVersion.model_id == model_id, db_models.DBModelVersion.status == "production")
            .first()
        )

    @staticmethod
    def promote_transactionally(db: Session, model_id: int, target_version: db_models.DBModelVersion):
        """
        Promote a version transactionally (Atomic).
        If any error occurs, all changes will be rolled back.
        """
        try:
            # 1. Find the current production version
            current_prod = (
                db.query(db_models.DBModelVersion)
                .filter(db_models.DBModelVersion.model_id == model_id, db_models.DBModelVersion.status == "production")
                .first()
            )

            # 2. Archive the current production version (if it exists)
            if current_prod:
                current_prod.status = "archived"

            # 3. Change the target version status to production
            target_version.status = "production"

            # 4. Commit all changes together in a single transaction
            db.commit()
            db.refresh(target_version)

            return target_version, current_prod

        except Exception as e:
            # 5. In case of any error, rollback the database to its previous state
            db.rollback()
            raise e
