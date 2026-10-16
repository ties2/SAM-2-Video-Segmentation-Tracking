from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"

    # --- Database -------------------------------------------------------------------
    # DATABASE_URL wins when set (e.g. sqlite:///./msp.db for local runs / unit tests);
    # otherwise the PostgreSQL URL is assembled from the POSTGRES_* variables.
    database_url: str | None = None
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None
    postgres_host: str | None = None
    postgres_port: int = 5432

    # --- Artifacts & cache ------------------------------------------------------------
    ARTIFACT_STORAGE_BACKEND: str = "local"
    LOCAL_ARTIFACT_DIR: str = "/app/model_artifacts/"
    # SAM 2 checkpoints are 80 MB–900 MB; bound the number of models held in RAM.
    model_cache_max_items: int = 4

    # --- Video segmentation (synchronous API guard-rails) ----------------------------
    vos_device: str = "cpu"
    vos_num_threads: int | None = None
    vos_max_upload_mb: int = 200
    vos_max_frames: int = 300
    vos_default_stride: int = 1

    # --- Monitoring -------------------------------------------------------------------
    gf_security_admin_user: str | None = None
    gf_security_admin_password: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        missing = [
            n for n in ("postgres_user", "postgres_password", "postgres_db", "postgres_host") if not getattr(self, n)
        ]
        if missing:
            raise RuntimeError(f"Set DATABASE_URL or the POSTGRES_* variables (missing: {', '.join(missing)})")
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
