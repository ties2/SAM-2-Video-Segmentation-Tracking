# ---- build stage: compile wheels once, keep compilers out of the runtime image ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 SAM2_BUILD_CUDA=0
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /wheels
COPY requirements.txt requirements-vos.txt ./
RUN pip wheel -r requirements.txt -w /wheels \
 && pip wheel --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1 -w /wheels \
 && pip wheel --no-deps "sam2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4" -w /wheels \
 && pip wheel hydra-core iopath tqdm matplotlib -w /wheels

# ---- runtime stage ----
FROM python:3.12-slim
LABEL authors="Nirvana Fanaelahi"
ENV PYTHONUNBUFFERED=1 APP_HOME=/app PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app
WORKDIR $APP_HOME
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels
COPY --chown=app:app . .
RUN mkdir -p /app/model_artifacts /app/weights && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
