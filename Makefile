VENV = .venv
PY = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: all setup setup-vos weights samples synth benchmark smoke report test test-slow lint format api seed-sam2 clean

all: setup

# ---- environments -------------------------------------------------------------------
$(VENV): requirements.txt requirements-dev.txt
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	@touch $(VENV)

setup: $(VENV)                      ## API + metrics + tests (no torch)

setup-vos: setup                    ## + PyTorch CPU wheels + official SAM 2
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
	SAM2_BUILD_CUDA=0 $(PIP) install -r requirements-vos.txt

# ---- data & weights -----------------------------------------------------------------
weights:                            ## SAM 2.1 hiera-tiny (Meta URL, GitHub mirror fallback)
	$(PY) -m src.vos.cli fetch-weights --variant tiny --out weights

samples:                            ## DAVIS car-shadow, SA-V example, SAM 2 demo clips
	$(PY) -m src.vos.cli fetch-samples --out data/samples

synth:                              ## synthetic stress-test videos with exact GT
	$(PY) -m src.vos.cli synth --out data/synthetic

# ---- experiments --------------------------------------------------------------------
smoke:                              ## 1-minute end-to-end check of the benchmark pipeline
	$(PY) -m src.vos.cli benchmark --config configs/smoke.yaml

benchmark:                          ## Phase-1 benchmark (≈1 h on 2 vCPU, minutes on a GPU)
	$(PY) -m src.vos.cli benchmark --config configs/phase1.yaml

report:                             ## plots, demo GIFs and results/phase1/REPORT.md
	$(PY) -m src.vos.report --results results/phase1 --config configs/phase1.yaml

# ---- quality ------------------------------------------------------------------------
test:                               ## fast suite (SQLite, fake segmenter) — what CI runs
	$(PY) -m pytest -m "not slow"

test-slow:                          ## real SAM 2 inference (needs weights)
	$(PY) -m pytest -m slow

lint:
	$(VENV)/bin/ruff check . && $(VENV)/bin/ruff format --check .

format:
	$(VENV)/bin/ruff check --fix . && $(VENV)/bin/ruff format .

# ---- serving ------------------------------------------------------------------------
api:                                ## run the API on the host with SQLite
	DATABASE_URL=sqlite:///./msp.db LOCAL_ARTIFACT_DIR=./model_artifacts $(VENV)/bin/uvicorn src.serving.app:app --reload

seed-sam2:                          ## register + promote the SAM 2 checkpoint in a running API
	$(PY) -m src.models.seed_sam2 --checkpoint weights/sam2.1_t.pt --version 2.1-tiny

clean:
	rm -rf $(VENV) .cache
