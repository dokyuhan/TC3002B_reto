# TC3002B Reto: Source Code Plagiarism and AI Detection

This repository contains the software for the capstone challenge in course TC3002B.570. The system detects pairwise Java source code plagiarism by scoring structural AST Markov transitions with selectable classifiers (`Pairwise LightGBM` default, `Pairwise CNN` selectable). Single file AI code classification is planned for Phase 2. The live application runs at https://tc3002b-reto.onrender.com/. Read the [architecture diagram](docs/architecture.mmd) and [evaluation report](docs/evaluation/metrics_tables.md) for system details and benchmark results.

## Team

The project team has 3 members:

* Fernando Daniel Monroy Monroy
* Diego De la Vega Saishio
* Do Kyu Han

## How it works

1. **Parse.** Each Java program is parsed into an Abstract Syntax Tree (AST) with Tree Sitter. Nodes map to 12 structural states (`IMPORT`, `FUNCTION`, `CLASS`, `ASSIGN`, `IF`, `FOR`, `WHILE`, `CALL`, `RETURN`, `EXCEPTION`, `OP`, `OTHER`).
2. **Markov matrix.** Each state sequence is converted into a 12 x 12 first order Markov transition matrix describing structural flow independent of identifier names and formatting.
3. **Classify.** Two permutation symmetric pairwise classifiers are persisted via `models/manifest.json` and selectable at runtime:
   * **Pairwise LightGBM (`lgbm`, default):** Platt calibrated gradient boosted classifier on smoothed (`alpha = 0.01`) symmetric features, divergence metrics, and sequence perplexities.
   * **Pairwise CNN (`cnn`, selectable):** Compact PyTorch trained, NumPy evaluated CNN on unsmoothed (`alpha = 0.0`) 3 channel tensors `[P_A, P_B, P_A - P_B]`, symmetrized across `(A, B)` and `(B, A)`.
4. **Compare.** Alongside the verdict and active decision threshold, the dashboard reports model agnostic Markov similarity (Jensen Shannon divergence), Frobenius distance, a character n gram TF IDF cosine baseline, an AST transition delta heatmap, and a side by side code diff.

Both models learn binary pairwise plagiarism labels (ConPlag verdicts and IR Plag cases), not AI authorship labels.

## Repository structure

* `src/vocabulary.py`: Defines the 12 AST states and supported languages.
* `src/preprocessing/`: Normalizes source code and parses files into AST state sequences.
* `src/features/`: Builds Markov transition matrices and pairwise tensors.
* `src/models/`: Implements similarity baselines (`baseline.py`), LightGBM classifier (`classifier.py`), NumPy CNN forward pass (`cnn_numpy.py`), the task isolated `pairwise_plagiarism` registry (`registry.py`), and the `ai_authorship` scorer contract (`authorship.py`).
* `src/inference.py`: Loads isolated `pairwise_plagiarism` and `ai_authorship` task bundles and runs inference in memory.
* `src/api/`: Exposes FastAPI endpoints (`/health`, `/predict/pairwise_plagiarism`, `/predict`, `/predict/ai_authorship`).
* `ui/`: Runs the Streamlit dashboard with separate `Pairwise Plagiarism` and `AI Authorship` tabs plus `3` Java demo pairs.
* `models/`: Split cleanly by task into `models/pairwise_plagiarism/` (`manifest.json`, `lgbm_metadata.json`, `cnn_metadata.json`, `plagiarism_lgbm.joblib`, `cnn_weights.npz`, `tfidf_baseline.joblib`) and `models/ai_authorship/` (`manifest.json` ready for single file AI authorship models).
* `model_training/`: Trains and evaluates the CNN with PyTorch and exports NumPy weights into `models/`. See its [README](model_training/README.md).
* `scripts/`: Runs dataset preparation (`build_datasets.py`), LightGBM + TF IDF training (`train.py`, `fit_tfidf.py`), cross domain evaluation (`evaluate.py`), and chart generation (`generate_plots.py`).
* `dataset/`: Holds `raw/` (ConPlag and IR Plag pairs) and `preprocessed/` (AST sequences). Derived `processed/` matrices and `results/` JSON files are gitignored.
* `docs/`: Stores the architecture diagram, evaluation metrics tables, and figures.
* `tests/`: Runs unit, parity, and FastAPI tests.

## Setup

Install Python 3.12 and [`uv`](https://docs.astral.sh/uv/) before you start. Run all commands from the repository root.

```bash
uv sync --locked                 # app dependencies (no PyTorch)
uv run ruff check --select E4,E7,E9,F .
uv run pytest
```

## Run the app

The app loads all registered artifacts from `models/manifest.json`.

**Option 1: two terminals (development).**

```bash
# terminal 1: API on http://localhost:8000
uv run uvicorn src.api.main:app --port 8000

# terminal 2: UI on http://localhost:8501
uv run streamlit run ui/app.py
```

The UI calls `API_URL` (default `http://localhost:8000`). Set `MODEL_DIR` to load artifacts from another directory (default `models/`).

**Option 2: Docker Compose (API on 8000, UI on 8501).**

```bash
docker compose up --build
```

**Option 3: single container (Render deployment).** `entrypoint.sh` starts FastAPI, waits until `/health` returns HTTP 200 with `model_loaded: true`, and then launches Streamlit on `$PORT` (default `7860`).

```bash
docker build -t tc3002b-reto .
docker run -p 7860:7860 tc3002b-reto
```

**Call the API directly:**

```bash
curl http://localhost:8000/health

# Pairwise plagiarism (2 files; /predict kept as backward compatible alias)
curl -X POST http://localhost:8000/predict/pairwise_plagiarism \
  -H "Content-Type: application/json" \
  -d '{"code_a": "<java source A>", "code_b": "<java source B>", "language": "java", "operating_mode": "balanced", "model": "lgbm"}'

# Single file AI authorship (1 file; returns HTTP 503 until an ai_authorship model is registered)
curl -X POST http://localhost:8000/predict/ai_authorship \
  -H "Content-Type: application/json" \
  -d '{"code": "<java source>", "language": "java", "operating_mode": "balanced"}'
```

`GET /health` returns per task readiness under `tasks` (`pairwise_plagiarism` and `ai_authorship`) alongside top level `default_model` and `available_models` descriptors. `POST /predict/pairwise_plagiarism` accepts optional `model` (`lgbm` or `cnn`, defaulting to `default_model`) and `operating_mode` (`balanced` or `low_fpr`). `POST /predict/ai_authorship` scores single file submissions via registered `AuthorshipScorer` adapters under `models/ai_authorship/`.

## Training and evaluation

```bash
# 1. Raw Java pairs -> AST sequences (dataset/preprocessed)
uv run python -m scripts.build_datasets

# 2. Train LightGBM + TF IDF baseline (updates models/lgbm_metadata.json and models/manifest.json)
uv run python -m scripts.train

# 3. Train and export optional PyTorch CNN (CPU wheel index via --group train)
uv run --group train python -m model_training.main build-data
uv run --group train python -m model_training.main train
uv run --group train python -m model_training.main evaluate
uv run --group train python -m model_training.main evaluate --ir-plag
uv run --group train python -m model_training.export_numpy
uv run --group train python -m pytest tests/test_numpy_parity.py

# 4. Run full evaluation report and regenerate charts (docs/evaluation/)
uv run python -m scripts.evaluate
uv run python -m scripts.generate_plots
```

## Deployment

CI (`.github/workflows/ci.yaml`) runs ruff lint and pytest on every push and pull request to `main`. Passing pushes to `main` trigger a Render deploy using `Dockerfile`, which packages `src/`, `ui/`, and `models/` without PyTorch or `dataset/`.
