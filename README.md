# TC3002B Reto: Source Code Plagiarism and AI Detection

This repository contains the software for the capstone challenge in course `TC3002B.570` (`Equipo 4`).

Phase 1 delivers pairwise Java source code plagiarism detection backed by Tree Sitter AST parsing, 12 state Markov transition tensors, mathematical similarity baselines, a calibrated symmetric LightGBM classifier, a FastAPI service, and a Streamlit dashboard. Single file AI generated code detection and C++ or Python calibration are staged for Phase 2.

## Team

* Fernando Daniel Monroy Monroy
* Diego De la Vega Saishio
* Do Kyu Han

## Architecture

The pipeline parses source code via Tree Sitter into 12 state AST sequences, builds `(3, 12, 12)` 1st order Markov transition tensors alongside character n gram TF IDF baselines, scores pairs with a calibrated symmetric LightGBM classifier, exposes predictions over FastAPI, and visualizes comparisons in Streamlit.

[Architecture Diagram](docs/architecture.mmd)

## Structure

* `src/vocabulary.py`: Shared 12 state AST vocabulary and language constants.
* `src/preprocessing/`: Source normalizer (`normalizer.py`) and Tree Sitter AST parser (`parser.py`).
* `src/features/`: Markov transition matrix math (`markov.py`) and batch `.npz` extraction (`extraction.py`).
* `src/models/`: TF IDF and Markov statistical baselines (`baseline.py`) and calibrated symmetric LightGBM classifier (`classifier.py`).
* `src/inference.py`: Runtime inference facade decoupling HTTP transport from ML internals.
* `src/api/`: Thin FastAPI HTTP service (`main.py`).
* `scripts/`: Offline pipeline CLIs for dataset preparation (`build_datasets.py`), training (`train.py`), evaluation (`evaluate.py`), and plotting (`generate_plots.py`).
* `ui/`: Streamlit interactive dashboard (`app.py`) and sanitized demo fixtures (`samples.py`).
* `models/`: Tracked lightweight production model bundle (`plagiarism_lgbm.joblib`, `tfidf_baseline.joblib`, `model_metadata.json`).
* `docs/`: Architecture diagram and reproducible evaluation tables and plots.
* `tests/`: Unit and API test suites.

## Quickstart

Prerequisites: Python 3.12 and `uv`.

```bash
# Install locked dependencies into .venv
uv sync --locked

# Run linter
uv run ruff check --select E4,E7,E9,F .

# Run unit tests
uv run pytest

# Run API and UI via Docker Compose
docker compose up --build
```
