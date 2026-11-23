# TC3002B Reto: Source Code Plagiarism and AI Detection

This repository contains the software for the capstone challenge in course TC3002B.570. The system detects pairwise Java source code plagiarism: you give it two programs and it returns the probability that one is a plagiarized version of the other. AI code classification is planned for Phase 2. The live application runs at https://tc3002b-reto.onrender.com/. Read the [architecture diagram](docs/architecture.mmd) for the system layout.

## Team

The project team has 3 members:

* Fernando Daniel Monroy Monroy
* Diego De la Vega Saishio
* Do Kyu Han

## How it works

1. **Parse.** Each Java program is parsed into an Abstract Syntax Tree (AST). Every node is mapped to one of 12 states (`IMPORT`, `FUNCTION`, `CLASS`, `ASSIGN`, `IF`, `FOR`, `WHILE`, `CALL`, `RETURN`, `EXCEPTION`, `OP`, `OTHER`), which turns a program into a sequence of states.
2. **Markov matrix.** For each program, count how often state *i* is followed by state *j* and normalize each row. The result is a 12 x 12 transition matrix that describes the structure of the code and ignores names and formatting.
3. **Classify.** A small CNN receives three channels `[P_A, P_B, P_A - P_B]` and outputs the probability that the pair is plagiarized. The app averages the answer for `(A, B)` and `(B, A)`, so the result does not depend on the order of the two programs.
4. **Compare.** The UI also shows reference measures next to the verdict: Markov similarity (Jensen-Shannon divergence), Frobenius distance, a TF-IDF text similarity baseline, and a heatmap of the transition difference.

The label the model learns is "plagiarized pair" (ConPlag verdicts and IR Plag cases). It is not an AI-authorship label.

## Repository structure

* `src/vocabulary.py`: Defines the 12 AST states and the supported languages.
* `src/preprocessing/`: Normalizes source code and parses files into AST state sequences.
* `src/features/`: Builds Markov transition matrices and extracts feature tensors.
* `src/models/`: Implements the similarity baselines (`baseline.py`) and the numpy forward pass of the CNN (`cnn_numpy.py`).
* `src/inference.py`: Runs the full pairwise analysis in memory.
* `src/api/`: Exposes the FastAPI HTTP endpoints.
* `ui/`: Runs the Streamlit dashboard and stores 3 Java demo pairs.
* `models/`: Stores the trained files the app loads (`cnn_weights.npz`, `tfidf_baseline.joblib`, `model_metadata.json`).
* `model_training/`: Trains and evaluates the CNN with PyTorch and exports its weights to `models/`. See its [README](model_training/README.md).
* `scripts/`: Builds the preprocessed datasets and fits the TF-IDF baseline.
* `dataset/`: Holds `raw/` (ConPlag and IR Plag pairs), `preprocessed/` (AST sequences), `processed/` (CNN input matrices, generated, not committed) and `results/` (training metrics, generated, not committed).
* `docs/`: Stores the architecture diagram.
* `tests/`: Runs unit and API tests.

## Setup

Install Python 3.12 and [`uv`](https://docs.astral.sh/uv/) before you start. Run all commands from the repository root.

```bash
uv sync --locked                 # app dependencies (no PyTorch)
uv run ruff check --select E4,E7,E9,F .
uv run pytest
```

## Run the app

The app needs the files in `models/`, which are committed to the repository.

**Option 1: two terminals (development).**

```bash
# terminal 1: API on http://localhost:8000
uv run uvicorn src.api.main:app --port 8000

# terminal 2: UI on http://localhost:8501
uv run streamlit run ui/app.py
```

The UI calls the API at `http://localhost:8000`. Set `API_URL` to point it somewhere else. Set `MODEL_DIR` to load the model files from another folder (default: `models/`).

**Option 2: Docker Compose (API on 8000, UI on 8501).**

```bash
docker compose up --build
```

**Option 3: one container (what Render runs).** `entrypoint.sh` starts the API, waits for `/health`, then starts the UI on `$PORT` (default 7860).

```bash
docker build -t tc3002b-reto .
docker run -p 7860:7860 tc3002b-reto
```

**Call the API directly:**

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code_a": "<java source A>", "code_b": "<java source B>", "language": "java", "operating_mode": "balanced"}'
```

`/predict` returns `plagiarism_probability`, `predicted_label`, `threshold_used`, the Markov and TF-IDF similarity scores, perplexities, AST token counts, parse warnings, and the 12 x 12 `transition_delta`. `operating_mode` can be `balanced` or `low_fpr`; both use the thresholds in `models/model_metadata.json`.

## Train the CNN

Training needs PyTorch, which is not part of the app image. It lives in its own dependency group, so the commands below use `uv run --group train` and uv installs PyTorch the first time you run them. Run the whole pipeline from the repository root:

```bash
# 1. Raw Java pairs -> AST sequences (dataset/preprocessed). Only needed if dataset/raw changes.
uv run python -m scripts.build_datasets

# 2. AST sequences -> 3-channel matrices (dataset/processed)
uv run --group train python -m model_training.main build-data

# 3. Train (keeps the best epoch in model_training/checkpoints/best.pt)
uv run --group train python -m model_training.main train

# 4. Evaluate on the ConPlag test split and on the unseen IR Plag dataset
uv run --group train python -m model_training.main evaluate
uv run --group train python -m model_training.main evaluate --ir-plag

# 5. Copy the trained weights into the app (models/cnn_weights.npz and models/model_metadata.json)
uv run --group train python -m model_training.export_numpy

# 6. Check that the app and PyTorch agree
uv run --group train python -m pytest model_training/tests
```

Run step 5 after every retraining, otherwise the app keeps serving the old weights. All hyperparameters live in `model_training/config.py`. Metrics are written to `dataset/results/`. The TF-IDF baseline is fitted separately with `uv run python -m scripts.fit_tfidf`.

See the [model_training README](model_training/README.md) for what every step and function does.

## Results

Metrics of the model the app serves (threshold 0.5, probabilities averaged over both input orders), from `models/model_metadata.json`:

| Dataset | Accuracy | Recall | False positive rate | PR-AUC |
|---|---|---|---|---|
| ConPlag validation (168 pairs) | 0.786 | 0.362 | 0.050 | 0.646 |
| ConPlag test (181 pairs) | 0.890 | 0.609 | 0.015 | 0.878 |
| IR Plag, never seen in training (460 pairs) | 0.611 | 0.589 | 0.314 | 0.868 |

Limitations:

* The model rarely accuses non-plagiarized code on ConPlag, but it misses part of the plagiarized pairs (recall 0.61).
* On IR Plag it separates easy plagiarism well but struggles on the hardest levels (`L5`, `L6`) and flags many non-plagiarized pairs.
* A 12 x 12 matrix discards names, nesting and statement order, which limits what any model can learn from it.
* Only Java is calibrated.
* The labels describe plagiarism between two programs, not whether a program was written by an AI.

## Deployment

CI (`.github/workflows/ci.yaml`) runs the linter and the tests on every push and pull request to `main`. A push to `main` that passes triggers a Render deploy, which builds the `Dockerfile`. The image contains `src/`, `ui/` and `models/` only. It does not contain PyTorch or `dataset/`.
