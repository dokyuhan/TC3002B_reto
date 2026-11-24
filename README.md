# TC3002B Reto: AI Code Detection

This repository contains the software for the capstone challenge in course TC3002B.570. The system estimates whether **one Java function** was written by a human or by an AI. You paste a function, and the model returns the probability that an AI wrote it, based on the structure of the code. Read the [architecture diagram](docs/architecture.mmd) for the system layout.

> This branch holds the single function AI detector. The earlier pairwise plagiarism checker (two programs in, copy probability out) lives on the `main` branch, so the two approaches can be compared.

## Team

The project team has 3 members:

* Fernando Daniel Monroy Monroy
* Diego De la Vega Saishio
* Do Kyu Han

## How it works

1. **Parse.** The Java function is parsed into an Abstract Syntax Tree (AST). Every node is mapped to one of 12 states (`IMPORT`, `FUNCTION`, `CLASS`, `ASSIGN`, `IF`, `FOR`, `WHILE`, `CALL`, `RETURN`, `EXCEPTION`, `OP`, `OTHER`), which turns the code into a sequence of states. Comments and formatting are ignored on purpose.
2. **Markov matrix.** Count how often state *i* is followed by state *j* and normalize each row. The result is a 12 x 12 transition matrix that describes the structure of the code.
3. **Classify.** A small CNN reads the matrix and outputs the probability that an AI wrote the function. The decision threshold depends on the sensitivity you choose: *Balanced*, or *Fewer false alarms*.

The model is trained on [HumanVsAICode](https://huggingface.co/datasets/OSS-forge/HumanVsAICode): the same Java functions written by a human and by ChatGPT, DeepSeek-Coder and Qwen2.5-Coder.

## Repository structure

* `src/vocabulary.py`: Defines the 12 AST states and the supported languages.
* `src/preprocessing/`: Normalizes source code and parses it into AST state sequences.
* `src/features/`: Builds the Markov transition matrix of a state sequence.
* `src/models/`: Runs the trained CNN with numpy (`cnn_numpy.py`), so the app does not need PyTorch.
* `src/inference.py`: Runs the full analysis of one function in memory.
* `src/api/`: Exposes the FastAPI HTTP endpoints.
* `ui/`: Runs the Streamlit dashboard and stores 4 Java demo functions.
* `models/`: Stores the files the app loads (`ai_detector_weights.npz`, `ai_detector_metadata.json`).
* `model_training/`: Trains and evaluates the CNN with PyTorch and exports its weights to `models/`. See its [README](model_training/README.md).
* `scripts/`: Builds the preprocessed dataset (`build_human_vs_ai.py`).
* `dataset/`: Holds `raw/` (HumanVsAICode, downloaded by you), `preprocessed/` (AST sequences), `processed/` (CNN input matrices) and `results/` (training history and metrics).
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
  -d '{"code": "<one java method>", "language": "java", "operating_mode": "balanced"}'
```

`/predict` returns `ai_probability`, `predicted_label` (1 = AI), `threshold_used`, `ast_tokens`, `parse_warnings`, an `input_note` when the input is a whole class instead of a method, and the 12 x 12 `transition_matrix` of the function. `operating_mode` can be `balanced` or `low_fpr`; their thresholds are stored in `models/ai_detector_metadata.json`.

## Train the model

Training needs PyTorch, which is not part of the app image. It lives in its own dependency group, so the commands below use `uv run --group train` and uv installs PyTorch the first time you run them. Run everything from the repository root.

```bash
# 0. Download the raw dataset once (505 MB, gitignored)
mkdir -p dataset/raw/human_vs_ai_code
curl -L -o dataset/raw/human_vs_ai_code/java_dataset.jsonl \
  https://huggingface.co/datasets/OSS-forge/HumanVsAICode/resolve/main/java_dataset.jsonl

# 1. Raw functions -> AST state sequences (dataset/preprocessed)
uv run python -m scripts.build_human_vs_ai

# 2. AST sequences -> one 12 x 12 matrix per function (dataset/processed)
uv run --group train python -m model_training.main build-data

# 3. Train (keeps the best epoch in model_training/checkpoints/best.pt)
uv run --group train python -m model_training.main train

# 4. Evaluate on the test split, with a per-author report
uv run --group train python -m model_training.main evaluate

# 5. Copy the trained weights into models/ for the app
uv run --group train python -m model_training.export_numpy

# 6. Check that the app and PyTorch agree
uv run --group train python -m pytest model_training/tests
```

All hyperparameters live in `model_training/config.py`; metrics are written to `dataset/results/`. See the [model_training README](model_training/README.md) for what every step and function does.

## Results

Test split of HumanVsAICode: 10,324 Java functions that never appear in training. The threshold (0.50) was chosen on validation.

| Metric | CNN | Logistic baseline |
|---|---|---|
| Balanced accuracy | 0.732 | 0.698 |
| ROC-AUC | 0.823 | 0.774 |
| AI code detected | 0.663 | 0.610 |
| Human code flagged as AI | 0.199 | 0.213 |

Detection per AI: DeepSeek 0.83, ChatGPT 0.63, Qwen 0.51. Training without Qwen at all gives the same Qwen detection (0.52), so the model is not memorizing the style of the AIs it saw. The *Fewer false alarms* mode (threshold 0.68) flags only 5% of human code as AI and detects 44.5% of AI code.

Limitations:

* The detector is a moderate signal, not proof: about 20% of human functions are flagged and a third of AI functions are missed.
* It works on single Java methods and has not been tested on whole files. The AI code in the training data was generated from docstrings.
* Only structure is used. Comments and formatting are ignored on purpose, because in this dataset they are shortcuts (human code has almost no comments).
* A 12 x 12 matrix discards names, nesting and statement order, which limits what any model can learn from it. A gradient-boosted tree model scored higher than the CNN on the same matrices in a quick check (ROC-AUC about 0.85 against 0.82).
* Only Java is calibrated.

## Deployment

CI (`.github/workflows/ci.yaml`) runs the linter and the tests on every push and pull request to `main`. A push to `main` that passes triggers a Render deploy, which builds the `Dockerfile`. The image contains `src/`, `ui/` and `models/` only. It does not contain PyTorch or `dataset/`.
