# Citations and Attributions

## Datasets

This branch trains a single function human versus AI code detector. It uses one dataset that we added ourselves; it is not part of the course dataset catalog (`TC3002B.570`, `data/raw/onedrive/url.txt`).

1. HumanVsAICode (Java subset). Each function has one human implementation and up to three AI implementations (ChatGPT gpt-3.5-turbo, DeepSeek-Coder-Instruct 33B, Qwen2.5-Coder-Instruct 32B). We sample about 20,000 functions and keep the raw files out of the repository.
   - Dataset: https://huggingface.co/datasets/OSS-forge/HumanVsAICode (license CC-BY-SA-4.0)
   - Paper: Cotroneo, Improta and Liguori. Human-Written vs. AI-Generated Code: A Large-Scale Study of Defects, Vulnerabilities, and Complexity. ISSRE 2025 (https://doi.org/10.1109/ISSRE66568.2025.00035)
   - Human code comes from CodeSearchNet through the HMCorp dataset (https://doi.org/10.1145/3705300)
2. H AIRosettaMP Replication Package. We reserved this dataset for a possible extension to whole file AI detection. We do not use it yet.
   - Archive: https://zenodo.org/records/13908858

The earlier pairwise plagiarism approach (ConPlag, IR Plag, and the datasets we evaluated and excluded) lives on the `main` branch and in the git history. See the `CITATION.md` of that branch for those references.

## Core Open Source Libraries

The project uses these open source libraries:

- `tree-sitter`, `tree-sitter-java`, `tree-sitter-python`, `tree-sitter-cpp`: Parse source code into concrete syntax trees.
- `numpy`: Compute the Markov matrices and run the trained network at inference time.
- `torch`, `scikit-learn`: Train and evaluate the CNN (the `train` dependency group; not installed in the app image).
- `fastapi`, `uvicorn`, `httpx`, `streamlit`, `matplotlib`: Provide the REST API, HTTP client, interactive dashboard, and heatmap plots.
- `pytest`, `ruff`, `uv`: Run automated tests, linting, and locked environment management.

## AI Assistance Disclosure

We followed the academic integrity policy of `TC3002B.570`. We used AI assistants (Google Antigravity and Gemini) for architecture planning, code implementation, test authoring, and documentation drafting. We computed all experimental metrics, cross validation splits, and benchmark tables through local execution on the cited datasets.
