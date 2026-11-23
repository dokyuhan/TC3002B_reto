# Citations and Attributions

## Datasets

We obtained all course datasets from the official `TC3002B.570` dataset catalog (`data/raw/onedrive/url.txt`):

1. ConPlag. We use `version_2` with templates stripped for primary training and cross validation. We use raw `version_1` for ablation testing.
   - Archive: https://zenodo.org/records/7332790
   - Paper: Towards a Dataset of Programming Contest Plagiarism in Java (https://arxiv.org/abs/2303.10763)
2. IR Plag. We use this dataset as a held out benchmark across Faidhi and Robinson obfuscation levels `L1` through `L6`.
   - Repository: https://github.com/oscarkarnalim/sourcecodeplagiarismdataset
3. PAN at FIRE 2014 (SOCO14). We evaluated and excluded this dataset because of extreme class imbalance and C syntax corruption.
   - Archive: https://pan.webis.de/fire14/pan14-web/#data
4. AI SOCO (FIRE 2020). We evaluated and excluded this dataset because it targets authorship identification across 1000 classes and misses train and test archives.
   - Repository: https://github.com/AliOsm/AI-SOCO
   - Paper: https://ceur-ws.org/Vol-2826/T2-1.pdf
5. H AIRosettaMP Replication Package. We reserved this dataset for the Phase 2 extension for single file AI detection.
   - Archive: https://zenodo.org/records/13908858

## Core Open Source Libraries

The project uses these open source libraries:

- `tree-sitter`, `tree-sitter-java`, `tree-sitter-python`, `tree-sitter-cpp`: Parse source code into concrete syntax trees.
- `numpy`, `scipy`, `polars`: Compute numerical arrays, statistical divergences, and tabular manifests.
- `scikit-learn`, `joblib`: Run TF IDF vectorization and model serialization.
- `fastapi`, `uvicorn`, `httpx`, `streamlit`, `matplotlib`: Provide the REST API, HTTP client, interactive dashboard, and evaluation plots.
- `pytest`, `ruff`, `uv`: Run automated tests, linting, and locked environment management.

## AI Assistance Disclosure

We followed the academic integrity policy of `TC3002B.570`. We used AI assistants (Google Antigravity and Gemini) for architecture planning, code implementation, test authoring, and documentation drafting. We computed all experimental metrics, cross validation splits, and benchmark tables through local execution on the cited datasets.
