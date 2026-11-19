# Citations And Attributions

## Datasets

All course datasets were sourced from the official `TC3002B.570` dataset catalog (`data/raw/onedrive/url.txt`):

1. **ConPlag** (Primary training and cross validation dataset, `version_2` template stripped; `version_1` raw ablation):
   - Archive: https://zenodo.org/records/7332790
   - Paper: Towards a Dataset of Programming Contest Plagiarism in Java (https://arxiv.org/abs/2303.10763)
2. **IR Plag** (Held out zero shot obfuscation benchmark across Faidhi and Robinson levels `L1` through `L6`):
   - Repository: https://github.com/oscarkarnalim/sourcecodeplagiarismdataset
3. **PAN at FIRE 2014 (SOCO14)** (Evaluated and excluded due to extreme class imbalance and C syntax corruption):
   - Archive: https://pan.webis.de/fire14/pan14-web/#data
4. **AI SOCO (FIRE 2020)** (Evaluated and excluded; 1000 class authorship identification with missing train/test archives):
   - Repository: https://github.com/AliOsm/AI-SOCO
   - Paper: https://ceur-ws.org/Vol-2826/T2-1.pdf
5. **H AIRosettaMP Replication Package** (Quarantined for Phase 2 single file AI detection extension):
   - Archive: https://zenodo.org/records/13908858

## Core Open Source Libraries

- `tree-sitter`, `tree-sitter-java`, `tree-sitter-python`, `tree-sitter-cpp`: Concrete syntax tree parsing.
- `numpy`, `scipy`, `polars`: Numerical arrays, statistical divergences, and tabular manifests.
- `scikit-learn`, `lightgbm`, `joblib`: TF IDF vectorization, `StratifiedGroupKFold`, probability calibration, gradient boosted decision trees, and model serialization.
- `fastapi`, `uvicorn`, `httpx`, `streamlit`, `matplotlib`: REST API, HTTP client, interactive dashboard, and evaluation plots.
- `pytest`, `ruff`, `uv`: Automated testing, linting, and locked environment management.

## AI Assistance Disclosure

In accordance with `TC3002B.570` academic integrity policy, AI coding and documentation assistants (Google Antigravity / Gemini) were used during architecture planning, code implementation, test authoring, and documentation drafting. All experimental metrics, cross validation splits, and benchmark tables were computed directly from local execution on the cited datasets.
