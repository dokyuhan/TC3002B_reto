"""Fit the shared character n-gram TF IDF baseline artifact (`models/tfidf_baseline.joblib`).

Run from the repository root:
    uv run python -m scripts.fit_tfidf
"""

from __future__ import annotations

from pathlib import Path

import joblib
import polars as pl

from scripts.train import DEFAULT_CONPLAG_V2_MANIFEST, DEFAULT_MODELS_DIR, fit_tfidf_baseline
from src.models.baseline import TfidfCosineBaseline
from src.preprocessing.normalizer import normalize_source

MANIFEST_PATH = DEFAULT_CONPLAG_V2_MANIFEST
OUTPUT_PATH = DEFAULT_MODELS_DIR / "tfidf_baseline.joblib"


def fit_tfidf(
    manifest_path: Path = MANIFEST_PATH,
    strip_boilerplate: bool = True,
) -> TfidfCosineBaseline:
    """Fit the TF IDF vectorizer on all normalized programs in the manifest."""
    df = pl.read_parquet(manifest_path)
    codes = df["code_a"].to_list() + df["code_b"].to_list()
    texts = [
        normalize_source(code, language="java", strip_boilerplate=strip_boilerplate)
        for code in codes
    ]
    return fit_tfidf_baseline(texts)


def main() -> None:
    """Fit and persist `models/tfidf_baseline.joblib`."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_tfidf(), OUTPUT_PATH, compress=3)
    print(f"Saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
