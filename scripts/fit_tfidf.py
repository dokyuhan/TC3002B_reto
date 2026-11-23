"""Fit the character n-gram TF-IDF baseline shown next to the CNN verdict in the UI.

The CNN itself is trained in `model_training/` and exported with `model_training/export_numpy.py`.
Run from the repository root:  uv run python -m scripts.fit_tfidf
"""

from __future__ import annotations

from pathlib import Path

import joblib
import polars as pl

from src.models.baseline import TfidfCosineBaseline
from src.preprocessing.normalizer import normalize_source

MANIFEST_PATH = Path("dataset/preprocessed/conplag_v2_manifest.parquet")
OUTPUT_PATH = Path("models/tfidf_baseline.joblib")


def fit_tfidf(manifest_path: Path = MANIFEST_PATH, strip_boilerplate: bool = True) -> TfidfCosineBaseline:
    """Fit the TF-IDF vectorizer on every normalized program of the ConPlag v2 manifest."""
    df = pl.read_parquet(manifest_path)
    codes = df["code_a"].to_list() + df["code_b"].to_list()
    texts = [normalize_source(c, language="java", strip_boilerplate=strip_boilerplate) for c in codes]
    baseline = TfidfCosineBaseline(
        sublinear_tf=True,
        ngram_range=(3, 5),
        analyzer="char_wb",
        max_features=20000,
    )
    return baseline.fit(texts)


def main() -> None:
    """Fit the baseline and save it to models/tfidf_baseline.joblib."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_tfidf(), OUTPUT_PATH, compress=3)
    print(f"Saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
