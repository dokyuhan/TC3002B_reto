"""Quantitative mathematical and lexical baselines for pairwise code similarity."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.markov import clean_sequence


class TfidfCosineBaseline:
    """Character or token n-gram TF-IDF cosine similarity baseline."""

    def __init__(
        self,
        sublinear_tf: bool = True,
        ngram_range: tuple[int, int] = (3, 5),
        analyzer: str = "char_wb",
        max_features: int | None = 50000,
    ) -> None:
        self.sublinear_tf = sublinear_tf
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: Sequence[str]) -> TfidfCosineBaseline:
        """Fit the TF-IDF vocabulary strictly on the provided training texts."""
        kwargs: dict[str, object] = {
            "sublinear_tf": self.sublinear_tf,
            "ngram_range": self.ngram_range,
            "analyzer": self.analyzer,
            "max_features": self.max_features,
            "norm": "l2",
            "dtype": np.float32,
        }
        if self.analyzer in ("char", "char_wb"):
            kwargs["token_pattern"] = None

        vectorizer = TfidfVectorizer(**kwargs)
        cleaned_texts = [t if t is not None else "" for t in texts]
        vectorizer.fit(cleaned_texts)
        self._vectorizer = vectorizer
        return self

    def score_pair(self, code_a: str, code_b: str) -> float:
        """Return cosine similarity in [0.0, 1.0] for a single code pair."""
        scores = self.score_pairs([code_a], [code_b])
        return float(scores[0])

    def score_pairs(
        self,
        codes_a: Sequence[str],
        codes_b: Sequence[str],
    ) -> np.ndarray:
        """Return 1D float64 array of cosine similarities in [0.0, 1.0] for code pairs."""
        if self._vectorizer is None:
            raise RuntimeError("TfidfCosineBaseline must be fitted before scoring.")
        if len(codes_a) != len(codes_b):
            raise ValueError(
                f"Mismatched sequence lengths: len(codes_a)={len(codes_a)}, "
                f"len(codes_b)={len(codes_b)}"
            )
        if len(codes_a) == 0:
            return np.empty((0,), dtype=np.float64)

        vec_a = self._vectorizer.transform([c if c is not None else "" for c in codes_a])
        vec_b = self._vectorizer.transform([c if c is not None else "" for c in codes_b])
        sims = np.asarray(vec_a.multiply(vec_b).sum(axis=1), dtype=np.float64).ravel()
        return np.clip(sims, 0.0, 1.0)


def frobenius_distance(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """Compute Frobenius norm ||P_A - P_B||_F between two transition matrices."""
    arr_a = np.asarray(p_a, dtype=np.float64)
    arr_b = np.asarray(p_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Matrix shapes must match: {arr_a.shape} vs {arr_b.shape}")
    return float(np.linalg.norm(arr_a - arr_b, ord="fro"))


def jensen_shannon_divergence(
    p_a: np.ndarray,
    p_b: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute symmetric mean row-wise base-2 Jensen-Shannon divergence in [0.0, 1.0]."""
    arr_a = np.asarray(p_a, dtype=np.float64)
    arr_b = np.asarray(p_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Matrix shapes must match: {arr_a.shape} vs {arr_b.shape}")
    if arr_a.size == 0:
        return 0.0

    mat_a = np.atleast_2d(np.clip(arr_a, 0.0, None))
    mat_b = np.atleast_2d(np.clip(arr_b, 0.0, None))

    row_sum_a = mat_a.sum(axis=-1, keepdims=True)
    row_sum_b = mat_b.sum(axis=-1, keepdims=True)
    both_zero = (row_sum_a <= 0.0) & (row_sum_b <= 0.0)

    # Add eps before row normalization so zero-mass entries remain finite in log2.
    norm_a = mat_a + eps
    norm_b = mat_b + eps
    norm_a = norm_a / norm_a.sum(axis=-1, keepdims=True)
    norm_b = norm_b / norm_b.sum(axis=-1, keepdims=True)

    mid = 0.5 * (norm_a + norm_b)
    kl_a = np.sum(norm_a * np.log2(norm_a / mid), axis=-1)
    kl_b = np.sum(norm_b * np.log2(norm_b / mid), axis=-1)
    row_jsd = 0.5 * (kl_a + kl_b)
    row_jsd = np.where(both_zero.ravel(), 0.0, row_jsd)

    return float(max(0.0, np.mean(row_jsd)))


def transition_perplexity(
    seq: Sequence[int] | np.ndarray,
    p_matrix: np.ndarray,
    smoothing_alpha: float = 0.01,
) -> float:
    """Compute transition perplexity exp(-mean log P(s_{t+1} | s_t)) of a state sequence."""
    mat = np.asarray(p_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        raise ValueError(f"p_matrix must be a non-empty square 2D array, got shape {mat.shape}")

    num_states = int(mat.shape[0])
    arr = clean_sequence(seq, num_states=num_states)
    if arr.size < 2:
        return float(num_states)

    floor = max(float(smoothing_alpha) / num_states, 1e-12) if smoothing_alpha > 0.0 else 1e-12
    probs = mat[arr[:-1], arr[1:]]
    probs = np.clip(probs, floor, 1.0)
    cross_entropy = -float(np.mean(np.log(probs)))
    return float(np.exp(cross_entropy))
