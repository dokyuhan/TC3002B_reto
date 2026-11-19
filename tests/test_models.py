"""Unit tests for quantitative baselines and permutation-symmetric LightGBM classifier."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pytest

from src.models.baseline import (
    TfidfCosineBaseline,
    frobenius_distance,
    jensen_shannon_divergence,
    transition_perplexity,
)
from src.models.classifier import PairwisePlagiarismClassifier


def _make_synthetic_pairwise_dataset(
    n_pos: int = 30,
    n_neg: int = 30,
    num_states: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic (N, 3, K, K) Markov tensors, binary labels, and perplexities."""
    rng = np.random.default_rng(seed)
    tensors: list[np.ndarray] = []
    labels: list[int] = []
    perp_a: list[float] = []
    perp_b: list[float] = []

    for _ in range(n_pos):
        base = rng.dirichlet(np.ones(num_states), size=num_states)
        noise = rng.normal(0.0, 0.015, size=(num_states, num_states))
        p_b = np.clip(base + noise, 1e-4, None)
        p_b = p_b / p_b.sum(axis=1, keepdims=True)
        tensors.append(np.stack([base, p_b, np.abs(base - p_b)], axis=0).astype(np.float32))
        labels.append(1)
        pa = float(rng.uniform(2.0, 4.0))
        perp_a.append(pa)
        perp_b.append(pa + float(rng.uniform(-0.15, 0.15)))

    for _ in range(n_neg):
        p_a = rng.dirichlet(np.ones(num_states), size=num_states)
        p_b = rng.dirichlet(np.ones(num_states), size=num_states)
        tensors.append(np.stack([p_a, p_b, np.abs(p_a - p_b)], axis=0).astype(np.float32))
        labels.append(0)
        perp_a.append(float(rng.uniform(2.0, 5.0)))
        perp_b.append(float(rng.uniform(5.5, 9.0)))

    X = np.stack(tensors, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.uint8)
    return X, y, np.asarray(perp_a, dtype=np.float64), np.asarray(perp_b, dtype=np.float64)


def test_frobenius_distance_properties() -> None:
    p_a = np.eye(4, dtype=np.float32)
    p_b = np.full((4, 4), 0.25, dtype=np.float32)

    assert frobenius_distance(p_a, p_a) == pytest.approx(0.0, abs=1e-12)
    d_ab = frobenius_distance(p_a, p_b)
    d_ba = frobenius_distance(p_b, p_a)
    assert d_ab == pytest.approx(d_ba, abs=1e-12)
    assert d_ab > 0.0
    assert np.isfinite(d_ab)

    with pytest.raises(ValueError, match="shapes must match"):
        frobenius_distance(np.eye(3), np.eye(4))


def test_jensen_shannon_divergence_properties() -> None:
    p_a = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    p_b = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    zeros = np.zeros((4, 4), dtype=np.float32)

    assert jensen_shannon_divergence(p_a, p_a) == pytest.approx(0.0, abs=1e-12)
    assert jensen_shannon_divergence(zeros, zeros) == pytest.approx(0.0, abs=1e-12)

    jsd_ab = jensen_shannon_divergence(p_a, p_b)
    jsd_ba = jensen_shannon_divergence(p_b, p_a)
    assert jsd_ab == pytest.approx(jsd_ba, abs=1e-12)
    assert 0.0 <= jsd_ab <= 1.0
    assert jsd_ab == pytest.approx(0.5, abs=1e-6)
    assert np.isfinite(jsd_ab)


def test_transition_perplexity_normal_and_degenerate() -> None:
    k = 12
    p_det = np.full((k, k), 1e-4, dtype=np.float32)
    for i in range(k):
        p_det[i, (i + 1) % k] = 1.0
    p_det = p_det / p_det.sum(axis=1, keepdims=True)
    p_uni = np.full((k, k), 1.0 / k, dtype=np.float32)

    seq = [0, 1, 2, 3, 4, 5]
    perp_det = transition_perplexity(seq, p_det, smoothing_alpha=0.01)
    perp_uni = transition_perplexity(seq, p_uni, smoothing_alpha=0.01)
    assert np.isfinite(perp_det) and perp_det > 0.0
    assert np.isfinite(perp_uni) and perp_uni > 0.0
    assert perp_det < perp_uni
    assert perp_uni == pytest.approx(float(k), rel=1e-4)

    # Degenerate sequences (< 2 tokens) must return a finite positive float.
    for degen in ([], [3], [-1, -1, 2]):
        val = transition_perplexity(degen, p_det, smoothing_alpha=0.01)
        assert np.isfinite(val)
        assert val > 0.0


@pytest.mark.parametrize("sublinear_tf", [False, True])
def test_tfidf_cosine_baseline(sublinear_tf: bool) -> None:
    corpus = [
        "class Main { static int add(int a, int b) { return a + b; } }",
        "class Main { static int sum(int x, int y) { return x + y; } }",
        "class Graph { void bfs(int start) { while (start > 0) start--; } }",
    ]
    baseline = TfidfCosineBaseline(sublinear_tf=sublinear_tf, ngram_range=(3, 5), analyzer="char_wb")
    baseline.fit(corpus)

    same = baseline.score_pair(corpus[0], corpus[0])
    similar = baseline.score_pair(corpus[0], corpus[1])
    different = baseline.score_pair(corpus[0], corpus[2])
    empty = baseline.score_pair("", corpus[0])

    assert 0.0 <= same <= 1.0
    assert same == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= similar <= 1.0
    assert 0.0 <= different <= 1.0
    assert similar > different
    assert empty == pytest.approx(0.0, abs=1e-12)

    batch = baseline.score_pairs([corpus[0], corpus[0]], [corpus[1], corpus[2]])
    assert batch.shape == (2,)
    assert np.allclose(batch, [similar, different], atol=1e-6)


def test_pairwise_classifier_fit_symmetry_and_roundtrip(tmp_path: Path) -> None:
    X, y, perp_a, perp_b = _make_synthetic_pairwise_dataset(n_pos=30, n_neg=30, num_states=12)

    clf = PairwisePlagiarismClassifier(
        n_estimators=20,
        max_depth=3,
        num_leaves=7,
        min_child_samples=5,
        calibration_cv=3,
        random_state=42,
    )
    clf.fit(X, y, perplexities=(perp_a, perp_b))

    probs_ab = clf.predict_proba(X, perplexities=(perp_a, perp_b))
    X_swapped = X[:, [1, 0, 2], :, :]
    probs_ba = clf.predict_proba(X_swapped, perplexities=(perp_b, perp_a))

    assert probs_ab.shape == (X.shape[0], 2)
    assert np.all((probs_ab >= 0.0) & (probs_ab <= 1.0))
    assert np.allclose(probs_ab.sum(axis=1), 1.0, atol=1e-6)
    assert float(np.max(np.abs(probs_ab - probs_ba))) < 1e-6

    # Standalone call without perplexities must also be permutation-symmetric and valid.
    probs_no_perp_ab = clf.predict_proba(X)
    probs_no_perp_ba = clf.predict_proba(X_swapped)
    assert probs_no_perp_ab.shape == (X.shape[0], 2)
    assert np.all((probs_no_perp_ab >= 0.0) & (probs_no_perp_ab <= 1.0))
    assert float(np.max(np.abs(probs_no_perp_ab - probs_no_perp_ba))) < 1e-6

    # Joblib save/load roundtrip preserves exact predictions.
    model_path = tmp_path / "plagiarism_lgbm.joblib"
    joblib.dump(clf, model_path, compress=3)
    loaded: PairwisePlagiarismClassifier = joblib.load(model_path)
    probs_loaded = loaded.predict_proba(X, perplexities=(perp_a, perp_b))
    assert np.allclose(probs_ab, probs_loaded, atol=1e-12)
