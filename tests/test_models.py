"""Unit tests for quantitative baselines (Frobenius, JSD, perplexity, TF-IDF)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.baseline import (
    TfidfCosineBaseline,
    frobenius_distance,
    jensen_shannon_divergence,
    transition_perplexity,
)


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
