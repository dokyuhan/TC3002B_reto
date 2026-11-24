"""Tests for the numpy CNN classifier and its integration into the inference API."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pytest

from src.api.main import app
from src.features.markov import sequence_to_transition_matrix
from src.inference import analyze_pair, load_model_bundle
from src.models.baseline import TfidfCosineBaseline
from src.models.cnn_numpy import NumpyCnnClassifier, _adaptive_avg_pool, _conv3x3, _max_pool2
from src.models.registry import ensure_manifest_entry, write_model_metadata

REPO_MODELS = Path(__file__).resolve().parents[1] / "models"
K = 12

JAVA_A = """
public class SumLoop {
    public static int compute(int n) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) { total += i; }
        }
        return total;
    }
}
"""
JAVA_B = """
public class Accumulator {
    public static int run(int limit) {
        int acc = 0;
        int idx = 0;
        while (idx < limit) { acc = acc + idx; idx++; }
        return acc;
    }
}
"""


def _random_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random weights with the shapes of the trained network (3 -> 16 -> 32 -> 64 -> 1)."""
    rng = np.random.default_rng(seed)
    shapes = {
        "conv1_w": (16, 3, 3, 3), "conv1_b": (16,),
        "conv2_w": (32, 16, 3, 3), "conv2_b": (32,),
        "fc1_w": (64, 32 * 4 * 4), "fc1_b": (64,),
        "fc2_w": (1, 64), "fc2_b": (1,),
    }
    return {k: rng.normal(0.0, 0.1, size=s).astype(np.float32) for k, s in shapes.items()}


def _random_matrices(n: int, seed: int) -> np.ndarray:
    """Row-normalized random transition matrices of shape (n, K, K)."""
    return np.random.default_rng(seed).dirichlet(np.ones(K), size=(n, K)).astype(np.float32)


def test_conv_matches_naive_loops() -> None:
    """The vectorized 3x3 convolution equals a direct nested-loop implementation."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 5, 5)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    b = rng.normal(size=4).astype(np.float32)
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    expected = np.zeros((2, 4, 5, 5), dtype=np.float32)
    for n in range(2):
        for o in range(4):
            for i in range(5):
                for j in range(5):
                    expected[n, o, i, j] = (padded[n, :, i : i + 3, j : j + 3] * w[o]).sum() + b[o]
    assert np.allclose(_conv3x3(x, w, b), expected, atol=1e-5)


def test_pooling_shapes_and_values() -> None:
    """Max pool halves 12 -> 6; adaptive average pool maps 6 -> 4 with torch's overlapping windows."""
    x = np.arange(2 * 1 * 6 * 6, dtype=np.float32).reshape(2, 1, 6, 6)
    assert _max_pool2(np.zeros((1, 1, 12, 12), dtype=np.float32)).shape == (1, 1, 6, 6)
    pooled = _adaptive_avg_pool(x, 4)
    assert pooled.shape == (2, 1, 4, 4)
    # Window 0 covers rows/cols 0..1 (ceil(1.5) = 2); window 1 covers 1..2 (floor(1.5) = 1, ceil(3) = 3).
    assert pooled[0, 0, 0, 0] == pytest.approx(x[0, 0, 0:2, 0:2].mean())
    assert pooled[0, 0, 1, 1] == pytest.approx(x[0, 0, 1:3, 1:3].mean())


def test_probabilities_in_range_and_symmetric() -> None:
    """predict_pairs returns probabilities in [0, 1] and is exactly symmetric in (A, B)."""
    clf = NumpyCnnClassifier(_random_weights(), n_states=K)
    p_a, p_b = _random_matrices(8, 1), _random_matrices(8, 2)
    ab = clf.predict_pairs(p_a, p_b)
    ba = clf.predict_pairs(p_b, p_a)
    assert ab.shape == (8,)
    assert np.all((ab >= 0.0) & (ab <= 1.0))
    assert np.allclose(ab, ba, atol=1e-9)
    assert clf.predict_pair(p_a[0], p_b[0]) == pytest.approx(ab[0], abs=1e-6)


def test_load_round_trip_and_bad_shapes(tmp_path: Path) -> None:
    """Weights saved as .npz load back identically; wrong input shapes raise ValueError."""
    weights = _random_weights()
    path = tmp_path / "cnn_weights.npz"
    np.savez(path, **weights, n_states=np.int64(K), pool_size=np.int64(4))
    loaded = NumpyCnnClassifier.load(path)
    original = NumpyCnnClassifier(weights, n_states=K)
    p_a, p_b = _random_matrices(3, 3), _random_matrices(3, 4)
    assert np.allclose(loaded.predict_pairs(p_a, p_b), original.predict_pairs(p_a, p_b))
    with pytest.raises(ValueError, match="Expected"):
        loaded.predict_logits(np.zeros((1, 3, 5, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="Missing CNN weights"):
        NumpyCnnClassifier({"conv1_w": weights["conv1_w"]}, n_states=K)

    bad_fc1 = dict(weights)
    bad_fc1["fc1_w"] = np.zeros((64, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid fc1_w shape"):
        NumpyCnnClassifier(bad_fc1, n_states=K)

    bad_conv2 = dict(weights)
    bad_conv2["conv2_w"] = np.zeros((64, 99, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid conv2_w shape"):
        NumpyCnnClassifier(bad_conv2, n_states=K)

    incomplete_npz = tmp_path / "incomplete.npz"
    np.savez(incomplete_npz, conv1_w=weights["conv1_w"], n_states=np.int64(K), pool_size=np.int64(4))
    with pytest.raises(ValueError, match="Missing CNN weights"):
        NumpyCnnClassifier.load(incomplete_npz)


def test_empty_batch_and_extreme_logit_stability() -> None:
    """Empty batches return empty arrays and extreme negative logits avoid overflow warnings."""
    clf = NumpyCnnClassifier(_random_weights(), n_states=K)
    empty_x = np.empty((0, 3, K, K), dtype=np.float32)
    empty_p = np.empty((0, K, K), dtype=np.float32)
    assert clf.predict_logits(empty_x).shape == (0,)
    assert clf.predict_pairs(empty_p, empty_p).shape == (0,)

    with pytest.raises(ValueError, match="Expected matching"):
        clf.predict_pairs(
            np.empty((0, 5, 5), dtype=np.float32),
            np.empty((0, 5, 5), dtype=np.float32),
        )

    extreme = np.full((1, 3, K, K), 1e4, dtype=np.float32)
    probs = clf.predict_proba_tensor(extreme)
    assert probs.shape == (1,)
    assert np.isfinite(probs).all()
    assert 0.0 <= float(probs[0]) <= 1.0


@pytest.fixture()
def cnn_model_dir(tmp_path: Path) -> Path:
    """Model dir with random CNN weights and manifest to test the CNN loading path."""
    np.savez(
        tmp_path / "cnn_weights.npz",
        **_random_weights(),
        n_states=np.int64(K),
        pool_size=np.int64(4),
    )
    baseline = TfidfCosineBaseline(max_features=256)
    baseline.fit([JAVA_A, JAVA_B, "class Helper { int x = 1; }"])
    joblib.dump(baseline, tmp_path / "tfidf_baseline.joblib")
    meta: dict[str, object] = {
        "task": "pairwise_plagiarism",
        "kind": "cnn",
        "display_name": "Pairwise CNN",
        "artifact": "cnn_weights.npz",
        "model_version": "0.0.0-cnn-test",
        "threshold": 0.5,
        "threshold_low_fpr": 0.6,
        "smoothing_alpha": 0.0,
        "num_states": K,
        "strip_boilerplate": True,
        "supported_languages": ["java"],
    }
    write_model_metadata(tmp_path / "cnn_metadata.json", meta)
    ensure_manifest_entry(
        tmp_path,
        model_id="cnn",
        metadata_filename="cnn_metadata.json",
        num_states=K,
        default_model="cnn",
    )
    return tmp_path


def test_api_uses_cnn_when_weights_present(cnn_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With cnn_weights.npz present, /predict uses the CNN: valid, symmetric, thresholds respected."""
    monkeypatch.setenv("MODEL_DIR", str(cnn_model_dir))
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == "0.0.0-cnn-test"
        ab = client.post("/predict", json={"code_a": JAVA_A, "code_b": JAVA_B}).json()
        ba = client.post("/predict", json={"code_a": JAVA_B, "code_b": JAVA_A}).json()
        low = client.post(
            "/predict", json={"code_a": JAVA_A, "code_b": JAVA_B, "operating_mode": "low_fpr"}
        ).json()

    assert 0.0 <= ab["plagiarism_probability"] <= 1.0
    assert ab["plagiarism_probability"] == pytest.approx(ba["plagiarism_probability"], abs=1e-6)
    assert ab["threshold_used"] == pytest.approx(0.5)
    assert low["threshold_used"] == pytest.approx(0.6)
    assert ab["predicted_label"] == int(ab["plagiarism_probability"] >= 0.5)


def test_cnn_input_matches_training_features() -> None:
    """Inference builds unsmoothed matrices, so a state with no transitions stays an all-zero row."""
    seq = np.array([0, 3, 4, 3, 8], dtype=np.int32)
    matrix = sequence_to_transition_matrix(seq, num_states=K)
    assert matrix[11].sum() == 0.0  # state never visited: row of zeros, as in training data
    assert matrix[3].sum() == pytest.approx(1.0)


@pytest.mark.skipif(
    not (REPO_MODELS / "pairwise_plagiarism" / "cnn_weights.npz").is_file(),
    reason="trained CNN not exported",
)
def test_real_exported_model_serves_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed models/ directory loads both registered models and serves predictions."""
    bundle = load_model_bundle(REPO_MODELS)
    assert bundle.model_loaded is True
    assert "cnn" in bundle.models
    assert "lgbm" in bundle.models

    cnn_analysis = analyze_pair(bundle, JAVA_A, JAVA_B, model_id="cnn")
    assert cnn_analysis.model_used == "cnn"
    assert cnn_analysis.model_version.endswith("cnn")
    assert 0.0 <= cnn_analysis.plagiarism_probability <= 1.0
    assert cnn_analysis.threshold_used == pytest.approx(0.5)

    monkeypatch.setenv("MODEL_DIR", str(REPO_MODELS))
    with TestClient(app) as client:
        health = client.get("/health").json()
        body = client.post("/predict", json={"code_a": JAVA_A, "code_b": JAVA_B}).json()
    assert health["model_loaded"] is True
    assert 0.0 <= body["plagiarism_probability"] <= 1.0
