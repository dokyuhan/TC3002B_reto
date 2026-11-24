"""Unit and integration tests for the FastAPI inference service."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pytest

from src.api.main import app
from src.models.baseline import TfidfCosineBaseline
from src.models.registry import ensure_manifest_entry, write_model_metadata
from src.vocabulary import DEFAULT_AST_STATES, NUM_STATES

JAVA_CODE_A = """
public class SumLoop {
    public static int compute(int n) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                total += i;
            }
        }
        return total;
    }
}
"""

JAVA_CODE_B = """
public class Accumulator {
    public static int run(int limit) {
        int acc = 0;
        int idx = 0;
        while (idx < limit) {
            acc = acc + idx;
            idx++;
        }
        return acc;
    }
}
"""

JAVA_PARTIAL_SYNTAX_ERROR = """
public class BrokenSyntax {
    public static int compute(int n) {
        int total = 0
        for (int i = 0; i < n; i++) {
            total += i;
        }
        return total;
"""

EXPECTED_PREDICT_KEYS = {
    "model_used",
    "model_version",
    "plagiarism_probability",
    "predicted_label",
    "threshold_used",
    "markov_similarity_pct",
    "tfidf_cosine_score",
    "markov_jsd",
    "markov_frobenius",
    "perplexity_a",
    "perplexity_b",
    "ast_tokens_a",
    "ast_tokens_b",
    "parse_warnings_a",
    "parse_warnings_b",
    "state_names",
    "transition_delta",
}


@pytest.fixture()
def synthetic_model_dir(tmp_path: Path) -> Path:
    """Create random weight CNN, tiny TF IDF baseline, and manifest metadata in tmp_path."""
    rng = np.random.default_rng(42)
    shapes = {
        "conv1_w": (16, 3, 3, 3), "conv1_b": (16,),
        "conv2_w": (32, 16, 3, 3), "conv2_b": (32,),
        "fc1_w": (64, 32 * 4 * 4), "fc1_b": (64,),
        "fc2_w": (1, 64), "fc2_b": (1,),
    }
    weights = {k: rng.normal(0.0, 0.1, size=s).astype(np.float32) for k, s in shapes.items()}

    tfidf_baseline = TfidfCosineBaseline(max_features=256)
    tfidf_baseline.fit([JAVA_CODE_A, JAVA_CODE_B, "class Helper { int x = 1; }"])

    metadata: dict[str, object] = {
        "task": "pairwise_plagiarism",
        "kind": "cnn",
        "display_name": "Pairwise CNN",
        "artifact": "cnn_weights.npz",
        "model_version": "0.1.0-test",
        "threshold": 0.35,
        "threshold_low_fpr": 0.65,
        "smoothing_alpha": 0.0,
        "num_states": NUM_STATES,
        "strip_boilerplate": True,
        "supported_languages": ["java"],
    }

    np.savez(
        tmp_path / "cnn_weights.npz",
        **weights,
        n_states=np.int64(NUM_STATES),
        pool_size=np.int64(4),
    )
    joblib.dump(tfidf_baseline, tmp_path / "tfidf_baseline.joblib")
    write_model_metadata(tmp_path / "cnn_metadata.json", metadata)
    ensure_manifest_entry(
        tmp_path,
        model_id="cnn",
        metadata_filename="cnn_metadata.json",
        num_states=NUM_STATES,
        default_model="cnn",
    )
    return tmp_path


def test_health_with_loaded_models(
    synthetic_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /health reports healthy status and calibrated metadata when artifacts load."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    expected_descriptor = {
        "id": "cnn",
        "display_name": "Pairwise CNN",
        "model_version": "0.1.0-test",
        "supported_languages": ["java"],
        "threshold": 0.35,
        "threshold_low_fpr": 0.65,
    }
    assert body == {
        "status": "ok",
        "model_loaded": True,
        "supported_languages": ["java"],
        "model_version": "0.1.0-test",
        "default_model": "cnn",
        "available_models": [expected_descriptor],
        "tasks": {
            "pairwise_plagiarism": {
                "model_loaded": True,
                "default_model": "cnn",
                "model_version": "0.1.0-test",
                "supported_languages": ["java"],
                "available_models": [expected_descriptor],
            },
            "ai_authorship": {
                "model_loaded": False,
                "default_model": None,
                "model_version": "unknown",
                "supported_languages": [],
                "available_models": [],
            },
        },
    }


def test_predict_valid_java_pair_and_thresholds(
    synthetic_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /predict returns full schema, 12x12 delta, and respects operating_mode thresholds."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        resp_balanced = client.post(
            "/predict",
            json={
                "code_a": JAVA_CODE_A,
                "code_b": JAVA_CODE_B,
                "language": "java",
                "operating_mode": "balanced",
            },
        )
        resp_low_fpr = client.post(
            "/predict",
            json={
                "code_a": JAVA_CODE_A,
                "code_b": JAVA_CODE_B,
                "language": "java",
                "operating_mode": "low_fpr",
            },
        )

    assert resp_balanced.status_code == 200
    assert resp_low_fpr.status_code == 200

    data_bal = resp_balanced.json()
    data_low = resp_low_fpr.json()

    assert set(data_bal.keys()) == EXPECTED_PREDICT_KEYS
    assert isinstance(data_bal["plagiarism_probability"], float)
    assert 0.0 <= data_bal["plagiarism_probability"] <= 1.0
    assert isinstance(data_bal["predicted_label"], int)
    assert data_bal["predicted_label"] in (0, 1)
    assert data_bal["threshold_used"] == pytest.approx(0.35)
    assert data_low["threshold_used"] == pytest.approx(0.65)
    assert data_bal["predicted_label"] == int(
        data_bal["plagiarism_probability"] >= data_bal["threshold_used"]
    )
    assert data_low["predicted_label"] == int(
        data_low["plagiarism_probability"] >= data_low["threshold_used"]
    )

    assert 0.0 <= data_bal["markov_similarity_pct"] <= 100.0
    assert 0.0 <= data_bal["tfidf_cosine_score"] <= 1.0
    assert data_bal["markov_jsd"] >= 0.0
    assert data_bal["markov_frobenius"] >= 0.0
    assert data_bal["perplexity_a"] > 0.0
    assert data_bal["perplexity_b"] > 0.0
    assert data_bal["ast_tokens_a"] >= 2
    assert data_bal["ast_tokens_b"] >= 2
    assert data_bal["parse_warnings_a"] is False
    assert data_bal["parse_warnings_b"] is False

    assert data_bal["state_names"] == list(DEFAULT_AST_STATES)
    assert len(data_bal["transition_delta"]) == NUM_STATES
    assert all(len(row) == NUM_STATES for row in data_bal["transition_delta"])


def test_predict_permutation_symmetry(
    synthetic_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /predict is permutation symmetric across (A, B) and (B, A)."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        resp_ab = client.post(
            "/predict",
            json={"code_a": JAVA_CODE_A, "code_b": JAVA_CODE_B, "language": "java"},
        )
        resp_ba = client.post(
            "/predict",
            json={"code_a": JAVA_CODE_B, "code_b": JAVA_CODE_A, "language": "java"},
        )

    assert resp_ab.status_code == 200
    assert resp_ba.status_code == 200

    ab = resp_ab.json()
    ba = resp_ba.json()

    assert abs(ab["plagiarism_probability"] - ba["plagiarism_probability"]) < 1e-6
    assert ab["markov_jsd"] == pytest.approx(ba["markov_jsd"], abs=1e-9)
    assert ab["markov_frobenius"] == pytest.approx(ba["markov_frobenius"], abs=1e-9)
    assert ab["markov_similarity_pct"] == ba["markov_similarity_pct"]
    assert ab["perplexity_a"] == pytest.approx(ba["perplexity_b"], abs=1e-9)
    assert ab["perplexity_b"] == pytest.approx(ba["perplexity_a"], abs=1e-9)
    assert np.allclose(ab["transition_delta"], ba["transition_delta"], atol=1e-7)


def test_predict_partial_syntax_error_returns_200_with_warning(
    synthetic_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial syntax error with >= 2 recovered AST states returns 200 and flags parse_warnings."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "code_a": JAVA_PARTIAL_SYNTAX_ERROR,
                "code_b": JAVA_CODE_B,
                "language": "java",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["parse_warnings_a"] is True
    assert body["parse_warnings_b"] is False
    assert body["ast_tokens_a"] >= 2


@pytest.mark.parametrize(
    ("code_a", "code_b"),
    [
        ("   \n\t  ", JAVA_CODE_B),
        (JAVA_CODE_A, ""),
    ],
)
def test_predict_blank_input_returns_422(
    synthetic_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_a: str,
    code_b: str,
) -> None:
    """Blank or whitespace-only source code inputs return HTTP 422."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"code_a": code_a, "code_b": code_b, "language": "java"},
        )

    assert response.status_code == 422


def test_predict_too_few_ast_states_returns_422(
    synthetic_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inputs yielding fewer than 2 AST states return HTTP 422."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "code_a": "// comment only, no executable AST nodes",
                "code_b": JAVA_CODE_B,
                "language": "java",
            },
        )

    assert response.status_code == 422
    assert "Insufficient valid AST states" in response.json()["detail"]


def test_predict_uncalibrated_language_returns_422(
    synthetic_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uncalibrated language ('python' in Phase 1) returns HTTP 422 with clear detail."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "code_a": "def f(x):\n    return x + 1\n",
                "code_b": "def g(y):\n    return y + 2\n",
                "language": "python",
            },
        )

    assert response.status_code == 422
    assert "Phase 1" in response.json()["detail"]


def test_missing_model_artifacts_health_and_predict_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing model artifacts keep /health alive with model_loaded=False and return 503 on /predict."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(app) as client:
        health_resp = client.get("/health")
        predict_resp = client.post(
            "/predict",
            json={"code_a": JAVA_CODE_A, "code_b": JAVA_CODE_B, "language": "java"},
        )

    assert health_resp.status_code == 200
    health_body = health_resp.json()
    assert health_body["status"] == "ok"
    assert health_body["model_loaded"] is False
    assert health_body["supported_languages"] == []

    assert predict_resp.status_code == 503
    assert "Model artifacts not loaded" in predict_resp.json()["detail"]


def test_predict_multi_model_selection_and_invalid_model_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /health lists both models; POST /predict selects lgbm or cnn and rejects unknown IDs."""
    repo_models = Path(__file__).resolve().parents[1] / "models"
    monkeypatch.setenv("MODEL_DIR", str(repo_models))
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is True
        assert health["default_model"] == "lgbm"
        assert [item["id"] for item in health["available_models"]] == ["lgbm", "cnn"]

        default_resp = client.post(
            "/predict",
            json={"code_a": JAVA_CODE_A, "code_b": JAVA_CODE_B, "language": "java"},
        )
        lgbm_resp = client.post(
            "/predict",
            json={"code_a": JAVA_CODE_A, "code_b": JAVA_CODE_B, "language": "java", "model": "lgbm"},
        )
        cnn_resp = client.post(
            "/predict",
            json={"code_a": JAVA_CODE_A, "code_b": JAVA_CODE_B, "language": "java", "model": "cnn"},
        )
        invalid_resp = client.post(
            "/predict",
            json={
                "code_a": JAVA_CODE_A,
                "code_b": JAVA_CODE_B,
                "language": "java",
                "model": "unknown_model",
            },
        )

    assert default_resp.status_code == 200
    assert lgbm_resp.status_code == 200
    assert cnn_resp.status_code == 200
    assert invalid_resp.status_code == 422

    default_body = default_resp.json()
    lgbm_body = lgbm_resp.json()
    cnn_body = cnn_resp.json()
    assert default_body["model_used"] == "lgbm"
    assert lgbm_body["model_used"] == "lgbm"
    assert cnn_body["model_used"] == "cnn"
    assert default_body["plagiarism_probability"] == pytest.approx(
        lgbm_body["plagiarism_probability"], abs=1e-9
    )
    assert lgbm_body["transition_delta"] == cnn_body["transition_delta"]
