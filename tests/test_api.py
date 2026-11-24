"""Unit and integration tests for the AI-code detector inference service."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.main import app
from src.vocabulary import DEFAULT_AST_STATES, NUM_STATES

REPO_MODELS = Path(__file__).resolve().parents[1] / "models"

JAVA_METHOD = """
public static int sumEven(int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
            total += i;
        }
    }
    return total;
}
"""

JAVA_OTHER_METHOD = """
protected String join(List<String> parts) {
    StringBuilder sb = new StringBuilder();
    for (String p : parts) {
        sb.append(p);
        sb.append("/");
    }
    return sb.toString();
}
"""

JAVA_FULL_CLASS = """
public class Summer {
    public static int sumEven(int n) {
        int total = 0;
        for (int i = 0; i < n; i++) { total += i; }
        return total;
    }
}
"""

JAVA_PARTIAL_SYNTAX_ERROR = """
public static int broken(int n) {
    int total = 0
    for (int i = 0; i < n; i++) {
        total += i;
    }
    return total;
"""

EXPECTED_PREDICT_KEYS = {
    "ai_probability",
    "predicted_label",
    "threshold_used",
    "operating_mode",
    "ast_tokens",
    "parse_warnings",
    "input_note",
    "state_names",
    "state_share",
    "transition_matrix",
}


def _random_weights(seed: int = 42) -> dict[str, np.ndarray]:
    """Random weights with the shapes of the trained network."""
    rng = np.random.default_rng(seed)
    shapes = {
        "conv1_w": (16, 1, 3, 3), "conv1_b": (16,),
        "conv2_w": (32, 16, 3, 3), "conv2_b": (32,),
        "fc1_w": (64, 32 * 4 * 4), "fc1_b": (64,),
        "fc2_w": (1, 64), "fc2_b": (1,),
    }
    return {k: rng.normal(0.0, 0.1, size=s).astype(np.float32) for k, s in shapes.items()}


@pytest.fixture()
def synthetic_model_dir(tmp_path: Path) -> Path:
    """Create a random-weight CNN and its metadata in tmp_path."""
    np.savez(
        tmp_path / "ai_detector_weights.npz",
        **_random_weights(),
        n_states=np.int64(NUM_STATES),
        pool_size=np.int64(4),
    )
    metadata = {
        "model_version": "0.0.0-test",
        "language": "java",
        "num_states": NUM_STATES,
        "threshold": 0.35,
        "threshold_low_fpr": 0.65,
    }
    (tmp_path / "ai_detector_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path


def test_health_with_loaded_model(synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /health reports healthy status and calibrated metadata when artifacts load."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "supported_languages": ["java"],
        "model_version": "0.0.0-test",
    }


def test_predict_schema_matrix_and_thresholds(
    synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /predict returns the full schema, a valid 12x12 matrix, and respects operating_mode thresholds."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        balanced = client.post("/predict", json={"code": JAVA_METHOD, "operating_mode": "balanced"})
        low_fpr = client.post("/predict", json={"code": JAVA_METHOD, "operating_mode": "low_fpr"})

    assert balanced.status_code == 200 and low_fpr.status_code == 200
    body, body_low = balanced.json(), low_fpr.json()

    assert set(body) == EXPECTED_PREDICT_KEYS
    assert 0.0 <= body["ai_probability"] <= 1.0
    assert body["threshold_used"] == pytest.approx(0.35)
    assert body_low["threshold_used"] == pytest.approx(0.65)
    assert body["ai_probability"] == pytest.approx(body_low["ai_probability"])
    assert body["predicted_label"] == int(body["ai_probability"] >= 0.35)
    assert body_low["predicted_label"] == int(body_low["ai_probability"] >= 0.65)
    assert body["operating_mode"] == "balanced" and body_low["operating_mode"] == "low_fpr"

    assert body["ast_tokens"] >= 5
    assert body["parse_warnings"] is False
    assert body["input_note"] is None
    assert body["state_names"] == list(DEFAULT_AST_STATES)
    assert sum(body["state_share"]) == pytest.approx(1.0)
    matrix = np.array(body["transition_matrix"])
    assert matrix.shape == (NUM_STATES, NUM_STATES)
    row_sums = matrix.sum(axis=1)
    assert np.all(np.isclose(row_sums, 1.0, atol=1e-5) | np.isclose(row_sums, 0.0, atol=1e-6))


def test_predict_is_deterministic_and_input_dependent(
    synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same code gives the same probability; different code gives a different one."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        first = client.post("/predict", json={"code": JAVA_METHOD}).json()
        again = client.post("/predict", json={"code": JAVA_METHOD}).json()
        other = client.post("/predict", json={"code": JAVA_OTHER_METHOD}).json()

    assert first["ai_probability"] == again["ai_probability"]
    assert first["ai_probability"] != other["ai_probability"]


def test_predict_comments_and_formatting_are_ignored(
    synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Comments and whitespace do not change the result (the model only sees the AST structure)."""
    commented = JAVA_METHOD.replace("int total = 0;", "// accumulate\n    int total   =   0;  /* start */")
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        plain = client.post("/predict", json={"code": JAVA_METHOD}).json()
        noisy = client.post("/predict", json={"code": commented}).json()

    assert plain["ai_probability"] == pytest.approx(noisy["ai_probability"])


def test_predict_full_class_adds_note(synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole class is accepted but flagged, because the model was trained on single methods."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post("/predict", json={"code": JAVA_FULL_CLASS})

    assert response.status_code == 200
    assert "single methods" in response.json()["input_note"]


def test_predict_partial_syntax_error_returns_200_with_warning(
    synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial syntax error with enough recovered AST states returns 200 and flags parse_warnings."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post("/predict", json={"code": JAVA_PARTIAL_SYNTAX_ERROR})

    assert response.status_code == 200
    assert response.json()["parse_warnings"] is True


@pytest.mark.parametrize("code", ["", "   \n\t  "])
def test_predict_blank_input_returns_422(
    synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    """Blank or whitespace-only source code returns HTTP 422."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post("/predict", json={"code": code})

    assert response.status_code == 422


def test_predict_too_few_ast_states_returns_422(
    synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inputs yielding fewer AST states than the model needs return HTTP 422."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post("/predict", json={"code": "// only a comment"})

    assert response.status_code == 422
    assert "Insufficient valid AST states" in response.json()["detail"]


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("int x = 1;", "No method found"),
        ("void f() { }", "no statements"),
        ("void f() {\n  // TODO\n}", "no statements"),
    ],
)
def test_predict_rejects_inputs_that_are_not_real_methods(
    synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch, code: str, message: str
) -> None:
    """Fields and empty-body stubs were excluded from training, so the API refuses them instead of guessing."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post("/predict", json={"code": code})

    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_predict_uncalibrated_language_returns_422(
    synthetic_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A language other than Java is rejected by request validation."""
    monkeypatch.setenv("MODEL_DIR", str(synthetic_model_dir))
    with TestClient(app) as client:
        response = client.post("/predict", json={"code": "def f(x):\n    return x + 1\n", "language": "python"})

    assert response.status_code == 422


def test_missing_model_artifacts_health_and_predict_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing model artifacts keep /health alive with model_loaded=False and return 503 on /predict."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(app) as client:
        health_resp = client.get("/health")
        predict_resp = client.post("/predict", json={"code": JAVA_METHOD})

    assert health_resp.status_code == 200
    assert health_resp.json()["model_loaded"] is False
    assert health_resp.json()["supported_languages"] == []
    assert predict_resp.status_code == 503
    assert "Model artifacts not loaded" in predict_resp.json()["detail"]


@pytest.mark.skipif(
    not (REPO_MODELS / "ai_detector_weights.npz").is_file(), reason="trained detector not exported"
)
def test_real_exported_model_serves_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed models/ directory loads the real detector and answers /predict."""
    monkeypatch.setenv("MODEL_DIR", str(REPO_MODELS))
    with TestClient(app) as client:
        health = client.get("/health").json()
        body = client.post("/predict", json={"code": JAVA_METHOD}).json()

    assert health["model_loaded"] is True
    assert health["model_version"].endswith("ai-detector")
    assert 0.0 <= body["ai_probability"] <= 1.0
    assert 0.0 < body["threshold_used"] < 1.0
