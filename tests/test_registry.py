"""Unit tests for the multi model registry, invariant validation, and failure policies."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import joblib
import pytest

from src.inference import InvalidInputError, analyze_pair, load_model_bundle
from src.models.classifier import PairwisePlagiarismClassifier
from src.models.registry import ensure_manifest_entry, write_model_metadata
from src.vocabulary import NUM_STATES

REPO_MODELS = Path(__file__).resolve().parents[1] / "models"

JAVA_A = """
public class A {
    public static int f(int n) {
        int s = 0;
        for (int i = 0; i < n; i++) { s += i; }
        return s;
    }
}
"""
JAVA_B = """
public class B {
    public static int g(int m) {
        int t = 0;
        for (int j = 0; j < m; j++) { t += j; }
        return t;
    }
}
"""


def _copy_repo_models(tmp_path: Path) -> Path:
    dest = tmp_path / "models"
    shutil.copytree(REPO_MODELS, dest)
    return dest


def test_fail_closed_when_default_model_missing(tmp_path: Path) -> None:
    """Missing default model artifact forces degraded startup (model_loaded=False)."""
    model_dir = _copy_repo_models(tmp_path)
    (model_dir / "plagiarism_lgbm.joblib").unlink()
    bundle = load_model_bundle(model_dir)
    assert bundle.model_loaded is False
    assert bundle.models == {}


def test_optional_non_default_model_failure_degrades_gracefully(tmp_path: Path) -> None:
    """Missing optional non default model is omitted while default model keeps serving."""
    model_dir = _copy_repo_models(tmp_path)
    (model_dir / "cnn_weights.npz").unlink()
    bundle = load_model_bundle(model_dir)
    assert bundle.model_loaded is True
    assert bundle.default_model == "lgbm"
    assert list(bundle.models.keys()) == ["lgbm"]

    with pytest.raises(InvalidInputError, match="Unknown model"):
        analyze_pair(bundle, JAVA_A, JAVA_B, model_id="cnn")


def test_manifest_and_metadata_invariant_validation(tmp_path: Path) -> None:
    """Invalid num_states, thresholds, smoothing alphas, language lists, and unfitted LGBM fail closed."""
    bad_states = _copy_repo_models(tmp_path / "states")
    manifest = json.loads((bad_states / "manifest.json").read_text(encoding="utf-8"))
    manifest["num_states"] = NUM_STATES - 1
    (bad_states / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_model_bundle(bad_states).model_loaded is False

    bad_langs = _copy_repo_models(tmp_path / "langs")
    lgbm_meta = json.loads((bad_langs / "lgbm_metadata.json").read_text(encoding="utf-8"))
    lgbm_meta["supported_languages"] = "java"
    write_model_metadata(bad_langs / "lgbm_metadata.json", lgbm_meta)
    assert load_model_bundle(bad_langs).model_loaded is False

    bad_thr = _copy_repo_models(tmp_path / "thr")
    lgbm_meta = json.loads((bad_thr / "lgbm_metadata.json").read_text(encoding="utf-8"))
    lgbm_meta["threshold"] = 1.5
    write_model_metadata(bad_thr / "lgbm_metadata.json", lgbm_meta)
    assert load_model_bundle(bad_thr).model_loaded is False

    bad_alpha = _copy_repo_models(tmp_path / "alpha")
    lgbm_meta = json.loads((bad_alpha / "lgbm_metadata.json").read_text(encoding="utf-8"))
    lgbm_meta["smoothing_alpha"] = -0.1
    write_model_metadata(bad_alpha / "lgbm_metadata.json", lgbm_meta)
    assert load_model_bundle(bad_alpha).model_loaded is False

    unfitted = _copy_repo_models(tmp_path / "unfitted")
    joblib.dump(PairwisePlagiarismClassifier(), unfitted / "plagiarism_lgbm.joblib")
    assert load_model_bundle(unfitted).model_loaded is False


def test_ensure_manifest_entry_preserves_siblings_and_detects_state_conflict(
    tmp_path: Path,
) -> None:
    """ensure_manifest_entry preserves sibling models and rejects conflicting num_states."""
    ensure_manifest_entry(
        tmp_path,
        model_id="lgbm",
        metadata_filename="lgbm_metadata.json",
        num_states=NUM_STATES,
        default_model="lgbm",
    )
    ensure_manifest_entry(
        tmp_path,
        model_id="cnn",
        metadata_filename="cnn_metadata.json",
        num_states=NUM_STATES,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["default_model"] == "lgbm"
    assert manifest["models"] == {
        "lgbm": "lgbm_metadata.json",
        "cnn": "cnn_metadata.json",
    }

    with pytest.raises(ValueError, match="conflicts"):
        ensure_manifest_entry(
            tmp_path,
            model_id="cnn",
            metadata_filename="cnn_metadata.json",
            num_states=NUM_STATES + 1,
        )
