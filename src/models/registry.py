"""Pluggable pairwise plagiarism scorer adapters and split manifest metadata registry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from src.features.markov import pair_to_tensor, sequence_to_transition_matrix
from src.models.baseline import transition_perplexity
from src.models.classifier import PairwisePlagiarismClassifier
from src.models.cnn_numpy import NumpyCnnClassifier
from src.vocabulary import NUM_STATES

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 2
DIAGNOSTIC_SMOOTHING_ALPHA = 0.01


@dataclass(frozen=True)
class PairFeatures:
    """Extracted pairwise representations shared across pairwise plagiarism scorers."""

    states_a: np.ndarray
    states_b: np.ndarray
    norm_a: str
    norm_b: str
    num_states: int
    smoothed_tensor: np.ndarray
    perp_a: float
    perp_b: float
    diagnostic_smoothing_alpha: float = DIAGNOSTIC_SMOOTHING_ALPHA


class PairwiseScorer(Protocol):
    """Protocol implemented by runtime pairwise plagiarism classifiers."""

    def predict_pair_proba(self, features: PairFeatures) -> float:
        """Return positive class plagiarism probability in `[0.0, 1.0]`."""


class LgbmPairwiseScorer:
    """Adapter exposing `PairwisePlagiarismClassifier` through `PairwiseScorer`."""

    def __init__(
        self,
        classifier: PairwisePlagiarismClassifier,
        num_states: int,
        smoothing_alpha: float = DIAGNOSTIC_SMOOTHING_ALPHA,
    ) -> None:
        if not isinstance(classifier, PairwisePlagiarismClassifier):
            raise TypeError(
                f"Expected PairwisePlagiarismClassifier, got {type(classifier).__name__}"
            )
        if classifier._num_states is not None and classifier._num_states != num_states:
            raise ValueError(
                f"LightGBM num_states mismatch: artifact has {classifier._num_states}, "
                f"expected {num_states}"
            )
        self.classifier = classifier
        self.num_states = int(num_states)
        self.smoothing_alpha = float(smoothing_alpha)

    def predict_pair_proba(self, features: PairFeatures) -> float:
        """Score a code pair using smoothed `|P_A - P_B|` tensors and perplexities."""
        if abs(self.smoothing_alpha - features.diagnostic_smoothing_alpha) < 1e-12:
            tensor = features.smoothed_tensor
            perp_a = features.perp_a
            perp_b = features.perp_b
        else:
            tensor = pair_to_tensor(
                features.states_a,
                features.states_b,
                num_states=self.num_states,
                smoothing_alpha=self.smoothing_alpha,
            )
            perp_a = transition_perplexity(
                features.states_a,
                tensor[0],
                smoothing_alpha=self.smoothing_alpha,
            )
            perp_b = transition_perplexity(
                features.states_b,
                tensor[1],
                smoothing_alpha=self.smoothing_alpha,
            )

        proba = self.classifier.predict_proba(
            tensor[np.newaxis, ...],
            perplexities=(np.array([perp_a]), np.array([perp_b])),
        )
        return float(proba[0, 1])


class CnnPairwiseScorer:
    """Adapter exposing `NumpyCnnClassifier` through `PairwiseScorer`."""

    def __init__(
        self,
        classifier: NumpyCnnClassifier,
        num_states: int,
        smoothing_alpha: float = 0.0,
    ) -> None:
        if not isinstance(classifier, NumpyCnnClassifier):
            raise TypeError(f"Expected NumpyCnnClassifier, got {type(classifier).__name__}")
        if classifier.n_states != num_states:
            raise ValueError(
                f"CNN n_states mismatch: artifact has {classifier.n_states}, expected {num_states}"
            )
        self.classifier = classifier
        self.num_states = int(num_states)
        self.smoothing_alpha = float(smoothing_alpha)

    def predict_pair_proba(self, features: PairFeatures) -> float:
        """Score a code pair using model specific transition matrices and signed differences."""
        mat_a = sequence_to_transition_matrix(
            features.states_a,
            num_states=self.num_states,
            smoothing_alpha=self.smoothing_alpha,
        )
        mat_b = sequence_to_transition_matrix(
            features.states_b,
            num_states=self.num_states,
            smoothing_alpha=self.smoothing_alpha,
        )
        return float(self.classifier.predict_pair(mat_a, mat_b))


@dataclass(frozen=True)
class ModelSpec:
    """Validated per model metadata parsed from `models/<model_id>_metadata.json`."""

    model_id: str
    kind: str
    display_name: str
    artifact: str
    model_version: str
    threshold: float
    threshold_low_fpr: float
    smoothing_alpha: float
    num_states: int
    strip_boilerplate: bool
    supported_languages: tuple[str, ...]


@dataclass(frozen=True)
class LoadedModel:
    """Loaded pairwise scorer paired with its validated per model metadata."""

    model_id: str
    kind: str
    display_name: str
    model_version: str
    scorer: PairwiseScorer
    threshold: float
    threshold_low_fpr: float
    smoothing_alpha: float
    num_states: int
    strip_boilerplate: bool
    supported_languages: tuple[str, ...]


def load_model_spec(model_id: str, metadata_path: Path) -> ModelSpec:
    """Read and validate a per model metadata JSON file."""
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Metadata file {metadata_path} must contain a JSON object.")

    kind = str(raw["kind"]).strip().lower()
    artifact = str(raw["artifact"]).strip()
    if not artifact:
        raise ValueError(f"Metadata file {metadata_path} has an empty artifact path.")

    supported_raw = raw.get("supported_languages", ["java"])
    supported_languages = tuple(str(lang).strip().lower() for lang in supported_raw)

    return ModelSpec(
        model_id=model_id,
        kind=kind,
        display_name=str(raw.get("display_name", model_id)),
        artifact=artifact,
        model_version=str(raw.get("model_version", "unknown")),
        threshold=float(raw["threshold"]),
        threshold_low_fpr=float(raw["threshold_low_fpr"]),
        smoothing_alpha=float(raw.get("smoothing_alpha", 0.0 if kind == "cnn" else 0.01)),
        num_states=int(raw.get("num_states", NUM_STATES)),
        strip_boilerplate=bool(raw.get("strip_boilerplate", True)),
        supported_languages=supported_languages,
    )


def load_scorer(model_dir: Path, spec: ModelSpec) -> PairwiseScorer:
    """Load and validate a model artifact into the matching `PairwiseScorer` adapter."""
    artifact_path = model_dir / spec.artifact
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing artifact {artifact_path} for model '{spec.model_id}'")

    if spec.kind == "lgbm":
        classifier = joblib.load(artifact_path)
        return LgbmPairwiseScorer(
            classifier=classifier,
            num_states=spec.num_states,
            smoothing_alpha=spec.smoothing_alpha,
        )

    if spec.kind == "cnn":
        classifier = NumpyCnnClassifier.load(artifact_path)
        return CnnPairwiseScorer(
            classifier=classifier,
            num_states=spec.num_states,
            smoothing_alpha=spec.smoothing_alpha,
        )

    raise ValueError(f"Unsupported model kind '{spec.kind}' for model '{spec.model_id}'")


def load_registered_model(
    model_dir: Path,
    model_id: str,
    metadata_filename: str,
    expected_num_states: int,
) -> LoadedModel:
    """Load metadata and artifact for one registered model and verify state dimensions."""
    metadata_path = model_dir / metadata_filename
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file {metadata_path} for model '{model_id}'")

    spec = load_model_spec(model_id=model_id, metadata_path=metadata_path)
    if spec.num_states != expected_num_states:
        raise ValueError(
            f"Model '{model_id}' num_states ({spec.num_states}) does not match "
            f"manifest num_states ({expected_num_states})"
        )

    scorer = load_scorer(model_dir=model_dir, spec=spec)
    return LoadedModel(
        model_id=spec.model_id,
        kind=spec.kind,
        display_name=spec.display_name,
        model_version=spec.model_version,
        scorer=scorer,
        threshold=spec.threshold,
        threshold_low_fpr=spec.threshold_low_fpr,
        smoothing_alpha=spec.smoothing_alpha,
        num_states=spec.num_states,
        strip_boilerplate=spec.strip_boilerplate,
        supported_languages=spec.supported_languages,
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def write_model_metadata(metadata_path: Path, payload: dict[str, object]) -> None:
    """Atomically write an independent per model metadata JSON file."""
    _write_json_atomic(metadata_path, payload)


def ensure_manifest_entry(
    models_dir: Path,
    model_id: str,
    metadata_filename: str,
    *,
    num_states: int = NUM_STATES,
    default_model: str | None = None,
    tfidf_artifact: str = "tfidf_baseline.joblib",
    diagnostic_smoothing_alpha: float = DIAGNOSTIC_SMOOTHING_ALPHA,
) -> Path:
    """Register or update `model_id` in `models/manifest.json` without clobbering siblings."""
    manifest_path = models_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Manifest {manifest_path} must contain a JSON object.")
    else:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "default_model": default_model or model_id,
            "num_states": int(num_states),
            "diagnostic_smoothing_alpha": float(diagnostic_smoothing_alpha),
            "tfidf_artifact": tfidf_artifact,
            "models": {},
        }

    models_map = dict(manifest.get("models", {}))
    models_map[model_id] = metadata_filename
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["num_states"] = int(manifest.get("num_states", num_states))
    manifest["diagnostic_smoothing_alpha"] = float(
        manifest.get("diagnostic_smoothing_alpha", diagnostic_smoothing_alpha)
    )
    manifest["tfidf_artifact"] = str(manifest.get("tfidf_artifact", tfidf_artifact))
    if default_model is not None:
        manifest["default_model"] = default_model
    elif not manifest.get("default_model"):
        manifest["default_model"] = model_id
    manifest["models"] = models_map

    _write_json_atomic(manifest_path, manifest)
    return manifest_path
