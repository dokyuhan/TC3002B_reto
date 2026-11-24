"""Task scoped metadata registry and pairwise plagiarism scorer adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Protocol

import joblib
import numpy as np

from src.features.markov import pair_to_tensor, sequence_to_transition_matrix
from src.models.baseline import transition_perplexity
from src.models.classifier import PairwisePlagiarismClassifier
from src.models.cnn_numpy import NumpyCnnClassifier
from src.vocabulary import NUM_STATES

TASK_PAIRWISE_PLAGIARISM = "pairwise_plagiarism"
TASK_AI_AUTHORSHIP = "ai_authorship"
SUPPORTED_TASKS: tuple[str, ...] = (TASK_PAIRWISE_PLAGIARISM, TASK_AI_AUTHORSHIP)

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 2
DIAGNOSTIC_SMOOTHING_ALPHA = 0.01


def validate_task_name(task: str) -> str:
    """Validate and normalize a task identifier against `SUPPORTED_TASKS`."""
    normalized = str(task).strip().lower()
    if normalized not in SUPPORTED_TASKS:
        supported = ", ".join(SUPPORTED_TASKS)
        raise ValueError(f"Unsupported task '{task}'. Supported tasks: {supported}.")
    return normalized


def _validate_non_negative_finite(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value}")
    return val


def _validate_probability_threshold(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val) or not (0.0 <= val <= 1.0):
        raise ValueError(f"{name} must be finite and within [0.0, 1.0], got {value}")
    return val


@dataclass(frozen=True)
class PairFeatures:
    """Extracted pairwise AST representations shared across pairwise plagiarism scorers."""

    states_a: np.ndarray
    states_b: np.ndarray
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
        if not classifier.is_fitted:
            raise ValueError("LightGBM classifier artifact is not fitted.")
        if classifier.num_states != num_states:
            raise ValueError(
                f"LightGBM num_states mismatch: artifact has {classifier.num_states}, "
                f"expected {num_states}"
            )
        self.classifier = classifier
        self.num_states = int(num_states)
        self.smoothing_alpha = _validate_non_negative_finite("smoothing_alpha", smoothing_alpha)

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
        self.smoothing_alpha = _validate_non_negative_finite("smoothing_alpha", smoothing_alpha)

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
    """Validated per model metadata parsed from `<task>/<model_id>_metadata.json`."""

    model_id: str
    task: str
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
    """Loaded pairwise scorer composed with its validated `ModelSpec`."""

    spec: ModelSpec
    scorer: PairwiseScorer

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def task(self) -> str:
        return self.spec.task

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def display_name(self) -> str:
        return self.spec.display_name

    @property
    def model_version(self) -> str:
        return self.spec.model_version

    @property
    def threshold(self) -> float:
        return self.spec.threshold

    @property
    def threshold_low_fpr(self) -> float:
        return self.spec.threshold_low_fpr

    @property
    def smoothing_alpha(self) -> float:
        return self.spec.smoothing_alpha

    @property
    def num_states(self) -> int:
        return self.spec.num_states

    @property
    def strip_boilerplate(self) -> bool:
        return self.spec.strip_boilerplate

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return self.spec.supported_languages


def load_model_spec(
    model_id: str,
    metadata_path: Path,
    *,
    expected_task: str | None = None,
) -> ModelSpec:
    """Read and validate a per model metadata JSON file."""
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Metadata file {metadata_path} must contain a JSON object.")

    raw_task = raw.get("task")
    if raw_task is None:
        raise ValueError(f"Metadata file {metadata_path} is missing required 'task' field.")
    task = validate_task_name(str(raw_task))
    if expected_task is not None and task != expected_task:
        raise ValueError(
            f"Model '{model_id}' task '{task}' does not match expected task '{expected_task}'."
        )

    kind = str(raw["kind"]).strip().lower()
    artifact = str(raw["artifact"]).strip()
    if not artifact:
        raise ValueError(f"Metadata file {metadata_path} has an empty artifact path.")

    supported_raw = raw.get("supported_languages", ["java"])
    if not isinstance(supported_raw, list) or not supported_raw:
        raise ValueError(
            f"Metadata file {metadata_path} 'supported_languages' must be a non empty list."
        )
    supported_languages: list[str] = []
    for lang in supported_raw:
        if not isinstance(lang, str) or not lang.strip():
            raise ValueError(
                f"Metadata file {metadata_path} contains invalid language entry: {lang!r}"
            )
        supported_languages.append(lang.strip().lower())

    threshold = _validate_probability_threshold("threshold", raw["threshold"])
    threshold_low_fpr = _validate_probability_threshold(
        "threshold_low_fpr", raw["threshold_low_fpr"]
    )
    smoothing_alpha = _validate_non_negative_finite(
        "smoothing_alpha",
        raw.get("smoothing_alpha", 0.0 if kind == "cnn" else 0.01),
    )
    num_states = int(raw.get("num_states", NUM_STATES))
    if num_states < 1:
        raise ValueError(f"num_states must be >= 1, got {num_states}")

    return ModelSpec(
        model_id=model_id,
        task=task,
        kind=kind,
        display_name=str(raw.get("display_name", model_id)),
        artifact=artifact,
        model_version=str(raw.get("model_version", "unknown")),
        threshold=threshold,
        threshold_low_fpr=threshold_low_fpr,
        smoothing_alpha=smoothing_alpha,
        num_states=num_states,
        strip_boilerplate=bool(raw.get("strip_boilerplate", True)),
        supported_languages=tuple(supported_languages),
    )


def load_scorer(model_dir: Path, spec: ModelSpec) -> PairwiseScorer:
    """Load and validate a pairwise artifact into the matching `PairwiseScorer` adapter."""
    if spec.task != TASK_PAIRWISE_PLAGIARISM:
        raise ValueError(
            f"Pairwise loader rejected model '{spec.model_id}' for task '{spec.task}'."
        )

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

    raise ValueError(f"Unsupported pairwise model kind '{spec.kind}' for model '{spec.model_id}'")


def load_registered_model(
    model_dir: Path,
    model_id: str,
    metadata_filename: str,
    expected_num_states: int,
) -> LoadedModel:
    """Load metadata and artifact for one registered `pairwise_plagiarism` model."""
    metadata_path = model_dir / metadata_filename
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file {metadata_path} for model '{model_id}'")

    spec = load_model_spec(
        model_id=model_id,
        metadata_path=metadata_path,
        expected_task=TASK_PAIRWISE_PLAGIARISM,
    )
    if spec.num_states != expected_num_states:
        raise ValueError(
            f"Model '{model_id}' num_states ({spec.num_states}) does not match "
            f"manifest num_states ({expected_num_states})"
        )

    scorer = load_scorer(model_dir=model_dir, spec=spec)
    return LoadedModel(spec=spec, scorer=scorer)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_model_metadata(metadata_path: Path, payload: dict[str, object]) -> None:
    """Atomically write an independent per model metadata JSON file."""
    _write_json_atomic(metadata_path, payload)


def ensure_manifest_entry(
    models_dir: Path,
    model_id: str,
    metadata_filename: str,
    *,
    task: str = TASK_PAIRWISE_PLAGIARISM,
    num_states: int = NUM_STATES,
    default_model: str | None = None,
    tfidf_artifact: str | None = "tfidf_baseline.joblib",
    diagnostic_smoothing_alpha: float = DIAGNOSTIC_SMOOTHING_ALPHA,
) -> Path:
    """Register or update `model_id` in a task scoped `manifest.json` without clobbering siblings."""
    validated_task = validate_task_name(task)
    manifest_path = models_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Manifest {manifest_path} must contain a JSON object.")
        existing_task = manifest.get("task")
        if existing_task is not None and validate_task_name(str(existing_task)) != validated_task:
            raise ValueError(
                f"Manifest task ({existing_task}) conflicts with requested ({validated_task})."
            )
        existing_states = manifest.get("num_states")
        if existing_states is not None and int(existing_states) != int(num_states):
            raise ValueError(
                f"Manifest num_states ({existing_states}) conflicts with requested ({num_states})."
            )
    else:
        manifest = {
            "task": validated_task,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "default_model": default_model or model_id,
            "num_states": int(num_states),
            "diagnostic_smoothing_alpha": _validate_non_negative_finite(
                "diagnostic_smoothing_alpha", diagnostic_smoothing_alpha
            ),
            "models": {},
        }
        if tfidf_artifact is not None:
            manifest["tfidf_artifact"] = tfidf_artifact

    models_map = dict(manifest.get("models", {}))
    models_map[model_id] = metadata_filename
    manifest["task"] = validated_task
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["num_states"] = int(num_states)
    manifest["diagnostic_smoothing_alpha"] = _validate_non_negative_finite(
        "diagnostic_smoothing_alpha",
        manifest.get("diagnostic_smoothing_alpha", diagnostic_smoothing_alpha),
    )
    if tfidf_artifact is not None and "tfidf_artifact" not in manifest:
        manifest["tfidf_artifact"] = tfidf_artifact
    if default_model is not None:
        manifest["default_model"] = default_model
    elif not manifest.get("default_model"):
        manifest["default_model"] = model_id
    manifest["models"] = models_map

    _write_json_atomic(manifest_path, manifest)
    return manifest_path
