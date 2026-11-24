"""Runtime inference facade: one Java function in, probability that an AI wrote it out."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np

from src.features.markov import sequence_to_transition_matrix
from src.models.cnn_numpy import NumpyCnnClassifier
from src.preprocessing.filters import declares_top_level_type, has_empty_body
from src.preprocessing.parser import parse_to_state_sequence
from src.vocabulary import DEFAULT_AST_STATES, NUM_STATES, STATE_TO_ID

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "ai_detector_weights.npz"
METADATA_FILE = "ai_detector_metadata.json"
MIN_AST_STATES = 5  # same minimum the training data used


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested without loaded model artifacts."""


class InvalidInputError(ValueError):
    """Raised when input source code or language fails domain validation."""


@dataclass(frozen=True)
class ModelBundle:
    """In-memory bundle of the loaded model and its calibrated metadata."""

    model_loaded: bool = False
    classifier: NumpyCnnClassifier | None = None
    model_version: str = "unknown"
    threshold: float = 0.5
    threshold_low_fpr: float = 0.5
    num_states: int = NUM_STATES
    supported_languages: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CodeAnalysis:
    """Structured result of analyzing one function."""

    ai_probability: float
    predicted_label: int
    threshold_used: float
    operating_mode: str
    ast_tokens: int
    parse_warnings: bool
    input_note: str | None
    state_names: list[str]
    state_share: list[float]
    transition_matrix: list[list[float]]


def resolve_model_dir() -> Path:
    """Resolve the model artifact directory from MODEL_DIR or repository defaults."""
    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    repo_models = Path(__file__).resolve().parents[1] / "models"
    if repo_models.exists():
        return repo_models

    container_models = Path("/app/models")
    if container_models.exists():
        return container_models

    return repo_models


def load_model_bundle(model_dir: Path) -> ModelBundle:
    """Load the CNN weights and metadata JSON without crashing startup."""
    weights_path = model_dir / WEIGHTS_FILE
    meta_path = model_dir / METADATA_FILE

    missing = [p.name for p in (weights_path, meta_path) if not p.is_file()]
    if missing:
        logger.warning(
            "Model artifacts missing in %s: %s. Starting in degraded mode.",
            model_dir,
            ", ".join(missing),
        )
        return ModelBundle(model_loaded=False)

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        classifier = NumpyCnnClassifier.load(weights_path)
    except Exception as exc:
        logger.warning("Failed to load model artifacts from %s: %s", model_dir, exc)
        return ModelBundle(model_loaded=False)

    language = str(metadata.get("language", "java")).strip().lower()
    return ModelBundle(
        model_loaded=True,
        classifier=classifier,
        model_version=str(metadata.get("model_version", "unknown")),
        threshold=float(metadata.get("threshold", 0.5)),
        threshold_low_fpr=float(metadata.get("threshold_low_fpr", 0.5)),
        num_states=int(metadata.get("num_states", NUM_STATES)),
        supported_languages=(language,),
    )


def analyze_code(
    bundle: ModelBundle,
    code: str,
    language: str = "java",
    operating_mode: Literal["balanced", "low_fpr"] = "balanced",
) -> CodeAnalysis:
    """Estimate the probability that one function was written by an AI, strictly in memory."""
    if not bundle.model_loaded or bundle.classifier is None:
        raise ModelNotLoadedError("Model artifacts not loaded")

    if not code.strip():
        raise InvalidInputError("The code must not be empty.")

    if language not in bundle.supported_languages:
        calibrated = ", ".join(bundle.supported_languages) or "none"
        raise InvalidInputError(
            f"Language '{language}' is not calibrated. Calibrated languages: {calibrated}."
        )

    if has_empty_body(code):
        raise InvalidInputError("The method body has no statements. Paste a method with code inside it.")

    note: str | None = None
    if declares_top_level_type(code):
        parsed_source = code
        note = (
            "This input declares its own class. The model was trained on single methods, "
            "so the result is less reliable. Paste just the method for a better estimate."
        )
    else:
        # The training data are bare methods wrapped in a class so tree-sitter sees a valid compilation unit.
        parsed_source = "class W {\n" + code + "\n}"

    result = parse_to_state_sequence(parsed_source, language=language, strip_boilerplate=False)
    if len(result.states) < MIN_AST_STATES:
        raise InvalidInputError(
            f"Insufficient valid AST states (found {len(result.states)}, need at least {MIN_AST_STATES}). "
            "Paste a complete function."
        )

    if STATE_TO_ID["FUNCTION"] not in result.states:
        raise InvalidInputError(
            "No method found. The model was trained on whole methods, so paste a complete method."
        )

    matrix = sequence_to_transition_matrix(result.states, num_states=bundle.num_states)
    probability = bundle.classifier.predict_one(matrix)
    threshold = bundle.threshold_low_fpr if operating_mode == "low_fpr" else bundle.threshold
    counts = np.bincount(np.asarray(result.states), minlength=bundle.num_states)

    return CodeAnalysis(
        ai_probability=probability,
        predicted_label=int(probability >= threshold),
        threshold_used=float(threshold),
        operating_mode=operating_mode,
        ast_tokens=int(result.token_count),
        parse_warnings=bool(result.has_error),
        input_note=note,
        state_names=list(DEFAULT_AST_STATES[: bundle.num_states]),
        state_share=[float(v) for v in (counts / counts.sum()).tolist()],
        transition_matrix=[[float(v) for v in row] for row in matrix.tolist()],
    )
