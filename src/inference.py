"""Cohesive runtime inference facade decoupling HTTP transport from ML internals."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from src.features.markov import pair_to_tensor
from src.models.baseline import (
    TfidfCosineBaseline,
    frobenius_distance,
    jensen_shannon_divergence,
    transition_perplexity,
)
from src.models.classifier import PairwisePlagiarismClassifier
from src.preprocessing.normalizer import normalize_source
from src.preprocessing.parser import parse_to_state_sequence
from src.preprocessing.vocabulary import DEFAULT_AST_STATES, NUM_STATES

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested without loaded model artifacts."""


class InvalidInputError(ValueError):
    """Raised when input source code or language fails domain validation."""


@dataclass(frozen=True)
class ModelBundle:
    """In-memory bundle of loaded model artifacts and calibrated metadata."""

    model_loaded: bool = False
    classifier: PairwisePlagiarismClassifier | None = None
    tfidf_baseline: TfidfCosineBaseline | None = None
    model_version: str = "unknown"
    threshold: float = 0.5
    threshold_low_fpr: float = 0.5
    smoothing_alpha: float = 0.01
    num_states: int = NUM_STATES
    strip_boilerplate: bool = True
    supported_languages: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PairAnalysis:
    """Structured plagiarism analysis result for a source code pair."""

    plagiarism_probability: float
    predicted_label: int
    threshold_used: float
    markov_similarity_pct: float
    tfidf_cosine_score: float
    markov_jsd: float
    markov_frobenius: float
    perplexity_a: float
    perplexity_b: float
    ast_tokens_a: int
    ast_tokens_b: int
    parse_warnings_a: bool
    parse_warnings_b: bool
    state_names: list[str]
    transition_delta: list[list[float]]


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
    """Load classifier, TF-IDF baseline, and metadata JSON without crashing startup."""
    clf_path = model_dir / "plagiarism_lgbm.joblib"
    tfidf_path = model_dir / "tfidf_baseline.joblib"
    meta_path = model_dir / "model_metadata.json"

    missing = [p.name for p in (clf_path, tfidf_path, meta_path) if not p.is_file()]
    if missing:
        logger.warning(
            "Model artifacts missing in %s: %s. Starting in degraded mode.",
            model_dir,
            ", ".join(missing),
        )
        return ModelBundle(model_loaded=False)

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        classifier = joblib.load(clf_path)
        tfidf_baseline = joblib.load(tfidf_path)
    except Exception as exc:
        logger.warning("Failed to load model artifacts from %s: %s", model_dir, exc)
        return ModelBundle(model_loaded=False)

    if not isinstance(classifier, PairwisePlagiarismClassifier) or not isinstance(
        tfidf_baseline, TfidfCosineBaseline
    ):
        logger.warning(
            "Loaded artifacts from %s do not match expected model classes.",
            model_dir,
        )
        return ModelBundle(model_loaded=False)

    supported_raw = metadata.get("supported_languages", ["java"])
    supported_languages = tuple(str(lang).strip().lower() for lang in supported_raw)

    return ModelBundle(
        model_loaded=True,
        classifier=classifier,
        tfidf_baseline=tfidf_baseline,
        model_version=str(metadata.get("model_version", "unknown")),
        threshold=float(metadata.get("threshold", 0.5)),
        threshold_low_fpr=float(metadata.get("threshold_low_fpr", 0.5)),
        smoothing_alpha=float(metadata.get("smoothing_alpha", 0.01)),
        num_states=int(metadata.get("num_states", NUM_STATES)),
        strip_boilerplate=bool(metadata.get("strip_boilerplate", True)),
        supported_languages=supported_languages,
    )


def analyze_pair(
    bundle: ModelBundle,
    code_a: str,
    code_b: str,
    language: str = "java",
    operating_mode: Literal["balanced", "low_fpr"] = "balanced",
) -> PairAnalysis:
    """Evaluate pairwise structural plagiarism strictly in memory."""
    if (
        not bundle.model_loaded
        or bundle.classifier is None
        or bundle.tfidf_baseline is None
    ):
        raise ModelNotLoadedError("Model artifacts not loaded")

    if not code_a.strip() or not code_b.strip():
        raise InvalidInputError("Both code_a and code_b must contain non-empty source code.")

    if language not in bundle.supported_languages:
        calibrated = ", ".join(bundle.supported_languages) or "none"
        raise InvalidInputError(
            f"Language '{language}' is not calibrated in Phase 1. "
            f"Calibrated languages: {calibrated}."
        )

    res_a = parse_to_state_sequence(
        code_a,
        language=language,
        strip_boilerplate=bundle.strip_boilerplate,
    )
    res_b = parse_to_state_sequence(
        code_b,
        language=language,
        strip_boilerplate=bundle.strip_boilerplate,
    )

    if len(res_a.states) < 2 or len(res_b.states) < 2:
        raise InvalidInputError("Insufficient valid AST states for Markov transition analysis")

    tensor = pair_to_tensor(
        res_a.states,
        res_b.states,
        num_states=bundle.num_states,
        smoothing_alpha=bundle.smoothing_alpha,
    )
    p_a, p_b, delta = tensor[0], tensor[1], tensor[2]

    perp_a = transition_perplexity(res_a.states, p_a, smoothing_alpha=bundle.smoothing_alpha)
    perp_b = transition_perplexity(res_b.states, p_b, smoothing_alpha=bundle.smoothing_alpha)

    proba = bundle.classifier.predict_proba(
        tensor[np.newaxis, ...],
        perplexities=(np.array([perp_a]), np.array([perp_b])),
    )
    plagiarism_probability = float(proba[0, 1])

    markov_jsd = float(jensen_shannon_divergence(p_a, p_b))
    markov_frobenius = float(frobenius_distance(p_a, p_b))
    markov_similarity_pct = round(max(0.0, min(100.0, (1.0 - markov_jsd) * 100.0)), 2)

    norm_a = normalize_source(code_a, language=language, strip_boilerplate=bundle.strip_boilerplate)
    norm_b = normalize_source(code_b, language=language, strip_boilerplate=bundle.strip_boilerplate)
    tfidf_cosine_score = float(bundle.tfidf_baseline.score_pair(norm_a, norm_b))

    threshold_used = (
        bundle.threshold_low_fpr if operating_mode == "low_fpr" else bundle.threshold
    )
    predicted_label = int(plagiarism_probability >= threshold_used)

    return PairAnalysis(
        plagiarism_probability=plagiarism_probability,
        predicted_label=predicted_label,
        threshold_used=float(threshold_used),
        markov_similarity_pct=markov_similarity_pct,
        tfidf_cosine_score=tfidf_cosine_score,
        markov_jsd=markov_jsd,
        markov_frobenius=markov_frobenius,
        perplexity_a=float(perp_a),
        perplexity_b=float(perp_b),
        ast_tokens_a=int(res_a.token_count),
        ast_tokens_b=int(res_b.token_count),
        parse_warnings_a=bool(res_a.has_error),
        parse_warnings_b=bool(res_b.has_error),
        state_names=list(DEFAULT_AST_STATES[: bundle.num_states]),
        transition_delta=[[float(val) for val in row] for row in delta.tolist()],
    )
