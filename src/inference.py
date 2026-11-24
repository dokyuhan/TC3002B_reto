"""Cohesive runtime inference facade decoupling HTTP transport from ML internals."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Literal

import joblib

from src.features.markov import pair_to_tensor
from src.models.baseline import (
    TfidfCosineBaseline,
    frobenius_distance,
    jensen_shannon_divergence,
    transition_perplexity,
)
from src.models.registry import (
    DIAGNOSTIC_SMOOTHING_ALPHA,
    MANIFEST_FILENAME,
    LoadedModel,
    PairFeatures,
    load_registered_model,
)
from src.preprocessing.normalizer import normalize_source
from src.preprocessing.parser import parse_to_state_sequence
from src.vocabulary import DEFAULT_AST_STATES, NUM_STATES

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested without loaded model artifacts."""


class InvalidInputError(ValueError):
    """Raised when input source code, model selection, or language fails validation."""


@dataclass(frozen=True)
class ModelBundle:
    """In memory bundle of registered pairwise models and shared lexical baseline."""

    model_loaded: bool = False
    models: dict[str, LoadedModel] = field(default_factory=dict)
    default_model: str | None = None
    tfidf_baseline: TfidfCosineBaseline | None = None
    diagnostic_smoothing_alpha: float = DIAGNOSTIC_SMOOTHING_ALPHA
    num_states: int = NUM_STATES

    @property
    def active_model(self) -> LoadedModel | None:
        """Return the loaded default model, or None when degraded."""
        if self.default_model is None:
            return None
        return self.models.get(self.default_model)

    @property
    def model_version(self) -> str:
        """Return the default model version string, or 'unknown' when degraded."""
        active = self.active_model
        return active.model_version if active is not None else "unknown"

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Return calibrated languages for the default model, or empty tuple when degraded."""
        active = self.active_model
        return active.supported_languages if active is not None else ()


@dataclass(frozen=True)
class PairAnalysis:
    """Structured plagiarism analysis result for a source code pair."""

    model_used: str
    model_version: str
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
    """Load registered models, TF IDF baseline, and manifest without crashing startup."""
    manifest_path = model_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        logger.warning(
            "Model manifest missing in %s (%s). Starting in degraded mode.",
            model_dir,
            MANIFEST_FILENAME,
        )
        return ModelBundle(model_loaded=False)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a JSON object.")
        models_map = manifest.get("models")
        if not isinstance(models_map, dict) or not models_map:
            raise ValueError("Manifest 'models' must be a non empty mapping.")
        default_model = str(manifest.get("default_model", "")).strip()
        if not default_model or default_model not in models_map:
            raise ValueError(
                f"Manifest default_model '{default_model}' is missing from registered models."
            )
        num_states = int(manifest.get("num_states", NUM_STATES))
        diagnostic_alpha = float(
            manifest.get("diagnostic_smoothing_alpha", DIAGNOSTIC_SMOOTHING_ALPHA)
        )
        tfidf_artifact = str(manifest.get("tfidf_artifact", "tfidf_baseline.joblib"))
    except Exception as exc:
        logger.warning("Failed to parse manifest in %s: %s", model_dir, exc)
        return ModelBundle(model_loaded=False)

    tfidf_path = model_dir / tfidf_artifact
    if not tfidf_path.is_file():
        logger.warning("Missing TF IDF artifact %s. Starting in degraded mode.", tfidf_path)
        return ModelBundle(model_loaded=False)

    try:
        tfidf_baseline = joblib.load(tfidf_path)
    except Exception as exc:
        logger.warning("Failed to load TF IDF artifact from %s: %s", tfidf_path, exc)
        return ModelBundle(model_loaded=False)

    if not isinstance(tfidf_baseline, TfidfCosineBaseline):
        logger.warning("Loaded TF IDF artifact from %s is not TfidfCosineBaseline.", tfidf_path)
        return ModelBundle(model_loaded=False)

    loaded_models: dict[str, LoadedModel] = {}
    for model_id, metadata_filename in models_map.items():
        try:
            loaded = load_registered_model(
                model_dir=model_dir,
                model_id=str(model_id),
                metadata_filename=str(metadata_filename),
                expected_num_states=num_states,
            )
        except Exception as exc:
            if model_id == default_model:
                logger.warning(
                    "Default model '%s' failed to load from %s: %s. Failing closed.",
                    model_id,
                    model_dir,
                    exc,
                )
                return ModelBundle(model_loaded=False)
            logger.warning(
                "Optional model '%s' failed to load from %s: %s. Omitting from registry.",
                model_id,
                model_dir,
                exc,
            )
            continue
        loaded_models[loaded.model_id] = loaded

    if default_model not in loaded_models:
        return ModelBundle(model_loaded=False)

    return ModelBundle(
        model_loaded=True,
        models=loaded_models,
        default_model=default_model,
        tfidf_baseline=tfidf_baseline,
        diagnostic_smoothing_alpha=diagnostic_alpha,
        num_states=num_states,
    )


def analyze_pair(
    bundle: ModelBundle,
    code_a: str,
    code_b: str,
    language: str = "java",
    operating_mode: Literal["balanced", "low_fpr"] = "balanced",
    model_id: str | None = None,
) -> PairAnalysis:
    """Evaluate pairwise structural plagiarism strictly in memory."""
    if (
        not bundle.model_loaded
        or not bundle.models
        or bundle.default_model is None
        or bundle.tfidf_baseline is None
    ):
        raise ModelNotLoadedError("Model artifacts not loaded")

    selected_id = bundle.default_model if model_id is None else model_id
    loaded = bundle.models.get(selected_id)
    if loaded is None:
        available = ", ".join(bundle.models.keys()) or "none"
        raise InvalidInputError(
            f"Unknown model '{selected_id}'. Available models: {available}."
        )

    if not code_a.strip() or not code_b.strip():
        raise InvalidInputError("Both code_a and code_b must contain non-empty source code.")

    if language not in loaded.supported_languages:
        calibrated = ", ".join(loaded.supported_languages) or "none"
        raise InvalidInputError(
            f"Language '{language}' is not calibrated in Phase 1. "
            f"Calibrated languages: {calibrated}."
        )

    res_a = parse_to_state_sequence(
        code_a,
        language=language,
        strip_boilerplate=loaded.strip_boilerplate,
    )
    res_b = parse_to_state_sequence(
        code_b,
        language=language,
        strip_boilerplate=loaded.strip_boilerplate,
    )

    if len(res_a.states) < 2 or len(res_b.states) < 2:
        raise InvalidInputError("Insufficient valid AST states for Markov transition analysis")

    # Diagnostic heatmap and Markov distances remain model agnostic across classifiers.
    smoothed_tensor = pair_to_tensor(
        res_a.states,
        res_b.states,
        num_states=bundle.num_states,
        smoothing_alpha=bundle.diagnostic_smoothing_alpha,
    )
    p_a, p_b, delta = smoothed_tensor[0], smoothed_tensor[1], smoothed_tensor[2]

    perp_a = transition_perplexity(
        res_a.states,
        p_a,
        smoothing_alpha=bundle.diagnostic_smoothing_alpha,
    )
    perp_b = transition_perplexity(
        res_b.states,
        p_b,
        smoothing_alpha=bundle.diagnostic_smoothing_alpha,
    )

    norm_a = normalize_source(
        code_a,
        language=language,
        strip_boilerplate=loaded.strip_boilerplate,
    )
    norm_b = normalize_source(
        code_b,
        language=language,
        strip_boilerplate=loaded.strip_boilerplate,
    )

    features = PairFeatures(
        states_a=res_a.states,
        states_b=res_b.states,
        norm_a=norm_a,
        norm_b=norm_b,
        num_states=bundle.num_states,
        smoothed_tensor=smoothed_tensor,
        perp_a=float(perp_a),
        perp_b=float(perp_b),
        diagnostic_smoothing_alpha=bundle.diagnostic_smoothing_alpha,
    )
    plagiarism_probability = float(loaded.scorer.predict_pair_proba(features))

    markov_jsd = float(jensen_shannon_divergence(p_a, p_b))
    markov_frobenius = float(frobenius_distance(p_a, p_b))
    markov_similarity_pct = round(max(0.0, min(100.0, (1.0 - markov_jsd) * 100.0)), 2)
    tfidf_cosine_score = float(bundle.tfidf_baseline.score_pair(norm_a, norm_b))

    threshold_used = (
        loaded.threshold_low_fpr if operating_mode == "low_fpr" else loaded.threshold
    )
    predicted_label = int(plagiarism_probability >= threshold_used)

    return PairAnalysis(
        model_used=loaded.model_id,
        model_version=loaded.model_version,
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
