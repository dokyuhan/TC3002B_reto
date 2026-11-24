"""Single file AI plagiarism scorer protocol and task isolated loader."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.models.registry import (
    DIAGNOSTIC_SMOOTHING_ALPHA,
    TASK_AI_PLAGIARISM,
    ModelSpec,
    load_model_spec,
)


@dataclass(frozen=True)
class SingleFileFeatures:
    """Extracted single file AST and normalized source features for AI plagiarism scoring."""

    states: np.ndarray
    norm_code: str
    num_states: int
    transition_matrix: np.ndarray
    perplexity: float
    diagnostic_smoothing_alpha: float = DIAGNOSTIC_SMOOTHING_ALPHA


class AiPlagiarismScorer(Protocol):
    """Protocol implemented by runtime single file AI plagiarism classifiers."""

    def predict_proba(self, features: SingleFileFeatures) -> float:
        """Return probability in `[0.0, 1.0]` that the submission is AI generated."""


@dataclass(frozen=True)
class LoadedAiPlagiarismModel:
    """Loaded single file AI plagiarism scorer composed with its validated `ModelSpec`."""

    spec: ModelSpec
    scorer: AiPlagiarismScorer

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


AiPlagiarismScorerLoader = Callable[[Path, ModelSpec], AiPlagiarismScorer]
_AI_PLAGIARISM_LOADERS: dict[str, AiPlagiarismScorerLoader] = {}


def register_ai_plagiarism_loader(kind: str, loader: AiPlagiarismScorerLoader) -> None:
    """Register a loader function for a single file `ai_plagiarism` model kind."""
    _AI_PLAGIARISM_LOADERS[str(kind).strip().lower()] = loader


def load_ai_plagiarism_scorer(model_dir: Path, spec: ModelSpec) -> AiPlagiarismScorer:
    """Load and validate an `ai_plagiarism` artifact through the registered kind loader."""
    if spec.task != TASK_AI_PLAGIARISM:
        raise ValueError(
            f"AI plagiarism loader rejected model '{spec.model_id}' for task '{spec.task}'."
        )

    artifact_path = model_dir / spec.artifact
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing artifact {artifact_path} for model '{spec.model_id}'")

    loader = _AI_PLAGIARISM_LOADERS.get(spec.kind)
    if loader is None:
        raise ValueError(
            f"Unsupported ai_plagiarism model kind '{spec.kind}' for model '{spec.model_id}'."
        )
    return loader(artifact_path, spec)


def load_registered_ai_plagiarism_model(
    model_dir: Path,
    model_id: str,
    metadata_filename: str,
    expected_num_states: int,
) -> LoadedAiPlagiarismModel:
    """Load metadata and artifact for one registered `ai_plagiarism` model."""
    metadata_path = model_dir / metadata_filename
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file {metadata_path} for model '{model_id}'")

    spec = load_model_spec(
        model_id=model_id,
        metadata_path=metadata_path,
        expected_task=TASK_AI_PLAGIARISM,
    )
    if spec.num_states != expected_num_states:
        raise ValueError(
            f"Model '{model_id}' num_states ({spec.num_states}) does not match "
            f"manifest num_states ({expected_num_states})"
        )

    scorer = load_ai_plagiarism_scorer(model_dir=model_dir, spec=spec)
    return LoadedAiPlagiarismModel(spec=spec, scorer=scorer)
