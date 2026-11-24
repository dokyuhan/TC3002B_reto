"""Single file AI authorship scorer protocol and task isolated loader."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.models.registry import (
    DIAGNOSTIC_SMOOTHING_ALPHA,
    TASK_AI_AUTHORSHIP,
    ModelSpec,
    load_model_spec,
)


@dataclass(frozen=True)
class SingleFileFeatures:
    """Extracted single file AST and normalized source features for AI authorship scoring."""

    states: np.ndarray
    norm_code: str
    num_states: int
    transition_matrix: np.ndarray
    perplexity: float
    diagnostic_smoothing_alpha: float = DIAGNOSTIC_SMOOTHING_ALPHA


class AuthorshipScorer(Protocol):
    """Protocol implemented by runtime single file AI authorship classifiers."""

    def predict_proba(self, features: SingleFileFeatures) -> float:
        """Return probability in `[0.0, 1.0]` that the submission is AI authored."""


@dataclass(frozen=True)
class LoadedAuthorshipModel:
    """Loaded single file authorship scorer composed with its validated `ModelSpec`."""

    spec: ModelSpec
    scorer: AuthorshipScorer

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


AuthorshipScorerLoader = Callable[[Path, ModelSpec], AuthorshipScorer]
_AUTHORSHIP_LOADERS: dict[str, AuthorshipScorerLoader] = {}


def register_authorship_loader(kind: str, loader: AuthorshipScorerLoader) -> None:
    """Register a loader function for a single file `ai_authorship` model kind."""
    _AUTHORSHIP_LOADERS[str(kind).strip().lower()] = loader


def load_authorship_scorer(model_dir: Path, spec: ModelSpec) -> AuthorshipScorer:
    """Load and validate an `ai_authorship` artifact through the registered kind loader."""
    if spec.task != TASK_AI_AUTHORSHIP:
        raise ValueError(
            f"AI authorship loader rejected model '{spec.model_id}' for task '{spec.task}'."
        )

    artifact_path = model_dir / spec.artifact
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing artifact {artifact_path} for model '{spec.model_id}'")

    loader = _AUTHORSHIP_LOADERS.get(spec.kind)
    if loader is None:
        raise ValueError(
            f"Unsupported ai_authorship model kind '{spec.kind}' for model '{spec.model_id}'."
        )
    return loader(artifact_path, spec)


def load_registered_authorship_model(
    model_dir: Path,
    model_id: str,
    metadata_filename: str,
    expected_num_states: int,
) -> LoadedAuthorshipModel:
    """Load metadata and artifact for one registered `ai_authorship` model."""
    metadata_path = model_dir / metadata_filename
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file {metadata_path} for model '{model_id}'")

    spec = load_model_spec(
        model_id=model_id,
        metadata_path=metadata_path,
        expected_task=TASK_AI_AUTHORSHIP,
    )
    if spec.num_states != expected_num_states:
        raise ValueError(
            f"Model '{model_id}' num_states ({spec.num_states}) does not match "
            f"manifest num_states ({expected_num_states})"
        )

    scorer = load_authorship_scorer(model_dir=model_dir, spec=spec)
    return LoadedAuthorshipModel(spec=spec, scorer=scorer)
