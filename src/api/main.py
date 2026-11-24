"""FastAPI HTTP transport layer separating pairwise plagiarism and AI plagiarism endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import (
    AiPlagiarismBundle,
    InvalidInputError,
    ModelBundle,
    ModelNotLoadedError,
    ServiceBundle,
    analyze_ai_plagiarism,
    analyze_pair,
    load_service_bundle,
    resolve_model_dir,
)
from src.models.registry import TASK_AI_PLAGIARISM, TASK_PAIRWISE_PLAGIARISM


class ModelDescriptor(BaseModel):
    """Metadata describing one loaded classifier available for inference."""

    id: str
    display_name: str
    model_version: str
    supported_languages: list[str]
    threshold: float
    threshold_low_fpr: float


class TaskHealth(BaseModel):
    """Readiness and available model inventory for one inference task."""

    model_loaded: bool
    default_model: str | None = None
    model_version: str = "unknown"
    supported_languages: list[str] = Field(default_factory=list)
    available_models: list[ModelDescriptor] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    model_loaded: bool
    supported_languages: list[str]
    model_version: str
    default_model: str | None = None
    available_models: list[ModelDescriptor] = Field(default_factory=list)
    tasks: dict[str, TaskHealth] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    """Request schema for POST /predict/pairwise_plagiarism and legacy POST /predict."""

    code_a: str
    code_b: str
    language: Literal["java", "python", "cpp"] = "java"
    operating_mode: Literal["balanced", "low_fpr"] = "balanced"
    model: str | None = None


class PredictResponse(BaseModel):
    """Response schema for POST /predict/pairwise_plagiarism and legacy POST /predict."""

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


class AiPlagiarismPredictRequest(BaseModel):
    """Request schema for single file POST /predict/ai_plagiarism."""

    code: str
    language: Literal["java", "python", "cpp"] = "java"
    operating_mode: Literal["balanced", "low_fpr"] = "balanced"
    model: str | None = None


class AiPlagiarismPredictResponse(BaseModel):
    """Response schema for single file POST /predict/ai_plagiarism."""

    model_used: str
    model_version: str
    ai_probability: float
    predicted_label: int
    threshold_used: float
    perplexity: float
    ast_tokens: int
    parse_warnings: bool
    state_names: list[str]
    transition_matrix: list[list[float]]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load isolated task bundles once on startup and clear references on shutdown."""
    service = load_service_bundle(resolve_model_dir())
    application.state.service = service
    application.state.bundle = service.pairwise_plagiarism
    yield
    empty = ServiceBundle()
    application.state.service = empty
    application.state.bundle = empty.pairwise_plagiarism


app = FastAPI(
    title="Code Plagiarism Detector API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_service(request: Request) -> ServiceBundle:
    """Return initialized multi task service bundle from FastAPI app.state."""
    service = getattr(request.app.state, "service", None)
    if isinstance(service, ServiceBundle):
        return service
    bundle = getattr(request.app.state, "bundle", None)
    if isinstance(bundle, ModelBundle):
        return ServiceBundle(pairwise_plagiarism=bundle)
    return ServiceBundle()


def _build_task_health(bundle: ModelBundle | AiPlagiarismBundle) -> TaskHealth:
    descriptors = [
        ModelDescriptor(
            id=loaded.model_id,
            display_name=loaded.display_name,
            model_version=loaded.model_version,
            supported_languages=list(loaded.supported_languages),
            threshold=loaded.threshold,
            threshold_low_fpr=loaded.threshold_low_fpr,
        )
        for loaded in bundle.models.values()
    ]
    return TaskHealth(
        model_loaded=bundle.model_loaded,
        default_model=bundle.default_model if bundle.model_loaded else None,
        model_version=bundle.model_version,
        supported_languages=list(bundle.supported_languages),
        available_models=descriptors if bundle.model_loaded else [],
    )


def _predict_pairwise(payload: PredictRequest, request: Request) -> PredictResponse:
    bundle = _get_service(request).pairwise_plagiarism
    try:
        analysis = analyze_pair(
            bundle=bundle,
            code_a=payload.code_a,
            code_b=payload.code_b,
            language=payload.language,
            operating_mode=payload.operating_mode,
            model_id=payload.model,
        )
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(**asdict(analysis))


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report service health, per task readiness, and available model inventories."""
    service = _get_service(request)
    pairwise_health = _build_task_health(service.pairwise_plagiarism)
    ai_plagiarism_health = _build_task_health(service.ai_plagiarism)
    return HealthResponse(
        status="ok",
        model_loaded=pairwise_health.model_loaded,
        supported_languages=pairwise_health.supported_languages,
        model_version=pairwise_health.model_version,
        default_model=pairwise_health.default_model,
        available_models=pairwise_health.available_models,
        tasks={
            TASK_PAIRWISE_PLAGIARISM: pairwise_health,
            TASK_AI_PLAGIARISM: ai_plagiarism_health,
        },
    )


@app.post("/predict/pairwise_plagiarism", response_model=PredictResponse)
def predict_pairwise_plagiarism(payload: PredictRequest, request: Request) -> PredictResponse:
    """Evaluate pairwise structural plagiarism strictly in memory."""
    return _predict_pairwise(payload, request)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Backward compatible alias for POST /predict/pairwise_plagiarism."""
    return _predict_pairwise(payload, request)


@app.post("/predict/ai_plagiarism", response_model=AiPlagiarismPredictResponse)
def predict_ai_plagiarism(
    payload: AiPlagiarismPredictRequest,
    request: Request,
) -> AiPlagiarismPredictResponse:
    """Evaluate single file AI plagiarism strictly in memory."""
    bundle = _get_service(request).ai_plagiarism
    try:
        analysis = analyze_ai_plagiarism(
            bundle=bundle,
            code=payload.code,
            language=payload.language,
            operating_mode=payload.operating_mode,
            model_id=payload.model,
        )
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AiPlagiarismPredictResponse(**asdict(analysis))
