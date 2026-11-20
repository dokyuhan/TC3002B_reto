"""FastAPI HTTP transport layer for structural code plagiarism detection."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference import (
    InvalidInputError,
    ModelBundle,
    ModelNotLoadedError,
    analyze_pair,
    load_model_bundle,
    resolve_model_dir,
)


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    model_loaded: bool
    supported_languages: list[str]
    model_version: str


class PredictRequest(BaseModel):
    """Request schema for POST /predict."""

    code_a: str
    code_b: str
    language: Literal["java", "python", "cpp"] = "java"
    operating_mode: Literal["balanced", "low_fpr"] = "balanced"


class PredictResponse(BaseModel):
    """Response schema for POST /predict."""

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


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load trained artifacts once on startup and clear references on shutdown."""
    application.state.bundle = load_model_bundle(resolve_model_dir())
    yield
    application.state.bundle = ModelBundle(model_loaded=False)


app = FastAPI(
    title="Structural Code Plagiarism Detector API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_bundle(request: Request) -> ModelBundle:
    """Return initialized model bundle from FastAPI app.state."""
    bundle = getattr(request.app.state, "bundle", None)
    if isinstance(bundle, ModelBundle):
        return bundle
    return ModelBundle(model_loaded=False)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report service health, model readiness, and calibrated languages."""
    bundle = _get_bundle(request)
    return HealthResponse(
        status="ok",
        model_loaded=bundle.model_loaded,
        supported_languages=list(bundle.supported_languages),
        model_version=bundle.model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Evaluate pairwise structural plagiarism strictly in memory."""
    bundle = _get_bundle(request)
    try:
        analysis = analyze_pair(
            bundle=bundle,
            code_a=payload.code_a,
            code_b=payload.code_b,
            language=payload.language,
            operating_mode=payload.operating_mode,
        )
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(**asdict(analysis))
