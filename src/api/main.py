"""FastAPI HTTP transport layer for the AI-code detector."""

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
    analyze_code,
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

    code: str
    language: Literal["java"] = "java"
    operating_mode: Literal["balanced", "low_fpr"] = "balanced"


class PredictResponse(BaseModel):
    """Response schema for POST /predict."""

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


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the trained model once on startup and clear references on shutdown."""
    application.state.bundle = load_model_bundle(resolve_model_dir())
    yield
    application.state.bundle = ModelBundle(model_loaded=False)


app = FastAPI(
    title="AI Code Detector API",
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
    """Estimate the probability that one function was written by an AI."""
    bundle = _get_bundle(request)
    try:
        analysis = analyze_code(
            bundle=bundle,
            code=payload.code,
            language=payload.language,
            operating_mode=payload.operating_mode,
        )
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(**asdict(analysis))
