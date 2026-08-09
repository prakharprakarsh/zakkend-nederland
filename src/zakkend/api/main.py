"""FastAPI service: /predict and /explain endpoints + demo frontend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from zakkend import config
from zakkend.explain.shap_explainer import SubsidenceExplainer
from zakkend.features.engineering import UnknownCategoryError
from zakkend.models.baseline import TrainedModel

# ───────────────────────────── Models in memory ─────────────────────────
_model: TrainedModel | None = None
_explainer: SubsidenceExplainer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model into memory on startup."""
    global _model, _explainer
    try:
        _model = TrainedModel.load()
    except FileNotFoundError as exc:
        raise RuntimeError("No trained model found. Run `python scripts/train.py` first.") from exc
    _explainer = SubsidenceExplainer(_model)
    yield


app = FastAPI(
    title="Zakkend Nederland",
    description="Foundation subsidence risk API for Dutch homes",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(UnknownCategoryError)
async def unknown_category_handler(request: Request, exc: UnknownCategoryError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────── Schemas ────────────────────────────────

# Literal types matching config.FOUNDATION_TYPES and config.SOIL_TYPES.
# FastAPI surfaces these as string enums in the OpenAPI schema.
# Keep in sync with both config constants if they change.
FoundationType = Literal["wooden_pile", "concrete_pile", "strip", "slab"]
SoilType = Literal["peat", "clay", "sandy_clay", "sand", "loess"]


class BuildingInput(BaseModel):
    """Feature inputs for a single building.

    Ranges reflect plausible Dutch data; the API validates conservatively.
    Extra fields are forbidden — building_age is derived server-side from year_built.
    """

    model_config = ConfigDict(extra="forbid")

    year_built: int = Field(..., ge=1700, le=2026)
    foundation_type: FoundationType = Field(..., examples=["wooden_pile"])
    soil_type: SoilType = Field(..., examples=["peat"])
    peat_thickness_m: float = Field(..., ge=0, le=20)
    groundwater_depth_m: float = Field(..., ge=0, le=10)
    groundwater_variability: float = Field(..., ge=0, le=1)
    insar_deformation_mm_yr: float = Field(..., ge=-10, le=30)
    drought_exposure_index: float = Field(..., ge=0, le=1)
    distance_to_water_m: int = Field(..., ge=0, le=50_000)
    neighborhood_damage_rate: float = Field(..., ge=0, le=1)


class ClassProbability(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    class_probabilities: list[ClassProbability]
    risk_score: float  # expected value — 0..1


class ContributionOut(BaseModel):
    feature: str
    value: str
    shap_value: float
    direction: str


class ExplanationResponse(PredictionResponse):
    base_value: float
    top_drivers: list[ContributionOut]


# ──────────────────────────────── Routes ────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


def _to_dataframe(inp: BuildingInput) -> pd.DataFrame:
    data = inp.model_dump()
    data["building_age"] = config.REFERENCE_YEAR - data["year_built"]
    return pd.DataFrame([data])


def _expected_risk_score(probs: list[float]) -> float:
    """Probability-weighted mean of ordinal class indices, normalized to 0..1."""
    n = len(config.RISK_CLASSES) - 1
    return float(sum(p * (i / n) for i, p in enumerate(probs)))


@app.post("/predict", response_model=PredictionResponse)
def predict(inp: BuildingInput) -> PredictionResponse:
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    df = _to_dataframe(inp)
    probs = _model.predict_proba(df)[0]
    cls_probs = [
        ClassProbability(class_name=n, probability=float(p))
        for n, p in zip(_model.class_names, probs, strict=False)
    ]
    return PredictionResponse(
        predicted_class=_model.class_names[int(probs.argmax())],
        class_probabilities=cls_probs,
        risk_score=_expected_risk_score(list(probs)),
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(inp: BuildingInput) -> ExplanationResponse:
    if _model is None or _explainer is None:
        raise HTTPException(503, "Model not loaded")
    df = _to_dataframe(inp)
    probs = _model.predict_proba(df)[0]
    expl = _explainer.explain(df)

    return ExplanationResponse(
        predicted_class=expl.predicted_class,
        class_probabilities=[
            ClassProbability(class_name=n, probability=float(p))
            for n, p in zip(_model.class_names, probs, strict=False)
        ],
        risk_score=_expected_risk_score(list(probs)),
        base_value=expl.base_value,
        top_drivers=[
            ContributionOut(
                feature=c.feature,
                value=str(c.value),
                shap_value=c.shap_value,
                direction=c.direction,
            )
            for c in expl.top_drivers(k=5)
        ],
    )
