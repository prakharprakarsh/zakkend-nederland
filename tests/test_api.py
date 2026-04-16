"""Integration tests for the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from zakkend import config
from zakkend.data.synthetic import generate
from zakkend.models.baseline import train


@pytest.fixture(scope="module", autouse=True)
def _ensure_model_exists():
    """Make sure a trained model is on disk before the API starts."""
    if not config.MODEL_PATH.exists():
        model = train(generate(n=2_000))
        model.save()


@pytest.fixture(scope="module")
def client():
    from zakkend.api.main import app

    with TestClient(app) as c:
        yield c


def _sample_input() -> dict:
    return {
        "year_built": 1920,
        "building_age": 106,
        "foundation_type": "wooden_pile",
        "soil_type": "peat",
        "peat_thickness_m": 2.5,
        "groundwater_depth_m": 1.3,
        "groundwater_variability": 0.4,
        "insar_deformation_mm_yr": 3.5,
        "drought_exposure_index": 0.6,
        "distance_to_water_m": 80,
        "neighborhood_damage_rate": 0.2,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json=_sample_input())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_class"] in config.RISK_CLASSES
    assert 0.0 <= body["risk_score"] <= 1.0
    assert len(body["class_probabilities"]) == len(config.RISK_CLASSES)


def test_explain(client):
    r = client.post("/explain", json=_sample_input())
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["top_drivers"]) == 5
    for d in body["top_drivers"]:
        assert d["direction"] in ("↑ risk", "↓ risk")


def test_predict_validates_ranges(client):
    bad = _sample_input() | {"year_built": 1500}  # out of allowed range
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_high_risk_profile_predicts_higher_than_low(client):
    low = _sample_input() | {
        "year_built": 2015,
        "building_age": 11,
        "foundation_type": "concrete_pile",
        "soil_type": "sand",
        "peat_thickness_m": 0.0,
        "groundwater_depth_m": 3.0,
        "insar_deformation_mm_yr": 0.2,
        "drought_exposure_index": 0.1,
        "neighborhood_damage_rate": 0.02,
    }
    high = _sample_input()  # 1920 wooden pile on peat

    r_low = client.post("/predict", json=low).json()
    r_high = client.post("/predict", json=high).json()
    assert r_high["risk_score"] > r_low["risk_score"]
