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


def test_foundation_type_changes_prediction(client):
    """Guards against train/serve category-encoding skew — foundation_type.

    On XGBoost 2.x, .astype("category") on a 1-row frame collapsed every value
    to code 0, making all four results identical. On 3.x (name-based lookup) this
    works correctly; this test locks that in.
    """
    results = []
    for foundation in ["wooden_pile", "concrete_pile", "strip", "slab"]:
        payload = _sample_input() | {"foundation_type": foundation}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text
        probs = tuple(round(p["probability"], 6) for p in r.json()["class_probabilities"])
        results.append(probs)
    assert (
        len(set(results)) > 1
    ), "foundation_type has no effect on predictions — categorical encoding is broken"


def test_soil_type_changes_prediction(client):
    """Guards against train/serve category-encoding skew — soil_type."""
    results = []
    for soil in ["peat", "clay", "sandy_clay", "sand", "loess"]:
        payload = _sample_input() | {"soil_type": soil}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text
        probs = tuple(round(p["probability"], 6) for p in r.json()["class_probabilities"])
        results.append(probs)
    assert (
        len(set(results)) > 1
    ), "soil_type has no effect on predictions — categorical encoding is broken"


def test_invalid_foundation_type_returns_422(client):
    """An invalid foundation_type must be rejected with 422.

    Pydantic validates against the Literal type before the request reaches the
    model. UnknownCategoryError is kept as defence-in-depth for values that pass
    Pydantic (e.g. valid Literal but absent from a subset-trained model).
    """
    payload = _sample_input() | {"foundation_type": "banana"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"Expected 422 for invalid foundation_type, got {r.status_code}."


def test_invalid_soil_type_returns_422(client):
    """An invalid soil_type must be rejected with 422 via Pydantic Literal validation."""
    payload = _sample_input() | {"soil_type": "gravel"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"Expected 422 for invalid soil_type, got {r.status_code}."


def test_building_age_field_rejected(client):
    """building_age is no longer a user input — the API returns 422 if it is sent.

    Before §1.5 fix: year_built=2020 with building_age=300 returned 200 OK and
    used the contradictory user-supplied age. After the fix: extra fields are
    forbidden (ConfigDict extra='forbid'), and building_age is derived server-side
    from year_built via config.REFERENCE_YEAR.
    """
    contradictory = _sample_input() | {"year_built": 2020, "building_age": 300}
    r = client.post("/predict", json=contradictory)
    assert (
        r.status_code == 422
    ), f"Expected 422 when building_age is supplied (extra field), got {r.status_code}."


def test_high_risk_profile_predicts_higher_than_low(client):
    low = _sample_input() | {
        "year_built": 2015,
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
