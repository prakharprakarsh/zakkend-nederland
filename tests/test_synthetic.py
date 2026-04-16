"""Tests for the synthetic data generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zakkend import config
from zakkend.data.synthetic import generate


def test_generate_shape_and_schema():
    df = generate(n=500)
    assert len(df) == 500
    required = set(config.FEATURE_COLUMNS) | {"risk_score", "risk_class", "lat", "lon"}
    assert required.issubset(df.columns)


def test_generate_is_reproducible():
    a = generate(n=200, seed=1)
    b = generate(n=200, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_risk_score_bounded_0_to_1():
    df = generate(n=2000)
    assert df["risk_score"].between(0, 1).all()


def test_all_risk_classes_represented():
    df = generate(n=10_000)
    classes = set(df["risk_class"].unique())
    # At least 3 of 4 classes should appear in a 10k sample
    assert len(classes & set(config.RISK_CLASSES)) >= 3


def test_peat_zone_buildings_are_riskier_on_average():
    """Domain sanity check: homes in the peat belt should average higher risk."""
    df = generate(n=5000)
    lat_in = (df["lat"] >= config.HIGH_PEAT_PROVINCE_LAT_RANGE[0]) & (
        df["lat"] <= config.HIGH_PEAT_PROVINCE_LAT_RANGE[1]
    )
    lon_in = (df["lon"] >= config.HIGH_PEAT_PROVINCE_LON_RANGE[0]) & (
        df["lon"] <= config.HIGH_PEAT_PROVINCE_LON_RANGE[1]
    )
    in_peat = lat_in & lon_in
    assert df.loc[in_peat, "risk_score"].mean() > df.loc[~in_peat, "risk_score"].mean()


def test_pre_1970_peat_homes_mostly_wooden_pile():
    """Historical pattern encoded in the generator."""
    df = generate(n=5000)
    mask = (df["year_built"] < 1970) & (df["soil_type"] == "peat")
    if mask.sum() > 50:
        wooden_share = (df.loc[mask, "foundation_type"] == "wooden_pile").mean()
        assert wooden_share > 0.50, f"Expected majority wooden pile, got {wooden_share:.2f}"


def test_latlon_within_netherlands_bounds():
    df = generate(n=1000)
    assert df["lat"].between(config.NL_LAT_MIN, config.NL_LAT_MAX).all()
    assert df["lon"].between(config.NL_LON_MIN, config.NL_LON_MAX).all()
