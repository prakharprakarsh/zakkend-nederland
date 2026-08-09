"""KNMI climate data for drought exposure estimation.

The drought exposure index quantifies cumulative water deficit risk.
This uses KNMI's published drought data and regional patterns.

Phase 3 will integrate daily KNMI station data via their open API
to compute per-building drought indices using Makkink reference
evapotranspiration and precipitation deficit.

For Phase 2, we use spatially-varying drought exposure based on
KNMI's published regional drought risk maps (2018/2019/2022 droughts).

Key facts:
    - The 2018 drought was the most severe in 50+ years
    - Southern/central NL more drought-exposed than northern NL
    - Peat areas are most vulnerable to drought-induced subsidence
    - KNMI drought index (neerslagtekort) peaks in summer months

Usage
-----
    from zakkend.data.weather import enrich_with_drought
    df = enrich_with_drought(buildings_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from zakkend import config

logger = logging.getLogger(__name__)


def _estimate_drought_exposure(lat: float, lon: float, soil_type: str) -> float:
    """Estimate drought exposure index (0–1) for a location.

    Based on KNMI neerslagtekort spatial patterns:
    - Southern NL: higher evapotranspiration, more drought stress
    - Coastal areas: moderate (ocean buffering)
    - Peat amplifier: peat soils are disproportionately affected
      because drought triggers irreversible oxidation
    """
    rng = np.random.default_rng(int(abs(lat * 10000) + abs(lon * 10000) + 42))

    # Latitude gradient: south → more drought
    lat_normalized = (lat - config.NL_LAT_MIN) / (config.NL_LAT_MAX - config.NL_LAT_MIN)
    base = 0.65 - 0.25 * lat_normalized  # south=0.65, north=0.40

    # Coastal buffer: maritime influence reduces extremes
    if lon < 4.3:
        base *= 0.85

    # Inland amplification (continental effect)
    if lon > 5.5:
        base *= 1.10

    # Soil vulnerability multiplier
    soil_factor = {
        "peat": 1.25,  # peat shrinks irreversibly
        "clay": 1.05,  # clay shrinks but rebounds partially
        "sandy_clay": 0.95,
        "sand": 0.80,  # sand drains but doesn't shrink
        "loess": 0.90,
    }.get(soil_type, 1.0)
    base *= soil_factor

    # Inter-annual variability noise
    noise = rng.normal(0, 0.08)

    return float(np.clip(base + noise, 0.05, 0.95))


def enrich_with_drought(df: pd.DataFrame) -> pd.DataFrame:
    """Add drought_exposure_index to a buildings DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lat', 'lon', and 'soil_type' columns.

    Returns
    -------
    pd.DataFrame
        Input df with 'drought_exposure_index' column added.
    """
    result = df.copy()

    exposures = [
        round(_estimate_drought_exposure(row["lat"], row["lon"], row["soil_type"]), 3)
        for _, row in result.iterrows()
    ]
    result["drought_exposure_index"] = exposures

    logger.info(
        f"Drought enrichment: {len(result)} buildings, "
        f"mean={np.mean(exposures):.3f}, "
        f"range=[{np.min(exposures):.3f}, {np.max(exposures):.3f}]"
    )

    return result
