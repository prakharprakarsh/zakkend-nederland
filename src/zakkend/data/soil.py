"""Soil type estimation from PDOK BRO + coordinate-based fallback.

The primary source is the BRO Bodemkaart (soil map) via PDOK WFS.
When the API is unavailable or for speed, a coordinate-based estimator
uses known Dutch geological boundaries to approximate soil type.

Geological basis for the fallback:
    - Western NL (roughly lon < 5.0): peat/clay dominant (Holocene deposits)
    - River area (Betuwe, ~51.8–52.0 lat, 5.0–6.0 lon): river clay
    - Eastern NL (lon > 5.5): sand dominant (Pleistocene)
    - Southern Limburg: loess
    - Peat is concentrated in the Groene Hart, Friesland, NW-Overijssel

Usage
-----
    from zakkend.data.soil import enrich_with_soil
    df = enrich_with_soil(buildings_df)  # adds soil_type, peat_thickness_m columns
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

from zakkend import config

logger = logging.getLogger(__name__)

_SOIL_WFS_TIMEOUT = 20


def _classify_soil_by_coordinates(lat: float, lon: float) -> str:
    """Rule-based soil classification from Dutch geological knowledge.

    This is a coarse approximation used when BRO WFS is unavailable.
    It correctly captures the macro patterns (west=peat/clay,
    east=sand, south=loess) that drive subsidence risk.
    """
    # Southern Limburg: loess
    if lat < 51.0 and lon > 5.7:
        return "loess"

    # Western Netherlands coastal — marine clay (before peat check!)
    if lon < 4.4:
        return "clay"

    # Groene Hart + peat belt (core subsidence zone)
    if 51.9 <= lat <= 52.5 and 4.4 <= lon <= 5.2:
        return "peat"

    # Friesland / NW-Overijssel peat
    if 52.6 <= lat <= 53.2 and 5.3 <= lon <= 6.2:
        return "peat"

    # Zaanstreek (north of Amsterdam) — heavy peat
    if 52.4 <= lat <= 52.55 and 4.7 <= lon <= 4.9:
        return "peat"

    # River clay (Betuwe / Rivierenland)
    if 51.75 <= lat <= 52.05 and 4.8 <= lon <= 6.0:
        return "clay"

    # Western transition zone
    if lon < 5.0:
        return "sandy_clay"

    # Eastern Netherlands — predominantly sand (Veluwe, Drenthe, Twente)
    if lon > 5.5:
        return "sand"

    # Default for transition zones
    return "sandy_clay"


def _estimate_peat_thickness(soil_type: str, lat: float, lon: float) -> float:
    """Estimate peat layer thickness in meters.

    Based on TNO DINOloket typical values:
    - Groene Hart: 2–8m peat layers
    - Friesland: 1–4m
    - River areas: thin or absent
    """
    if soil_type != "peat":
        return 0.0

    rng = np.random.default_rng(int(abs(lat * 1000) + abs(lon * 1000)))

    # Groene Hart has thicker peat
    if 51.9 <= lat <= 52.3 and 4.4 <= lon <= 5.0:
        return float(np.clip(rng.gamma(3.0, 1.5), 1.0, 10.0))
    else:
        return float(np.clip(rng.gamma(2.0, 1.0), 0.5, 6.0))


def try_pdok_bro_soil(lat: float, lon: float) -> str | None:
    """Attempt a point query on the PDOK BRO Bodemkaart WFS.

    Returns the simplified soil class or None if the API is unreachable.
    """
    # Buffer around point for the spatial query (approx 50m in degrees)
    buf = 0.0005
    bbox = f"{lat - buf},{lon - buf},{lat + buf},{lon + buf},EPSG:4326"

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "bodemkaart50000:bodemvlak",
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": bbox,
        "count": 1,
    }

    try:
        resp = requests.get(
            config.PDOK_BRO_SOIL_WFS, params=params, timeout=_SOIL_WFS_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        if not features:
            return None

        # Parse BRO soil code to our simplified schema
        props = features[0].get("properties", {})
        bodem_code = str(props.get("bodemsubgroep", props.get("code", "")))
        return _bro_code_to_simple(bodem_code)
    except Exception as e:
        logger.debug(f"BRO WFS query failed (using fallback): {e}")
        return None


def _bro_code_to_simple(code: str) -> str:
    """Map BRO bodemsubgroep codes to our 5 soil classes.

    BRO codes follow the Dutch soil classification system:
    - V* = veen (peat)
    - M*, K*, R* = klei (clay) variants
    - Z*, H*, Y* = zand (sand) variants
    - L* = leem/loess
    """
    code_upper = code.upper().strip()
    if not code_upper:
        return "sandy_clay"  # safe default

    first = code_upper[0]
    if first == "V" or "VEEN" in code_upper:
        return "peat"
    elif first in ("M", "K", "R") or "KLEI" in code_upper:
        return "clay"
    elif first == "L" or "LEEM" in code_upper or "LOSS" in code_upper:
        return "loess"
    elif first in ("Z", "H", "Y", "W") or "ZAND" in code_upper:
        return "sand"
    else:
        return "sandy_clay"


def enrich_with_soil(
    df: pd.DataFrame,
    use_api: bool = True,
    api_sample_rate: float = 0.05,
) -> pd.DataFrame:
    """Add soil_type and peat_thickness_m columns to a buildings DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lat' and 'lon' columns.
    use_api : bool
        Whether to attempt PDOK BRO queries (slower, more accurate).
    api_sample_rate : float
        Fraction of buildings to query via BRO API (others use coordinate
        fallback). Set to 1.0 for maximum accuracy, 0.0 for speed.

    Returns
    -------
    pd.DataFrame
        Input df with 'soil_type' and 'peat_thickness_m' columns added.
    """
    result = df.copy()
    n = len(result)
    soil_types = []
    peat_thicknesses = []

    # Determine which rows get API lookups
    rng = np.random.default_rng(config.RANDOM_STATE)
    api_mask = rng.random(n) < api_sample_rate if use_api else np.zeros(n, dtype=bool)

    api_hits = 0
    api_misses = 0

    for i, row in result.iterrows():
        lat, lon = row["lat"], row["lon"]
        soil = None

        if api_mask[i if isinstance(i, int) else 0]:
            soil = try_pdok_bro_soil(lat, lon)
            if soil:
                api_hits += 1
            else:
                api_misses += 1

        if soil is None:
            soil = _classify_soil_by_coordinates(lat, lon)

        soil_types.append(soil)
        peat_thicknesses.append(_estimate_peat_thickness(soil, lat, lon))

    result["soil_type"] = soil_types
    result["peat_thickness_m"] = [round(p, 2) for p in peat_thicknesses]

    logger.info(
        f"Soil enrichment complete: {n} buildings, "
        f"{api_hits} API hits, {api_misses} API misses, "
        f"{n - api_hits - api_misses} coordinate-based"
    )
    logger.info(f"Soil distribution:\n{result['soil_type'].value_counts().to_string()}")

    return result
