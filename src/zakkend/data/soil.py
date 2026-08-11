"""Soil type classification from BRO Bodemkaart + coordinate-based fallback.

Primary source
--------------
The BRO Bodemkaart (SGM, 1:50 000) is published by PDOK as a bulk download only — no
WFS endpoint exists. It is available from the PDOK Atom feed:
  https://service.pdok.nl/tno/bro-bodemkaart/atom/bro-bodemkaart.xml
  GeoPackage: BRO_DownloadBodemkaart.gpkg  (~146 MB, EPSG:28992)

The GeoPackage is downloaded once into data/raw/ (gitignored) and the result of a
point-in-polygon spatial join against all processed buildings is cached in
data/raw/bro_soil_cache.parquet (committed, 20 KB). Tests and CI use the cache; the
GeoPackage is only needed to regenerate the cache or extend it to new buildings.

Coverage
--------
The BRO Bodemkaart at 1:50 000 does not map urban hardscape, water bodies, or
infrastructure zones. In the four target municipalities, 29.5% of buildings (1,128 of
3,828) fell inside a soil polygon; the remaining 70.5% fall back to the coordinate rule.
Both the soil type and the lookup source are recorded per building.

Coordinate fallback
-------------------
    - Western NL (lon < 5.0): peat/clay dominant (Holocene deposits)
    - River area (Betuwe, ~51.8–52.0 lat, 5.0–6.0 lon): river clay
    - Eastern NL (lon > 5.5): sand dominant (Pleistocene)
    - Southern Limburg: loess
    - Peat concentrated in Groene Hart, Friesland, NW-Overijssel

Usage
-----
    from zakkend.data.soil import enrich_with_soil
    df = enrich_with_soil(buildings_df)  # adds soil_type, peat_thickness_m, soil_source
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from zakkend import config

logger = logging.getLogger(__name__)

# Mapping from BRO GeoPackage mainsoilclassification (Dutch) to our five categories.
# Built from the soilarea → soilarea_soilunit → soil_units join in the GeoPackage.
_MAINSOIL_TO_SIMPLE: dict[str, str] = {
    "Veengronden": "peat",
    "Moerige gronden": "peat",
    "Zeekleigronden": "clay",
    "Rivierkleigronden": "clay",
    "Niet-gerijpte minerale gronden": "clay",
    "Oude rivierkleigronden": "clay",
    "Zeer oude mariene afzettingen": "clay",
    "Kalkloze zandgronden": "sand",
    "Kalkhoudende zandgronden": "sand",
    "Podzolgronden": "sand",
    "Dikke eerdgronden": "sand",
    "Leemgronden": "loess",
    "Brikgronden": "loess",
    "Kalksteen verweringsgronden": "loess",
    "Keileemgronden": "sandy_clay",
    "Zeer oude fluviatiele afzettingen": "sandy_clay",
    "Gedefinieerde associaties": "sandy_clay",
}

# Lazy singleton — loaded once on first call to lookup_bro_soil().
_BRO_CACHE: dict[tuple[float, float], str] | None = None


def _load_bro_cache() -> dict[tuple[float, float], str]:
    """Load bro_soil_cache.parquet into memory (lazy, cached after first call)."""
    global _BRO_CACHE
    if _BRO_CACHE is not None:
        return _BRO_CACHE

    cache_path = config.RAW_DATA_DIR / "bro_soil_cache.parquet"
    if not cache_path.exists():
        logger.warning(
            "BRO soil cache not found at %s — all buildings will use the coordinate rule. "
            "Run scripts/build_bro_cache.py to regenerate it from the GeoPackage.",
            cache_path,
        )
        _BRO_CACHE = {}
        return _BRO_CACHE

    df = pd.read_parquet(cache_path)
    _BRO_CACHE = {
        (float(row.lat5), float(row.lon5)): str(row.bro_soil_type)
        for row in df.itertuples(index=False)
    }
    logger.info("Loaded BRO soil cache: %d entries from %s", len(_BRO_CACHE), cache_path)
    return _BRO_CACHE


def lookup_bro_soil(lat: float, lon: float) -> str | None:
    """Return the BRO Bodemkaart soil type for a coordinate, or None on cache miss.

    A miss means the building centroid fell in an area the 1:50 000 soil map does
    not cover (urban hardscape, water, infrastructure). The caller should fall back
    to _classify_soil_by_coordinates in that case.
    """
    cache = _load_bro_cache()
    return cache.get((round(lat, 5), round(lon, 5)))


def _classify_soil_by_coordinates(lat: float, lon: float) -> str:
    """Rule-based soil classification from Dutch geological knowledge.

    Used as a fallback when the BRO cache has no entry for a point. Captures the
    macro patterns (west=peat/clay, east=sand, south=loess) that drive subsidence
    risk, but misclassifies cities whose centroids fall between rule boundaries
    (notably Dordrecht, which misses both the river-clay and peat-belt rules by
    0.08 deg longitude and 0.07 deg latitude respectively and falls through to the
    catch-all sandy_clay).
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

    # Catch-all — western transition zone not covered above
    return "sandy_clay"


def _bro_code_to_simple(code: str) -> str:
    """Map a BRO bodemsubgroep code to our five soil classes.

    Used when processing the GeoPackage directly (e.g. to rebuild the cache).
    The GeoPackage mainsoilclassification field is preferred; this function handles
    raw soilunit_code values for edge cases.

    BRO code conventions (Dutch soil classification system):
      V* = veen (peat) | M*, K*, R* = klei (clay) | Z*, H*, Y* = zand (sand) | L* = loess
    """
    code_upper = code.upper().strip()
    if not code_upper:
        return "sandy_clay"

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


def _estimate_peat_thickness(soil_type: str, lat: float, lon: float) -> float:
    """Estimate peat layer thickness in metres (TNO DINOloket typical values).

    Groene Hart: 2–8 m | Friesland: 1–4 m | River/other areas: absent (0.0).
    """
    if soil_type != "peat":
        return 0.0

    rng = np.random.default_rng(int(abs(lat * 1000) + abs(lon * 1000)))

    if 51.9 <= lat <= 52.3 and 4.4 <= lon <= 5.0:  # Groene Hart — thicker peat
        return float(np.clip(rng.gamma(3.0, 1.5), 1.0, 10.0))
    else:
        return float(np.clip(rng.gamma(2.0, 1.0), 0.5, 6.0))


def enrich_with_soil(df: pd.DataFrame) -> pd.DataFrame:
    """Add soil_type, peat_thickness_m, and soil_source columns to a buildings DataFrame.

    For each building the BRO cache is checked first (lookup_bro_soil). On a cache
    miss — which occurs when the centroid lies in an area the 1:50 000 soil map does
    not cover — the coordinate-based rule is used. The soil_source column records
    which path was taken ('BRO' or 'coord_rule') so the fallback rate is auditable.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lat' and 'lon' columns (WGS84 decimal degrees).

    Returns
    -------
    pd.DataFrame
        Input df with 'soil_type', 'peat_thickness_m', and 'soil_source' added.
    """
    result = df.copy()
    lat = result["lat"].values
    lon = result["lon"].values

    # ── BRO cache lookup (vectorised over keys) ───────────────────────────────
    cache = _load_bro_cache()
    bro_values = np.array(
        [
            cache.get((round(float(la), 5), round(float(lo), 5)))
            for la, lo in zip(lat, lon, strict=True)
        ],
        dtype=object,
    )
    bro_mask = bro_values != None  # noqa: E711  (object array, not using 'is not None')

    # ── Coordinate rule fallback (vectorised via np.select) ───────────────────
    conditions = [
        (lat < 51.0) & (lon > 5.7),
        lon < 4.4,
        (lat >= 51.9) & (lat <= 52.5) & (lon >= 4.4) & (lon <= 5.2),
        (lat >= 52.6) & (lat <= 53.2) & (lon >= 5.3) & (lon <= 6.2),
        (lat >= 52.4) & (lat <= 52.55) & (lon >= 4.7) & (lon <= 4.9),
        (lat >= 51.75) & (lat <= 52.05) & (lon >= 4.8) & (lon <= 6.0),
        lon < 5.0,
        lon > 5.5,
    ]
    choices = ["loess", "clay", "peat", "peat", "peat", "clay", "sandy_clay", "sand"]
    coord_values = np.select(conditions, choices, default="sandy_clay")

    # ── Merge: BRO where available, coord_rule otherwise ─────────────────────
    final_soil = np.where(bro_mask, bro_values, coord_values).astype(str)
    result["soil_type"] = final_soil
    result["soil_source"] = np.where(bro_mask, "BRO", "coord_rule")

    # peat_thickness_m is 0.0 for non-peat; only compute for peat rows
    peat_mask = final_soil == "peat"
    thicknesses = np.zeros(len(result))
    for i in np.where(peat_mask)[0]:
        thicknesses[i] = _estimate_peat_thickness("peat", float(lat[i]), float(lon[i]))
    result["peat_thickness_m"] = np.round(thicknesses, 2)

    bro_n = int(bro_mask.sum())
    coord_n = len(result) - bro_n
    logger.info(
        "Soil enrichment: %d BRO (%.1f%%), %d coord_rule (%.1f%%)",
        bro_n,
        bro_n / len(result) * 100,
        coord_n,
        coord_n / len(result) * 100,
    )
    logger.info("Soil distribution:\n%s", result["soil_type"].value_counts().to_string())
    return result
