"""InSAR ground deformation data from BodemDalingsKaart / Sentinel-1.

BodemDalingsKaart (BDK) provides processed Sentinel-1 InSAR measurements
showing vertical ground motion in mm/year across the Netherlands.

Phase 2 approach:
    We use a coordinate-based estimator calibrated against published BDK
    values for known subsidence zones. This is more accurate than Phase 1's
    random generator because it encodes the spatial subsidence pattern.

Phase 3 will integrate actual BDK raster data or the NCG point dataset.

Published subsidence rates (from BDK 2.0 + TNO research):
    - Gouda city center:  3–8 mm/yr
    - Rotterdam-Zuid:      2–5 mm/yr
    - Zaanstad:            2–6 mm/yr
    - Dordrecht:           1–4 mm/yr
    - Eastern NL (sand):   0–0.5 mm/yr
    - Groningen (gas):     2–10 mm/yr (different mechanism — gas extraction)

Usage
-----
    from zakkend.data.insar import enrich_with_insar
    df = enrich_with_insar(buildings_df)  # adds insar_deformation_mm_yr
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from zakkend import config

logger = logging.getLogger(__name__)


def _estimate_insar_deformation(
    lat: float, lon: float, soil_type: str, year_built: int
) -> float:
    """Estimate InSAR vertical deformation rate (mm/yr, positive = subsidence).

    Calibrated against published BodemDalingsKaart values for known areas.
    Combines spatial pattern with building-level factors:
    - Peat areas subside faster
    - Older buildings on soft soil have had more cumulative settlement
    - Distance from known subsidence hotspots
    """
    rng = np.random.default_rng(int(abs(lat * 10000) + abs(lon * 10000)))

    # Base rate by region (from BDK published data)
    base_rate = 0.3  # default: stable ground

    # Gouda area — highest subsidence rates in NL
    if 51.98 <= lat <= 52.03 and 4.68 <= lon <= 4.73:
        base_rate = rng.normal(5.5, 1.5)
    # Rotterdam-Zuid
    elif 51.88 <= lat <= 51.93 and 4.45 <= lon <= 4.52:
        base_rate = rng.normal(3.5, 1.0)
    # Zaanstad
    elif 52.43 <= lat <= 52.50 and 4.78 <= lon <= 4.87:
        base_rate = rng.normal(4.0, 1.2)
    # Dordrecht
    elif 51.78 <= lat <= 51.83 and 4.62 <= lon <= 4.72:
        base_rate = rng.normal(2.5, 0.8)
    # Groene Hart (broad peat area)
    elif 51.9 <= lat <= 52.5 and 4.3 <= lon <= 5.2:
        base_rate = rng.normal(2.5, 1.0)
    # Friesland peat
    elif 52.6 <= lat <= 53.2 and 5.3 <= lon <= 6.2:
        base_rate = rng.normal(1.8, 0.8)
    # Amsterdam region
    elif 52.3 <= lat <= 52.45 and 4.8 <= lon <= 5.0:
        base_rate = rng.normal(1.5, 0.7)
    # Eastern NL sand — very stable
    elif lon > 5.5:
        base_rate = rng.normal(0.2, 0.15)
    # Coastal clay
    elif lon < 4.5:
        base_rate = rng.normal(0.8, 0.4)

    # Soil-type modifier
    soil_modifier = {
        "peat": 1.3,
        "clay": 0.9,
        "sandy_clay": 0.6,
        "sand": 0.3,
        "loess": 0.4,
    }.get(soil_type, 0.5)
    base_rate *= soil_modifier

    # Older buildings: slightly higher deformation (cumulative compaction)
    age_factor = 1.0 + max(0, (2026 - year_built) - 50) * 0.002
    base_rate *= age_factor

    # Add measurement noise (real InSAR has ~0.5 mm/yr noise floor)
    noise = rng.normal(0, 0.4)

    return float(max(0, base_rate + noise))


def enrich_with_insar(df: pd.DataFrame) -> pd.DataFrame:
    """Add InSAR deformation rate to a buildings DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lat', 'lon', 'soil_type', and either 'bouwjaar'
        or 'year_built'.

    Returns
    -------
    pd.DataFrame
        Input df with 'insar_deformation_mm_yr' column added.
    """
    result = df.copy()

    # Handle column name variants
    year_col = "year_built" if "year_built" in result.columns else "bouwjaar"

    deformations = [
        round(
            _estimate_insar_deformation(
                row["lat"], row["lon"], row["soil_type"], row[year_col]
            ),
            2,
        )
        for _, row in result.iterrows()
    ]

    result["insar_deformation_mm_yr"] = deformations

    logger.info(
        f"InSAR enrichment: {len(result)} buildings, "
        f"mean={np.mean(deformations):.2f} mm/yr, "
        f"max={np.max(deformations):.2f} mm/yr"
    )

    return result
