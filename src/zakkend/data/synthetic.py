"""Generate synthetic training data for Phase 1.

The synthetic generator encodes **real domain dynamics** from Dutch
subsidence research:

* Pre-1945 homes are predominantly on wooden pile foundations.
* Wooden piles rot when groundwater drops below the pile heads.
* Peat soils are concentrated in the western/northern Netherlands.
* Peat shrinks irreversibly in drought (paalrot + peat oxidation).
* InSAR deformation above ~2 mm/yr subsidence is a leading indicator.

Risk is derived from a weighted combination of these factors, so a model
trained on this data learns patterns that **transfer to real data**
once swapped in during Phase 2.

Usage
-----
    python -m zakkend.data.synthetic --n 10000 --out data/processed/training.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zakkend import config


def _in_peat_zone(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Boolean mask: is this point in the approximate Dutch peat belt?"""
    lat_ok = (lat >= config.HIGH_PEAT_PROVINCE_LAT_RANGE[0]) & (
        lat <= config.HIGH_PEAT_PROVINCE_LAT_RANGE[1]
    )
    lon_ok = (lon >= config.HIGH_PEAT_PROVINCE_LON_RANGE[0]) & (
        lon <= config.HIGH_PEAT_PROVINCE_LON_RANGE[1]
    )
    return lat_ok & lon_ok


def _sample_soil(rng: np.random.Generator, in_peat: np.ndarray) -> np.ndarray:
    """Soil sampling conditioned on peat-zone membership."""
    # Peat zone: heavy peat/clay. Outside: sand/loess dominant.
    n = len(in_peat)
    soil = np.empty(n, dtype=object)
    peat_probs = np.array([0.45, 0.30, 0.15, 0.08, 0.02])
    non_peat_probs = np.array([0.03, 0.20, 0.22, 0.45, 0.10])
    for i in range(n):
        probs = peat_probs if in_peat[i] else non_peat_probs
        soil[i] = rng.choice(config.SOIL_TYPES, p=probs)
    return soil


def _sample_foundation(
    rng: np.random.Generator, year_built: np.ndarray, soil: np.ndarray
) -> np.ndarray:
    """Foundation type conditioned on era + soil."""
    n = len(year_built)
    foundation = np.empty(n, dtype=object)
    for i in range(n):
        pre_1970 = year_built[i] < config.WOODEN_PILE_ERA_END
        soft_soil = soil[i] in ("peat", "clay", "sandy_clay")
        if pre_1970 and soft_soil:
            probs = np.array([0.70, 0.20, 0.08, 0.02])  # wooden-pile heavy
        elif pre_1970:
            probs = np.array([0.10, 0.20, 0.55, 0.15])  # strip-foundation heavy
        elif soft_soil:
            probs = np.array([0.05, 0.75, 0.15, 0.05])  # concrete piles modern
        else:
            probs = np.array([0.01, 0.25, 0.45, 0.29])
        foundation[i] = rng.choice(config.FOUNDATION_TYPES, p=probs)
    return foundation


def _compute_risk_score(df: pd.DataFrame) -> np.ndarray:
    """Deterministic(ish) risk score derived from features.

    Weights reflect published domain knowledge:
    - Wooden piles + dropping groundwater = paalrot (high risk)
    - Peat + drought = subsidence from oxidation
    - InSAR deformation is a direct observational signal
    """
    score = np.zeros(len(df))

    # Foundation risk
    score += np.where(df["foundation_type"] == "wooden_pile", 0.35, 0.0)
    score += np.where(df["foundation_type"] == "strip", 0.10, 0.0)

    # Soil risk
    score += np.where(df["soil_type"] == "peat", 0.25, 0.0)
    score += np.where(df["soil_type"] == "clay", 0.10, 0.0)

    # Groundwater depth — deeper means piles exposed to air (rot)
    score += np.clip((df["groundwater_depth_m"] - 0.8) * 0.15, 0, 0.20)

    # Variability — fluctuating tables accelerate rot
    score += np.clip(df["groundwater_variability"] * 0.15, 0, 0.15)

    # InSAR signal — direct evidence
    score += np.clip(df["insar_deformation_mm_yr"] * 0.04, 0, 0.25)

    # Drought exposure
    score += np.clip(df["drought_exposure_index"] * 0.10, 0, 0.10)

    # Peat thickness amplifier
    score += np.clip(df["peat_thickness_m"] * 0.03, 0, 0.15)

    # Age amplifier — older is more exposed cumulatively
    score += np.clip(df["building_age"] * 0.001, 0, 0.10)

    # Neighborhood spillover (spatial correlation proxy)
    score += df["neighborhood_damage_rate"] * 0.15

    # Small noise so the model has something genuine to learn
    rng = np.random.default_rng(config.RANDOM_STATE)
    score += rng.normal(0, 0.05, size=len(df))

    return np.clip(score, 0, 1)


def _score_to_class(scores: np.ndarray) -> np.ndarray:
    """Bucket continuous risk into 4 ordinal classes."""
    t = config.RISK_THRESHOLDS
    classes = np.full(len(scores), config.RISK_CLASSES[0], dtype=object)
    classes[scores >= t[0]] = config.RISK_CLASSES[1]
    classes[scores >= t[1]] = config.RISK_CLASSES[2]
    classes[scores >= t[2]] = config.RISK_CLASSES[3]
    return classes


def generate(
    n: int = 10_000,
    seed: int = config.RANDOM_STATE,
    *,
    lat_min: float = config.NL_LAT_MIN,
    lat_max: float = config.NL_LAT_MAX,
    lon_min: float = config.NL_LON_MIN,
    lon_max: float = config.NL_LON_MAX,
) -> pd.DataFrame:
    """Generate n synthetic buildings across the Netherlands.

    Parameters
    ----------
    n : int
        Number of synthetic buildings to generate.
    seed : int
        Reproducibility seed.
    lat_min : float
        Southern latitude bound (degrees). Defaults to NL-wide minimum.
    lat_max : float
        Northern latitude bound (degrees). Defaults to NL-wide maximum.
    lon_min : float
        Western longitude bound (degrees). Defaults to NL-wide minimum.
    lon_max : float
        Eastern longitude bound (degrees). Defaults to NL-wide maximum.

    Returns
    -------
    pd.DataFrame
        Feature matrix with `risk_score` (float) and `risk_class` (ordinal).
    """
    rng = np.random.default_rng(seed)

    # --- Location ---
    lat = rng.uniform(lat_min, lat_max, n)
    lon = rng.uniform(lon_min, lon_max, n)
    in_peat = _in_peat_zone(lat, lon)

    # --- Building attributes ---
    # Bimodal year: 1890s–1940s spike + post-war boom + new-builds.
    era_choice = rng.choice(3, size=n, p=[0.35, 0.45, 0.20])
    year_built = np.where(
        era_choice == 0,
        rng.integers(1880, 1945, size=n),
        np.where(
            era_choice == 1, rng.integers(1945, 1990, size=n), rng.integers(1990, 2025, size=n)
        ),
    )
    building_age = 2026 - year_built

    soil = _sample_soil(rng, in_peat)
    foundation = _sample_foundation(rng, year_built, soil)

    # --- Soil / hydrology ---
    peat_thickness = np.where(
        soil == "peat",
        rng.gamma(2.0, 1.2, size=n),  # heavy tail
        np.where(soil == "clay", rng.gamma(0.5, 0.3, size=n), np.zeros(n)),
    )
    groundwater_depth = np.where(
        in_peat,
        rng.normal(0.9, 0.3, n),  # shallow in peat areas
        rng.normal(1.8, 0.6, n),
    ).clip(0.1, 5.0)
    groundwater_variability = rng.beta(2, 5, size=n)  # skewed low

    # --- Remote sensing / climate ---
    # InSAR: subsidence (positive mm/yr) correlated with peat + shallow water
    base_insar = (
        0.5
        + 0.8 * (soil == "peat")
        + 0.4 * (soil == "clay")
        + 0.3 * (groundwater_depth < 1.0)
        + rng.normal(0, 0.5, n)
    ).clip(0, 8)
    insar = base_insar + rng.exponential(0.3, n)

    # Drought exposure: cumulative normalized index (0..1)
    drought = (
        0.3 + 0.2 * (lat < 52.3) + rng.beta(2, 4, n) * 0.5  # south slightly more drought-exposed
    ).clip(0, 1)

    # --- Spatial features ---
    distance_to_water = rng.exponential(500, n).clip(5, 5000)

    # Neighborhood damage rate: spatial autocorrelation via lat/lon hashing
    neigh_hash = np.floor(lat * 50) + np.floor(lon * 50)
    rate_lookup = {h: rng.beta(1, 9) for h in np.unique(neigh_hash)}
    neighborhood_rate = np.array([rate_lookup[h] for h in neigh_hash])

    df = pd.DataFrame(
        {
            "building_id": [f"NL-{i:07d}" for i in range(n)],
            "lat": lat.round(5),
            "lon": lon.round(5),
            "year_built": year_built.astype(int),
            "building_age": building_age.astype(int),
            "foundation_type": foundation,
            "soil_type": soil,
            "peat_thickness_m": peat_thickness.round(2),
            "groundwater_depth_m": groundwater_depth.round(2),
            "groundwater_variability": groundwater_variability.round(3),
            "insar_deformation_mm_yr": insar.round(2),
            "drought_exposure_index": drought.round(3),
            "distance_to_water_m": distance_to_water.round(0).astype(int),
            "neighborhood_damage_rate": neighborhood_rate.round(3),
        }
    )

    df["risk_score"] = _compute_risk_score(df).round(3)
    df["risk_class"] = _score_to_class(df["risk_score"].values)
    return df


def generate_for_municipality(
    municipality: str,
    n: int = 2_500,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Generate synthetic buildings clipped to a target municipality bounding box.

    Parameters
    ----------
    municipality : str
        Name of the municipality. Must be a key in `config.TARGET_MUNICIPALITIES`.
    n : int
        Number of synthetic buildings to generate.
    seed : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Feature matrix with `risk_score`, `risk_class`, and a `municipality` column
        set to the provided municipality name.

    Raises
    ------
    ValueError
        If `municipality` is not in `config.TARGET_MUNICIPALITIES`.
    """
    if municipality not in config.TARGET_MUNICIPALITIES:
        raise ValueError(
            f"Unknown municipality {municipality!r}. "
            f"Must be one of: {sorted(config.TARGET_MUNICIPALITIES)}"
        )

    bbox = config.TARGET_MUNICIPALITIES[municipality]["bbox"]
    lon_min, lat_min, lon_max, lat_max = bbox

    df = generate(
        n=n,
        seed=seed,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    df["municipality"] = municipality
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000, help="Number of rows")
    parser.add_argument(
        "--out",
        type=Path,
        default=config.PROCESSED_DATA_DIR / "training.parquet",
        help="Output path (parquet)",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()

    df = generate(n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"✓ Wrote {len(df):,} rows to {args.out}")
    print("\nClass distribution:")
    print(df["risk_class"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
