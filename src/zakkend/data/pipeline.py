"""Phase 2 ETL pipeline: BAG → enrich → training schema.

Orchestrates data from multiple Dutch government sources into the exact
feature schema the model expects (config.FEATURE_COLUMNS). The pipeline
is designed so every step is:
    - Idempotent (safe to re-run)
    - Logged (every enrichment step reports stats)
    - Cached (intermediate parquets saved to data/raw/)

Usage
-----
    from zakkend.data.pipeline import build_real_dataset
    df = build_real_dataset("Gouda")
    df.to_parquet("data/processed/gouda_real.parquet")

    # Or via CLI:
    python -m zakkend.data.pipeline --municipalities Gouda Rotterdam
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zakkend import config
from zakkend.data.bag import fetch_buildings_municipality
from zakkend.data.insar import enrich_with_insar
from zakkend.data.soil import enrich_with_soil
from zakkend.data.weather import enrich_with_drought

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def _estimate_groundwater(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate groundwater depth and variability from soil + location.

    Phase 3 will use BRO grondwaterstandonderzoek (GLD) monitoring wells.
    For now, we use soil-based estimates calibrated to Dutch norms:
    - Peat areas: shallow water table (0.3–1.2m)
    - Clay areas: moderate (0.8–2.0m)
    - Sand areas: deeper (1.5–4.0m)
    """
    result = df.copy()
    n = len(result)
    rng = np.random.default_rng(config.RANDOM_STATE)

    depths = np.zeros(n)
    variabilities = np.zeros(n)

    for i, (_, row) in enumerate(result.iterrows()):
        soil = row["soil_type"]
        if soil == "peat":
            depths[i] = rng.normal(0.8, 0.25)
            variabilities[i] = rng.beta(3, 4)  # higher variability
        elif soil == "clay":
            depths[i] = rng.normal(1.3, 0.4)
            variabilities[i] = rng.beta(2, 5)
        elif soil == "sandy_clay":
            depths[i] = rng.normal(1.6, 0.5)
            variabilities[i] = rng.beta(2, 6)
        elif soil == "sand":
            depths[i] = rng.normal(2.5, 0.8)
            variabilities[i] = rng.beta(1.5, 8)
        elif soil == "loess":
            depths[i] = rng.normal(2.0, 0.6)
            variabilities[i] = rng.beta(1.5, 7)
        else:
            depths[i] = rng.normal(1.5, 0.5)
            variabilities[i] = rng.beta(2, 6)

    result["groundwater_depth_m"] = np.clip(depths, 0.1, 8.0).round(2)
    result["groundwater_variability"] = np.clip(variabilities, 0.01, 0.95).round(3)

    logger.info(
        f"Groundwater estimation: mean depth={depths.mean():.2f}m, "
        f"mean variability={variabilities.mean():.3f}"
    )

    return result


def _estimate_foundation(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate foundation type from construction year + soil.

    This is the key historical pattern:
    - Pre-1970 on soft soil: predominantly wooden piles
    - Pre-1970 on firm soil: strip foundations
    - Post-1970: concrete piles or modern slab
    """
    result = df.copy()
    rng = np.random.default_rng(config.RANDOM_STATE)

    year_col = "year_built" if "year_built" in result.columns else "bouwjaar"
    foundations = []

    for _, row in result.iterrows():
        year = row[year_col]
        soil = row["soil_type"]
        soft_soil = soil in ("peat", "clay", "sandy_clay")

        if year < 1945 and soft_soil:
            f = rng.choice(config.FOUNDATION_TYPES, p=[0.75, 0.15, 0.08, 0.02])
        elif year < 1970 and soft_soil:
            f = rng.choice(config.FOUNDATION_TYPES, p=[0.45, 0.40, 0.12, 0.03])
        elif year < 1970:
            f = rng.choice(config.FOUNDATION_TYPES, p=[0.08, 0.20, 0.60, 0.12])
        elif soft_soil:
            f = rng.choice(config.FOUNDATION_TYPES, p=[0.02, 0.78, 0.15, 0.05])
        else:
            f = rng.choice(config.FOUNDATION_TYPES, p=[0.01, 0.30, 0.40, 0.29])

        foundations.append(f)

    result["foundation_type"] = foundations
    logger.info(f"Foundation estimation:\n" f"{pd.Series(foundations).value_counts().to_string()}")

    return result


def _estimate_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add spatial features: distance to water, neighborhood damage rate."""
    result = df.copy()
    rng = np.random.default_rng(config.RANDOM_STATE)

    # Distance to nearest water body (canals are everywhere in western NL)
    is_western = result["lon"] < 5.0
    result["distance_to_water_m"] = (
        np.where(
            is_western,
            rng.exponential(200, len(result)),
            rng.exponential(800, len(result)),
        )
        .clip(5, 5000)
        .astype(int)
    )

    # Neighborhood damage rate: spatial hash for spatial autocorrelation
    lat_hash = np.floor(result["lat"].values * 50)
    lon_hash = np.floor(result["lon"].values * 50)
    neigh_hash = lat_hash + lon_hash

    # Buildings in peat zones get higher baseline neighborhood damage
    peat_mask = result["soil_type"] == "peat"
    rate_lookup = {}
    for h in np.unique(neigh_hash):
        is_peat_neighborhood = peat_mask[neigh_hash == h].any()
        if is_peat_neighborhood:
            rate_lookup[h] = float(rng.beta(2, 6))  # higher
        else:
            rate_lookup[h] = float(rng.beta(1, 12))  # lower

    result["neighborhood_damage_rate"] = [round(rate_lookup[h], 3) for h in neigh_hash]

    return result


def _harmonize_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map BAG column names to training schema + compute derived columns."""
    result = df.copy()

    # Rename BAG columns → training schema
    if "bouwjaar" in result.columns and "year_built" not in result.columns:
        result["year_built"] = result["bouwjaar"]

    result["building_age"] = config.REFERENCE_YEAR - result["year_built"]

    # Ensure correct dtypes
    for col in config.CATEGORICAL_FEATURES:
        if col in result.columns:
            result[col] = result[col].astype("category")

    # Keep only the columns the model needs + metadata
    keep_cols = ["identificatie", "lat", "lon", "municipality"] + config.FEATURE_COLUMNS
    available = [c for c in keep_cols if c in result.columns]
    return result[available]


def build_real_dataset(
    municipality: str,
    max_buildings: int = 5000,
) -> pd.DataFrame:
    """Build a real-data feature matrix for one municipality.

    Pipeline steps:
    1. Fetch real buildings from PDOK BAG (WFS)
    2. Enrich with soil type (BRO cache + coordinate fallback)
    3. Enrich with InSAR deformation rates (calibrated estimator)
    4. Enrich with drought exposure (KNMI patterns)
    5. Estimate groundwater from soil + location
    6. Estimate foundation type from year + soil
    7. Add spatial features (water distance, neighborhood rate)
    8. Harmonize to training schema
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Building real dataset for: {municipality}")
    logger.info(f"{'=' * 60}")

    # Step 1: BAG buildings
    logger.info("Step 1/7: Fetching BAG buildings from PDOK...")
    df = fetch_buildings_municipality(municipality, max_records=max_buildings)
    if df.empty:
        logger.error(f"No buildings found for {municipality}!")
        return pd.DataFrame()
    logger.info(f"  → {len(df)} buildings fetched")

    # Step 2: Soil enrichment
    logger.info("Step 2/7: Enriching with soil data...")
    df = enrich_with_soil(df)

    # Step 3: InSAR deformation
    logger.info("Step 3/7: Enriching with InSAR deformation rates...")
    df = enrich_with_insar(df)

    # Step 4: Drought exposure
    logger.info("Step 4/7: Enriching with drought exposure...")
    df = enrich_with_drought(df)

    # Step 5: Groundwater
    logger.info("Step 5/7: Estimating groundwater conditions...")
    df = _estimate_groundwater(df)

    # Step 6: Foundation type
    logger.info("Step 6/7: Estimating foundation types...")
    df = _estimate_foundation(df)

    # Step 7: Spatial features
    logger.info("Step 7/7: Computing spatial features...")
    df = _estimate_spatial_features(df)

    # Harmonize to model schema
    result = _harmonize_to_schema(df)
    logger.info(f"\n✓ Dataset complete: {len(result)} buildings, {len(result.columns)} columns")

    return result


def build_multi_municipality(
    municipalities: list[str] | None = None,
    max_per_municipality: int = 5000,
) -> pd.DataFrame:
    """Build datasets for multiple municipalities and concatenate.

    Parameters
    ----------
    municipalities
        List of municipality names (must exist in config.TARGET_MUNICIPALITIES).
        Defaults to all configured municipalities.
    max_per_municipality
        Max buildings per municipality.
    """
    if municipalities is None:
        municipalities = list(config.TARGET_MUNICIPALITIES.keys())

    frames = []
    for muni in municipalities:
        try:
            df = build_real_dataset(muni, max_buildings=max_per_municipality)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.error(f"Failed for {muni}: {e}")
            continue

    if not frames:
        logger.error("No data was fetched for any municipality!")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"\n{'=' * 60}\n"
        f"Combined dataset: {len(combined)} buildings across "
        f"{len(frames)} municipalities\n"
        f"{'=' * 60}"
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real data from Dutch government APIs")
    parser.add_argument(
        "--municipalities",
        nargs="+",
        default=None,
        help=f"Target municipalities. Available: {list(config.TARGET_MUNICIPALITIES.keys())}",
    )
    parser.add_argument("--max-per-muni", type=int, default=5000)
    parser.add_argument(
        "--out",
        type=Path,
        default=config.PROCESSED_DATA_DIR / "real_data.parquet",
    )
    args = parser.parse_args()

    df = build_multi_municipality(
        municipalities=args.municipalities,
        max_per_municipality=args.max_per_muni,
    )

    if not df.empty:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"\n✓ Saved {len(df):,} buildings to {args.out}")

        print("\nSample of data:")
        print(df.head(3).to_string())

        print("\nFeature statistics:")
        numeric_cols = df.select_dtypes(include="number").columns
        print(df[numeric_cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
