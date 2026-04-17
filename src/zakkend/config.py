"""Configuration constants for Zakkend Nederland."""

from __future__ import annotations

from pathlib import Path

# ──────────────────────────── Paths ────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

for _dir in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

MODEL_PATH: Path = MODELS_DIR / "subsidence_xgb.joblib"

# ──────────────────── Netherlands geographic bounds ─────────────────
NL_LAT_MIN: float = 50.75
NL_LAT_MAX: float = 53.55
NL_LON_MIN: float = 3.35
NL_LON_MAX: float = 7.25

# ──────────────────────── Soil categories ──────────────────────────
SOIL_TYPES: tuple[str, ...] = ("peat", "clay", "sandy_clay", "sand", "loess")

HIGH_PEAT_PROVINCE_LAT_RANGE: tuple[float, float] = (51.85, 53.25)
HIGH_PEAT_PROVINCE_LON_RANGE: tuple[float, float] = (4.20, 5.60)

# ─────────────────────── Foundation types ──────────────────────────
FOUNDATION_TYPES: tuple[str, ...] = ("wooden_pile", "concrete_pile", "strip", "slab")
WOODEN_PILE_ERA_END: int = 1970

# ───────────────────── Risk-score thresholds ───────────────────────
RISK_CLASSES: tuple[str, ...] = ("low", "moderate", "high", "critical")
RISK_THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.75)

# ──────────────────────── Feature schema ───────────────────────────
FEATURE_COLUMNS: list[str] = [
    "year_built",
    "building_age",
    "foundation_type",
    "soil_type",
    "peat_thickness_m",
    "groundwater_depth_m",
    "groundwater_variability",
    "insar_deformation_mm_yr",
    "drought_exposure_index",
    "distance_to_water_m",
    "neighborhood_damage_rate",
]
CATEGORICAL_FEATURES: list[str] = ["foundation_type", "soil_type"]
TARGET_COLUMN: str = "risk_class"

RANDOM_STATE: int = 42

# ──────────────── Target municipalities for Phase 2 ────────────────
# These are known high-risk areas for subsidence.
TARGET_MUNICIPALITIES: dict[str, dict] = {
    "Gouda": {
        "bbox": (4.68, 52.00, 4.75, 52.03),  # (lon_min, lat_min, lon_max, lat_max)
        "municipality_code": "0513",
    },
    "Rotterdam": {
        "bbox": (4.40, 51.88, 4.56, 51.96),
        "municipality_code": "0599",
    },
    "Zaanstad": {
        "bbox": (4.75, 52.43, 4.88, 52.50),
        "municipality_code": "0479",
    },
    "Dordrecht": {
        "bbox": (4.62, 51.78, 4.72, 51.83),
        "municipality_code": "0505",
    },
}

# ──────────────────── PDOK API endpoints ───────────────────────────
PDOK_BAG_WFS: str = "https://service.pdok.nl/lv/bag/wfs/v2_0"
PDOK_BRO_SOIL_WFS: str = "https://service.pdok.nl/bzk/bro-bodemkaart/wfs/v1_0"
KNMI_OPEN_DATA_BASE: str = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
