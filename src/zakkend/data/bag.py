"""BAG (Basisregistratie Adressen en Gebouwen) loader via PDOK WFS.

Fetches **real building data** from the Dutch national buildings registry
through PDOK's open WFS endpoint. No authentication required.

PDOK BAG WFS docs:
    https://service.pdok.nl/lv/bag/wfs/v2_0?request=GetCapabilities&service=WFS

Each building record includes:
    - identificatie: unique BAG building ID
    - bouwjaar: construction year
    - oppervlakte: floor area (m²)
    - status: building lifecycle status
    - gebruiksdoel: use type (woonfunctie = residential)
    - geometry: building footprint polygon (centroid extracted for lat/lon)

Usage
-----
    from zakkend.data.bag import fetch_buildings_bbox
    buildings = fetch_buildings_bbox(lon_min=4.68, lat_min=52.00,
                                     lon_max=4.75, lat_max=52.03)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import requests

from zakkend import config

logger = logging.getLogger(__name__)

_WFS_MAX_FEATURES = 1000
_WFS_TIMEOUT = 30


@dataclass(frozen=True)
class BAGBuilding:
    """Parsed BAG building record."""

    identificatie: str
    lat: float
    lon: float
    bouwjaar: int
    oppervlakte: float
    gebruiksdoel: str
    status: str


def _extract_centroid(geometry: dict) -> tuple[float, float] | None:
    """Extract centroid (lon, lat) from a GeoJSON geometry."""
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates")
    if not coords:
        return None

    try:
        if gtype == "Point":
            return (coords[0], coords[1])
        elif gtype == "Polygon":
            ring = coords[0]
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            return (sum(lons) / len(lons), sum(lats) / len(lats))
        elif gtype == "MultiPolygon":
            largest = max(coords, key=lambda poly: len(poly[0]))
            ring = largest[0]
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            return (sum(lons) / len(lons), sum(lats) / len(lats))
        else:
            logger.warning(f"Unexpected geometry type: {gtype}")
            return None
    except (IndexError, TypeError):
        return None


def fetch_buildings_bbox(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    max_records: int = 5000,
    residential_only: bool = True,
) -> pd.DataFrame:
    """Fetch BAG buildings within a bounding box from PDOK WFS.

    Parameters
    ----------
    lon_min, lat_min, lon_max, lat_max
        WGS84 bounding box coordinates.
    max_records
        Safety cap on total records (paginated in batches of 1000).
    residential_only
        If True, only fetch buildings with gebruiksdoel='woonfunctie'.

    Returns
    -------
    pd.DataFrame
        Columns: identificatie, lat, lon, bouwjaar, oppervlakte,
        gebruiksdoel, status
    """
    all_features: list[dict] = []
    start_index = 0

    cql_parts = ["status='Pand in gebruik'"]
    if residential_only:
        cql_parts.append("gebruiksdoel='woonfunctie'")
    cql_filter = " AND ".join(cql_parts)

    while start_index < max_records:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": "bag:pand",
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "bbox": f"{lat_min},{lon_min},{lat_max},{lon_max},EPSG:4326",
            "count": min(_WFS_MAX_FEATURES, max_records - start_index),
            "startIndex": start_index,
            "CQL_FILTER": cql_filter,
        }

        logger.info(
            f"Fetching BAG batch: startIndex={start_index}, "
            f"bbox=({lon_min:.3f},{lat_min:.3f},{lon_max:.3f},{lat_max:.3f})"
        )

        try:
            resp = requests.get(config.PDOK_BAG_WFS, params=params, timeout=_WFS_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"PDOK WFS request failed: {e}")
            break

        data = resp.json()
        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        start_index += len(features)

        logger.info(f"  → received {len(features)} features (total: {len(all_features)})")

        if len(features) < _WFS_MAX_FEATURES:
            break

        # Be respectful to PDOK's servers
        time.sleep(0.5)

    if not all_features:
        logger.warning("No buildings found in the specified bounding box.")
        return pd.DataFrame()

    return _features_to_dataframe(all_features)


def fetch_buildings_municipality(
    municipality_name: str,
    max_records: int = 5000,
    residential_only: bool = True,
) -> pd.DataFrame:
    """Fetch buildings for a known target municipality.

    Uses pre-configured bounding boxes from config.TARGET_MUNICIPALITIES.
    """
    muni = config.TARGET_MUNICIPALITIES.get(municipality_name)
    if not muni:
        available = ", ".join(config.TARGET_MUNICIPALITIES.keys())
        raise ValueError(f"Unknown municipality '{municipality_name}'. Available: {available}")

    lon_min, lat_min, lon_max, lat_max = muni["bbox"]
    logger.info(f"Fetching buildings for {municipality_name} (bbox: {muni['bbox']})")

    df = fetch_buildings_bbox(
        lon_min=lon_min,
        lat_min=lat_min,
        lon_max=lon_max,
        lat_max=lat_max,
        max_records=max_records,
        residential_only=residential_only,
    )
    if not df.empty:
        df["municipality"] = municipality_name
    return df


def _features_to_dataframe(features: list[dict]) -> pd.DataFrame:
    """Parse GeoJSON features into a clean DataFrame."""
    records = []
    seen_ids: set[str] = set()

    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry")

        bag_id = str(props.get("identificatie", ""))
        if not bag_id or bag_id in seen_ids:
            continue
        seen_ids.add(bag_id)

        centroid = _extract_centroid(geom) if geom else None
        if centroid is None:
            continue

        lon, lat = centroid
        bouwjaar = props.get("bouwjaar")
        if not bouwjaar or int(bouwjaar) < 1600:
            continue

        records.append(
            {
                "identificatie": bag_id,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "bouwjaar": int(bouwjaar),
                "oppervlakte": float(props.get("oppervlakte", 0)),
                "gebruiksdoel": props.get("gebruiksdoel", "onbekend"),
                "status": props.get("status", "onbekend"),
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["identificatie"])
        logger.info(
            f"Parsed {len(df)} unique buildings "
            f"(year range: {df['bouwjaar'].min()}–{df['bouwjaar'].max()})"
        )
    return df
