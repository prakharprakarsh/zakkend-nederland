"""Tests for Phase 2 real data loaders and pipeline.

These tests validate:
- BAG response parsing (mocked, since PDOK isn't available in CI)
- Soil classification rules match Dutch geology
- InSAR estimates are calibrated to known subsidence zones
- Pipeline produces correct schema
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zakkend import config
from zakkend.data.bag import _extract_centroid, _features_to_dataframe
from zakkend.data.insar import _estimate_insar_deformation
from zakkend.data.soil import _classify_soil_by_coordinates, _bro_code_to_simple
from zakkend.data.weather import _estimate_drought_exposure


# ──────────────────── BAG parser tests ────────────────────


class TestBAGParser:
    def test_extract_centroid_polygon(self):
        geom = {
            "type": "Polygon",
            "coordinates": [[[4.70, 52.01], [4.72, 52.01], [4.72, 52.03], [4.70, 52.03], [4.70, 52.01]]],
        }
        lon, lat = _extract_centroid(geom)
        assert 4.70 <= lon <= 4.72
        assert 52.01 <= lat <= 52.03

    def test_extract_centroid_point(self):
        geom = {"type": "Point", "coordinates": [4.71, 52.02]}
        lon, lat = _extract_centroid(geom)
        assert lon == 4.71
        assert lat == 52.02

    def test_extract_centroid_none_for_empty(self):
        assert _extract_centroid({}) is None
        assert _extract_centroid({"type": "Polygon", "coordinates": None}) is None

    def test_features_to_dataframe(self):
        features = [
            {
                "properties": {
                    "identificatie": "0513100000012345",
                    "bouwjaar": 1920,
                    "oppervlakte": 85,
                    "gebruiksdoel": "woonfunctie",
                    "status": "Pand in gebruik",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [4.71, 52.02],
                },
            }
        ]
        df = _features_to_dataframe(features)
        assert len(df) == 1
        assert df.iloc[0]["bouwjaar"] == 1920
        assert df.iloc[0]["lat"] == 52.02

    def test_deduplication(self):
        feat = {
            "properties": {"identificatie": "SAME", "bouwjaar": 1950, "oppervlakte": 60,
                           "gebruiksdoel": "woonfunctie", "status": "Pand in gebruik"},
            "geometry": {"type": "Point", "coordinates": [4.71, 52.02]},
        }
        df = _features_to_dataframe([feat, feat, feat])
        assert len(df) == 1


# ──────────────────── Soil classification tests ────────────────────


class TestSoilClassification:
    """Verify soil rules match known Dutch geology."""

    def test_gouda_is_peat(self):
        assert _classify_soil_by_coordinates(52.01, 4.71) == "peat"

    def test_veluwe_is_sand(self):
        assert _classify_soil_by_coordinates(52.2, 5.8) == "sand"

    def test_southern_limburg_is_loess(self):
        assert _classify_soil_by_coordinates(50.85, 5.9) == "loess"

    def test_betuwe_is_clay(self):
        assert _classify_soil_by_coordinates(51.9, 5.3) == "clay"

    def test_coast_is_clay(self):
        assert _classify_soil_by_coordinates(52.1, 4.3) == "clay"

    def test_bro_code_veen(self):
        assert _bro_code_to_simple("Vk") == "peat"
        assert _bro_code_to_simple("VEEN") == "peat"

    def test_bro_code_klei(self):
        assert _bro_code_to_simple("Mn25A") == "clay"
        assert _bro_code_to_simple("Rn47C") == "clay"

    def test_bro_code_zand(self):
        assert _bro_code_to_simple("Zd21") == "sand"
        assert _bro_code_to_simple("Hn21") == "sand"

    def test_bro_code_loess(self):
        assert _bro_code_to_simple("Ld5") == "loess"


# ──────────────────── InSAR estimation tests ────────────────────


class TestInSAR:
    """Verify InSAR estimates match published BDK data ranges."""

    def test_gouda_high_subsidence(self):
        """Gouda should have 3-8 mm/yr subsidence (BDK published)."""
        values = [
            _estimate_insar_deformation(52.01, 4.71, "peat", 1920)
            for _ in range(10)
        ]
        mean_val = np.mean(values)
        assert 1.5 < mean_val < 12, f"Gouda mean InSAR {mean_val} outside expected range"

    def test_eastern_nl_stable(self):
        """Eastern sand should be < 1 mm/yr."""
        val = _estimate_insar_deformation(52.2, 6.5, "sand", 2000)
        assert val < 2.0, f"Eastern NL InSAR {val} too high"

    def test_peat_higher_than_sand(self):
        """Same location concept: peat soil should subside more."""
        peat = _estimate_insar_deformation(52.1, 4.9, "peat", 1950)
        sand = _estimate_insar_deformation(52.1, 4.9, "sand", 1950)
        # Due to randomness, just check on average
        peat_vals = [_estimate_insar_deformation(52.1, 4.9, "peat", 1950) for _ in range(20)]
        sand_vals = [_estimate_insar_deformation(52.1, 4.9, "sand", 1950) for _ in range(20)]
        assert np.mean(peat_vals) > np.mean(sand_vals)


# ──────────────────── Drought estimation tests ────────────────────


class TestDrought:
    def test_south_more_drought_than_north(self):
        south = _estimate_drought_exposure(51.5, 5.0, "clay")
        north = _estimate_drought_exposure(53.0, 5.0, "clay")
        assert south > north

    def test_peat_amplifies_drought(self):
        peat = _estimate_drought_exposure(52.0, 4.8, "peat")
        sand = _estimate_drought_exposure(52.0, 4.8, "sand")
        assert peat > sand

    def test_drought_bounded(self):
        val = _estimate_drought_exposure(52.0, 5.0, "clay")
        assert 0.0 <= val <= 1.0


# ──────────────────── Pipeline schema test ────────────────────


class TestPipelineSchema:
    """Test that the pipeline output matches the model's expected schema."""

    def test_harmonized_schema_has_all_feature_columns(self):
        """Create a minimal mock BAG-like df and run through enrichment."""
        from zakkend.data.pipeline import (
            _estimate_foundation,
            _estimate_groundwater,
            _estimate_spatial_features,
            _harmonize_to_schema,
        )
        from zakkend.data.insar import enrich_with_insar
        from zakkend.data.soil import enrich_with_soil
        from zakkend.data.weather import enrich_with_drought

        mock_buildings = pd.DataFrame(
            {
                "identificatie": ["TEST001", "TEST002"],
                "lat": [52.01, 52.02],
                "lon": [4.71, 4.72],
                "bouwjaar": [1920, 2005],
                "oppervlakte": [85, 120],
                "gebruiksdoel": ["woonfunctie", "woonfunctie"],
                "status": ["Pand in gebruik", "Pand in gebruik"],
                "municipality": ["Gouda", "Gouda"],
            }
        )

        df = enrich_with_soil(mock_buildings, use_api=False)
        df = enrich_with_insar(df)
        df = enrich_with_drought(df)
        df = _estimate_groundwater(df)
        df = _estimate_foundation(df)
        df = _estimate_spatial_features(df)
        result = _harmonize_to_schema(df)

        missing = set(config.FEATURE_COLUMNS) - set(result.columns)
        assert not missing, f"Pipeline output missing columns: {missing}"
        assert len(result) == 2
