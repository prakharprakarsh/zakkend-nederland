"""Tests for the LangGraph remediation agent.

Tests cover:
- Knowledge base lookups
- Agent graph construction and execution
- Report content validation
- Quality check logic
"""

from __future__ import annotations

import pytest

from zakkend import config
from zakkend.agent.knowledge import (
    get_applicable_funding,
    get_applicable_remediations,
    get_inspection_bodies,
    REMEDIATION_OPTIONS,
    FUNDING_SCHEMES,
    INSPECTION_BODIES,
)


# ──────────────── Knowledge base tests ────────────────


class TestKnowledgeBase:
    def test_remediation_options_not_empty(self):
        assert len(REMEDIATION_OPTIONS) >= 4

    def test_funding_schemes_not_empty(self):
        assert len(FUNDING_SCHEMES) >= 3

    def test_inspection_bodies_not_empty(self):
        assert len(INSPECTION_BODIES) >= 3

    def test_critical_gets_immediate_remediation(self):
        options = get_applicable_remediations("critical", "wooden_pile", "peat")
        names = [o.name for o in options]
        assert any("herstel" in n.lower() or "replacement" in n.lower() for n in names)

    def test_low_risk_gets_monitoring_only(self):
        options = get_applicable_remediations("low", "concrete_pile", "sand")
        assert all(o.urgency == "monitoring" for o in options)

    def test_critical_gets_all_funding(self):
        funding = get_applicable_funding("critical")
        assert len(funding) == len(FUNDING_SCHEMES)

    def test_low_risk_gets_minimal_funding(self):
        funding = get_applicable_funding("low")
        assert len(funding) < len(FUNDING_SCHEMES)

    def test_wooden_pile_gets_shr_inspection(self):
        bodies = get_inspection_bodies("wooden_pile")
        names = [b.name for b in bodies]
        assert any("SHR" in n for n in names)

    def test_all_funding_schemes_have_urls(self):
        for f in FUNDING_SCHEMES:
            assert f.url.startswith("http"), f"{f.name} missing URL"

    def test_all_inspection_bodies_have_urls(self):
        for b in INSPECTION_BODIES:
            assert b.url.startswith("http"), f"{b.name} missing URL"


# ──────────────── Agent graph tests ────────────────


class TestAgentGraph:
    @pytest.fixture(scope="class")
    def _ensure_model(self):
        """Train a small model if none exists."""
        if not config.MODEL_PATH.exists():
            from zakkend.data.synthetic import generate
            from zakkend.models.baseline import train
            model = train(generate(n=2_000))
            model.save()

    @pytest.fixture
    def high_risk_features(self):
        return {
            "year_built": 1920,
            "building_age": 106,
            "foundation_type": "wooden_pile",
            "soil_type": "peat",
            "peat_thickness_m": 3.5,
            "groundwater_depth_m": 1.2,
            "groundwater_variability": 0.45,
            "insar_deformation_mm_yr": 4.0,
            "drought_exposure_index": 0.65,
            "distance_to_water_m": 80,
            "neighborhood_damage_rate": 0.22,
        }

    @pytest.fixture
    def low_risk_features(self):
        return {
            "year_built": 2015,
            "building_age": 11,
            "foundation_type": "concrete_pile",
            "soil_type": "sand",
            "peat_thickness_m": 0.0,
            "groundwater_depth_m": 3.0,
            "groundwater_variability": 0.1,
            "insar_deformation_mm_yr": 0.2,
            "drought_exposure_index": 0.1,
            "distance_to_water_m": 500,
            "neighborhood_damage_rate": 0.02,
        }

    def test_agent_produces_report(self, _ensure_model, high_risk_features):
        from zakkend.agent.graph import create_remediation_agent
        agent = create_remediation_agent()
        result = agent.invoke({"building_features": high_risk_features})

        assert "report" in result
        assert len(result["report"]) > 500  # meaningful report

    def test_report_contains_required_sections(self, _ensure_model, high_risk_features):
        from zakkend.agent.graph import create_remediation_agent
        agent = create_remediation_agent()
        result = agent.invoke({"building_features": high_risk_features})

        report = result["report"].upper()
        assert "RISK ASSESSMENT" in report
        assert "REMEDIATION" in report
        assert "FUNDING" in report

    def test_quality_check_passes(self, _ensure_model, high_risk_features):
        from zakkend.agent.graph import create_remediation_agent
        agent = create_remediation_agent()
        result = agent.invoke({"building_features": high_risk_features})
        assert result.get("quality_passed", False) is True

    def test_high_risk_mentions_urgency(self, _ensure_model, high_risk_features):
        from zakkend.agent.graph import create_remediation_agent
        agent = create_remediation_agent()
        result = agent.invoke({"building_features": high_risk_features})
        report_lower = result["report"].lower()
        assert "urgent" in report_lower or "immediate" in report_lower or "critical" in report_lower

    def test_report_includes_shap_drivers(self, _ensure_model, high_risk_features):
        from zakkend.agent.graph import create_remediation_agent
        agent = create_remediation_agent()
        result = agent.invoke({"building_features": high_risk_features})
        assert "shap_drivers" in result
        assert len(result["shap_drivers"]) == 5

    def test_low_risk_gets_monitoring_recommendation(self, _ensure_model, low_risk_features):
        from zakkend.agent.graph import create_remediation_agent
        agent = create_remediation_agent()
        result = agent.invoke({"building_features": low_risk_features})
        report_lower = result["report"].lower()
        assert "monitor" in report_lower
