"""LangGraph agent for drafting foundation remediation reports.

Architecture:
    assess_risk → lookup_remediation → draft_report → quality_check → END

The agent works in two modes:
    1. **Template mode** (default): Structured report from domain knowledge.
       No API key needed — always works in demos and HF Spaces.
    2. **LLM mode**: Uses an LLM to produce more natural language reports.
       Activated by setting ANTHROPIC_API_KEY or OPENAI_API_KEY env var.

This demonstrates a production-ready LangGraph architecture where:
    - State is typed and validated (TypedDict)
    - Each node has a single responsibility
    - The graph has a conditional edge (quality gate)
    - Knowledge retrieval is separated from generation

Usage
-----
    from zakkend.agent.graph import create_remediation_agent
    agent = create_remediation_agent()
    result = agent.invoke({"building_features": {...}})
    print(result["report"])
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from zakkend import config
from zakkend.agent.knowledge import (
    FundingScheme,
    InspectionBody,
    RemediationOption,
    get_applicable_funding,
    get_applicable_remediations,
    get_inspection_bodies,
)

logger = logging.getLogger(__name__)


# ──────────────────────── State schema ─────────────────────────

class AgentState(TypedDict, total=False):
    """Typed state flowing through the LangGraph pipeline."""

    # Input
    building_features: dict[str, Any]

    # After assess_risk
    risk_class: str
    risk_score: float
    class_probabilities: dict[str, float]
    shap_drivers: list[dict[str, Any]]

    # After lookup_remediation
    remediation_options: list[dict]
    funding_schemes: list[dict]
    inspection_bodies: list[dict]

    # After draft_report
    report: str
    report_sections: list[str]

    # After quality_check
    quality_passed: bool
    missing_sections: list[str]
    iteration: int


# ──────────────────────── Node: assess_risk ─────────────────────

def assess_risk(state: AgentState) -> AgentState:
    """Run the XGBoost model + SHAP explainer on building features."""
    from zakkend.explain.shap_explainer import SubsidenceExplainer
    from zakkend.models.baseline import TrainedModel

    features = state["building_features"]
    logger.info(f"Assessing risk for building: year={features.get('year_built')}, "
                f"foundation={features.get('foundation_type')}, soil={features.get('soil_type')}")

    model = TrainedModel.load()
    explainer = SubsidenceExplainer(model)

    import pandas as pd
    df = pd.DataFrame([features])
    probs = model.predict_proba(df)[0]
    explanation = explainer.explain(df)

    class_probs = {
        name: float(p) for name, p in zip(model.class_names, probs)
    }

    shap_drivers = [
        {
            "feature": c.feature,
            "value": c.value,
            "shap_value": c.shap_value,
            "direction": c.direction,
        }
        for c in explanation.top_drivers(k=5)
    ]

    # Expected risk score (probability-weighted)
    n = len(config.RISK_CLASSES) - 1
    risk_score = sum(p * (i / n) for i, p in enumerate(probs))

    return {
        **state,
        "risk_class": explanation.predicted_class,
        "risk_score": round(risk_score, 3),
        "class_probabilities": class_probs,
        "shap_drivers": shap_drivers,
    }


# ──────────────────── Node: lookup_remediation ──────────────────

def lookup_remediation(state: AgentState) -> AgentState:
    """Query the knowledge base for applicable remediation options."""
    risk_class = state["risk_class"]
    features = state["building_features"]
    foundation = features.get("foundation_type", "unknown")
    soil = features.get("soil_type", "unknown")

    logger.info(f"Looking up remediation for: risk={risk_class}, "
                f"foundation={foundation}, soil={soil}")

    options = get_applicable_remediations(risk_class, foundation, soil)
    funding = get_applicable_funding(risk_class)
    inspectors = get_inspection_bodies(foundation)

    return {
        **state,
        "remediation_options": [
            {
                "name": o.name,
                "description": o.description,
                "applicable_when": o.applicable_when,
                "estimated_cost": o.estimated_cost_eur,
                "duration": o.typical_duration,
                "urgency": o.urgency,
            }
            for o in options
        ],
        "funding_schemes": [
            {
                "name": f.name,
                "provider": f.provider,
                "description": f.description,
                "url": f.url,
                "max_amount": f.max_amount_eur,
                "eligibility": f.eligibility,
            }
            for f in funding
        ],
        "inspection_bodies": [
            {
                "name": b.name,
                "description": b.description,
                "url": b.url,
                "service_type": b.service_type,
            }
            for b in inspectors
        ],
    }


# ──────────────────── Node: draft_report ────────────────────────

def _risk_emoji(risk_class: str) -> str:
    return {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}.get(risk_class, "⚪")


def draft_report(state: AgentState) -> AgentState:
    """Draft a remediation report — template mode or LLM mode."""
    # Check if an LLM API key is available
    has_llm = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    if has_llm:
        report = _draft_report_llm(state)
    else:
        report = _draft_report_template(state)

    # Track which sections are present
    sections = []
    for section_name in ["RISK ASSESSMENT", "KEY RISK DRIVERS", "REMEDIATION",
                          "FUNDING", "INSPECTION", "NEXT STEPS"]:
        if section_name in report.upper():
            sections.append(section_name)

    return {
        **state,
        "report": report,
        "report_sections": sections,
        "iteration": state.get("iteration", 0) + 1,
    }


def _draft_report_template(state: AgentState) -> str:
    """Structured template-based report — no API key needed."""
    features = state["building_features"]
    risk_class = state["risk_class"]
    risk_score = state["risk_score"]
    drivers = state["shap_drivers"]
    options = state["remediation_options"]
    funding = state["funding_schemes"]
    inspectors = state["inspection_bodies"]
    probs = state["class_probabilities"]

    emoji = _risk_emoji(risk_class)
    pct = round(risk_score * 100)
    date_str = datetime.now().strftime("%d %B %Y")

    # Build the drivers table
    driver_lines = []
    for d in drivers:
        arrow = "↑" if d["shap_value"] > 0 else "↓"
        driver_lines.append(
            f"| {d['feature']:30s} | {str(d['value']):15s} | "
            f"{d['shap_value']:+.3f} | {arrow} risk |"
        )
    drivers_table = "\n".join(driver_lines)

    # Build remediation section
    remediation_lines = []
    for i, opt in enumerate(options, 1):
        remediation_lines.append(
            f"### {i}. {opt['name']}\n\n"
            f"**When applicable:** {opt['applicable_when']}\n\n"
            f"{opt['description']}\n\n"
            f"- **Estimated cost:** {opt['estimated_cost']}\n"
            f"- **Typical duration:** {opt['duration']}\n"
            f"- **Urgency:** {opt['urgency'].replace('_', ' ').title()}\n"
        )

    # Build funding section
    funding_lines = []
    for f in funding:
        funding_lines.append(
            f"- **{f['name']}** ({f['provider']}): {f['description']} "
            f"Max: {f['max_amount']}. "
            f"[More info]({f['url']})"
        )

    # Build inspection section
    inspection_lines = []
    for b in inspectors:
        inspection_lines.append(
            f"- **{b['name']}** — {b['service_type']}. "
            f"{b['description']}. [{b['url']}]({b['url']})"
        )

    report = f"""# 🏚️ Foundation Subsidence Risk Report

**Generated:** {date_str}
**System:** Zakkend Nederland v0.1 — ML-powered risk assessment
**Disclaimer:** This is an automated assessment. Always consult a certified
foundation inspector (F3O protocol) before making repair decisions.

---

## {emoji} Risk Assessment

| Metric | Value |
|--------|-------|
| **Risk class** | **{risk_class.upper()}** {emoji} |
| **Risk score** | **{pct}/100** |
| **P(low)** | {probs.get('low', 0):.1%} |
| **P(moderate)** | {probs.get('moderate', 0):.1%} |
| **P(high)** | {probs.get('high', 0):.1%} |
| **P(critical)** | {probs.get('critical', 0):.1%} |

### Building Profile

| Attribute | Value |
|-----------|-------|
| Year built | {features.get('year_built', 'N/A')} |
| Foundation type | {features.get('foundation_type', 'N/A')} |
| Soil type | {features.get('soil_type', 'N/A')} |
| Peat thickness | {features.get('peat_thickness_m', 'N/A')} m |
| Groundwater depth | {features.get('groundwater_depth_m', 'N/A')} m |
| InSAR deformation | {features.get('insar_deformation_mm_yr', 'N/A')} mm/yr |

---

## 🔍 Key Risk Drivers (SHAP Analysis)

The following factors had the greatest influence on the risk score.
Values are derived from SHAP (SHapley Additive exPlanations), providing
per-prediction transparency in compliance with EU AI Act Article 13.

| Feature                        | Value           | SHAP   | Effect  |
|--------------------------------|-----------------|--------|---------|
{drivers_table}

---

## 🔧 Recommended Remediation Actions

{chr(10).join(remediation_lines)}

---

## 💰 Available Funding & Support

{chr(10).join(funding_lines)}

---

## 🔬 Recommended Inspection Bodies

{chr(10).join(inspection_lines)}

---

## 📋 Next Steps

{"### ⚠️ URGENT — Critical Risk Detected" if risk_class == "critical" else "### Recommended Actions"}

1. **Commission a foundation inspection** (F3O protocol) through a
   KCAF-certified inspector. Cost: typically €1,500–€3,500.

2. {"**Contact the Nationaal Fonds Funderingsherstel** to discuss financing options for foundation repair." if risk_class in ("critical", "high") else "**Monitor** the building for signs of settlement (cracks, doors sticking, uneven floors)."}

3. **Contact your municipality** to ask about local subsidence programmes
   and whether your neighborhood is eligible for collective repair schemes.

4. {"**Do not delay.** Foundation damage accelerates exponentially once it begins. Early intervention is significantly cheaper than emergency repair." if risk_class == "critical" else "**Set up a monitoring programme** with tilt sensors and crack monitors if visible signs appear."}

---

*This report was generated by Zakkend Nederland, an ML-powered foundation
subsidence risk assessment tool. The model uses XGBoost with SHAP
explainability, trained on real Dutch building data from PDOK BAG.*

*For questions or feedback: [GitHub](https://github.com/prakharprakarsh/zakkend-nederland)*
"""
    return report.strip()


def _draft_report_llm(state: AgentState) -> str:
    """LLM-enhanced report generation — richer language, same structure.

    Falls back to template mode if the API call fails.
    """
    try:
        import anthropic

        client = anthropic.Anthropic()
        features = state["building_features"]
        risk_class = state["risk_class"]
        drivers = state["shap_drivers"]
        options = state["remediation_options"]
        funding = state["funding_schemes"]

        prompt = f"""You are a Dutch foundation engineering expert writing a remediation 
report for a homeowner. Be professional, clear, and empathetic.

Building: year={features.get('year_built')}, foundation={features.get('foundation_type')}, 
soil={features.get('soil_type')}, peat={features.get('peat_thickness_m')}m,
groundwater={features.get('groundwater_depth_m')}m, InSAR={features.get('insar_deformation_mm_yr')} mm/yr

Risk class: {risk_class} (score: {state['risk_score']})
Top SHAP drivers: {drivers}
Remediation options: {[o['name'] for o in options]}
Funding available: {[f['name'] for f in funding]}

Write a complete remediation report with these sections:
1. Risk Assessment (with the specific numbers)
2. Key Risk Drivers (explain SHAP values in plain language)
3. Remediation recommendations (specific to this building)
4. Funding options (with URLs)
5. Inspection recommendations
6. Next steps (prioritized, with urgency)

Use markdown formatting. Be specific to Dutch context."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    except Exception as e:
        logger.warning(f"LLM report generation failed ({e}), falling back to template")
        return _draft_report_template(state)


# ──────────────────── Node: quality_check ───────────────────────

REQUIRED_SECTIONS = ["RISK ASSESSMENT", "KEY RISK DRIVERS", "REMEDIATION",
                     "FUNDING", "INSPECTION", "NEXT STEPS"]


def quality_check(state: AgentState) -> AgentState:
    """Validate the report has all required sections."""
    report_upper = state.get("report", "").upper()
    missing = [s for s in REQUIRED_SECTIONS if s not in report_upper]

    passed = len(missing) == 0
    iteration = state.get("iteration", 1)

    if not passed:
        logger.warning(f"Quality check failed (iteration {iteration}): missing {missing}")
    else:
        logger.info(f"Quality check passed (iteration {iteration})")

    return {
        **state,
        "quality_passed": passed,
        "missing_sections": missing,
        "iteration": iteration,
    }


# ──────────────────── Conditional edge ──────────────────────────

def should_redraft(state: AgentState) -> str:
    """Route: retry draft if quality check failed (max 2 iterations)."""
    if state.get("quality_passed", False):
        return "end"
    if state.get("iteration", 0) >= 2:
        logger.warning("Max iterations reached, accepting report as-is")
        return "end"
    return "redraft"


# ──────────────────── Graph construction ────────────────────────

def create_remediation_agent() -> StateGraph:
    """Build and compile the LangGraph remediation agent.

    Graph topology:
        assess_risk → lookup_remediation → draft_report → quality_check
                                                              ↓
                                                    (pass) → END
                                                    (fail) → draft_report (retry)
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("assess_risk", assess_risk)
    workflow.add_node("lookup_remediation", lookup_remediation)
    workflow.add_node("draft_report", draft_report)
    workflow.add_node("quality_check", quality_check)

    # Add edges
    workflow.set_entry_point("assess_risk")
    workflow.add_edge("assess_risk", "lookup_remediation")
    workflow.add_edge("lookup_remediation", "draft_report")
    workflow.add_edge("draft_report", "quality_check")

    # Conditional edge: quality gate
    workflow.add_conditional_edges(
        "quality_check",
        should_redraft,
        {
            "end": END,
            "redraft": "draft_report",
        },
    )

    return workflow.compile()
