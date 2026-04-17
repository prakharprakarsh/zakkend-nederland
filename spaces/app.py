"""Zakkend Nederland — Hugging Face Spaces Streamlit App.

This is the recruiter-facing demo. It runs the full pipeline:
    building features → XGBoost risk model → SHAP explanation
    → LangGraph remediation agent → downloadable PDF report

Deploy to HF Spaces:
    huggingface-cli login
    cd spaces/
    git init && git remote add origin https://huggingface.co/spaces/<username>/zakkend-nederland
    git add . && git commit -m "Deploy" && git push -u origin main
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ──────────── Ensure the package is importable ────────────
# When running in HF Spaces, the package is in the same directory
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


# ──────────── Page config ────────────
st.set_page_config(
    page_title="Zakkend Nederland — Foundation Risk",
    page_icon="🏚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────── Custom CSS ────────────
st.markdown("""
<style>
    .risk-critical { color: #e74c3c; font-size: 3rem; font-weight: 800; }
    .risk-high { color: #e67e22; font-size: 3rem; font-weight: 800; }
    .risk-moderate { color: #f1c40f; font-size: 3rem; font-weight: 800; }
    .risk-low { color: #2ecc71; font-size: 3rem; font-weight: 800; }
    .metric-label { color: #8a93ad; font-size: 0.85rem; text-transform: uppercase;
                    letter-spacing: 0.05em; }
    .shap-positive { color: #e74c3c; }
    .shap-negative { color: #2ecc71; }
    div[data-testid="stSidebar"] { background-color: #0e1525; }
    .stDownloadButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)


# ──────────── Sidebar: Building input form ────────────
with st.sidebar:
    st.markdown("## 🏚️ Zakkend Nederland")
    st.caption("Foundation subsidence risk for Dutch homes")
    st.markdown("---")

    st.markdown("### Building attributes")
    year_built = st.slider("Year built", 1700, 2025, 1925, step=5)

    foundation_type = st.selectbox(
        "Foundation type",
        ["wooden_pile", "concrete_pile", "strip", "slab"],
        help="Pre-1970 homes on soft soil typically have wooden piles"
    )

    st.markdown("### Soil & hydrology")
    soil_type = st.selectbox(
        "Soil type",
        ["peat", "clay", "sandy_clay", "sand", "loess"],
        help="Western NL is predominantly peat/clay"
    )

    peat_thickness = st.slider("Peat thickness (m)", 0.0, 10.0, 2.5, step=0.5)
    groundwater_depth = st.slider("Groundwater depth (m)", 0.1, 5.0, 1.1, step=0.1)
    groundwater_var = st.slider("Groundwater variability (0–1)", 0.0, 1.0, 0.45, step=0.05)

    st.markdown("### Remote sensing")
    insar = st.slider("InSAR deformation (mm/yr)", 0.0, 15.0, 3.2, step=0.2)
    drought = st.slider("Drought exposure (0–1)", 0.0, 1.0, 0.55, step=0.05)

    st.markdown("### Spatial context")
    distance_water = st.slider("Distance to water (m)", 5, 2000, 120, step=10)
    neighborhood_rate = st.slider("Neighborhood damage rate", 0.0, 0.5, 0.18, step=0.02)

    assess_button = st.button("🔎 Assess Risk", use_container_width=True, type="primary")

# ──────────── Build features dict ────────────
features = {
    "year_built": year_built,
    "building_age": 2026 - year_built,
    "foundation_type": foundation_type,
    "soil_type": soil_type,
    "peat_thickness_m": peat_thickness,
    "groundwater_depth_m": groundwater_depth,
    "groundwater_variability": groundwater_var,
    "insar_deformation_mm_yr": insar,
    "drought_exposure_index": drought,
    "distance_to_water_m": distance_water,
    "neighborhood_damage_rate": neighborhood_rate,
}


# ──────────── Main content ────────────
if not assess_button:
    # Landing page
    st.markdown("# 🏚️ Zakkend Nederland")
    st.markdown(
        "### ML-powered foundation subsidence risk assessment for Dutch homes"
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Homes at risk", "1,000,000+")
        st.caption("Estimated by KCAF / TNO")
    with col2:
        st.metric("Projected damage", "€60B+")
        st.caption("By 2050, nationwide")
    with col3:
        st.metric("Model accuracy", "83.3%")
        st.caption("XGBoost on real BAG data")

    st.markdown("---")

    st.markdown("""
    **How to use:** Configure a building profile in the sidebar and click
    **🔎 Assess Risk**. The system will:

    1. **Predict** the foundation risk class using XGBoost
    2. **Explain** the prediction with SHAP feature attribution
    3. **Generate** a remediation report with actionable next steps
    4. **Provide** links to funding schemes and inspection bodies

    The model is trained on **real building data** from the Dutch national
    registry (PDOK BAG) enriched with soil, InSAR satellite deformation,
    and climate data.
    """)

    st.markdown("---")
    st.caption(
        "Built by [Prakhar Prakarsh](https://linkedin.com/in/pprakarsh04/) · "
        "[GitHub](https://github.com/prakharprakarsh/zakkend-nederland) · "
        "XGBoost + SHAP + LangGraph + FastAPI"
    )

else:
    # ──────────── Run assessment ────────────
    with st.spinner("Running risk assessment..."):
        try:
            from zakkend.models.baseline import TrainedModel
            from zakkend.explain.shap_explainer import SubsidenceExplainer
            from zakkend.agent.graph import create_remediation_agent

            # Load model
            model = TrainedModel.load()
            explainer = SubsidenceExplainer(model)

            # Predict
            df = pd.DataFrame([features])
            probs = model.predict_proba(df)[0]
            explanation = explainer.explain(df)

            n = len(model.class_names) - 1
            risk_score = sum(p * (i / n) for i, p in enumerate(probs))
            risk_class = explanation.predicted_class
            pct = round(risk_score * 100)

            # Run agent
            agent = create_remediation_agent()
            agent_result = agent.invoke({"building_features": features})
            report = agent_result.get("report", "Report generation failed.")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure you've trained the model first: `python scripts/train.py`")
            st.stop()

    # ──────────── Display results ────────────
    st.markdown("# 🏚️ Assessment Results")
    st.markdown("---")

    # Risk score header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        emoji = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}.get(risk_class, "⚪")
        st.markdown(f'<p class="risk-{risk_class}">{pct}/100</p>', unsafe_allow_html=True)
        st.markdown(f"**{risk_class.upper()}** {emoji}")
    with col2:
        st.metric("Year built", year_built)
    with col3:
        st.metric("Foundation", foundation_type.replace("_", " ").title())
    with col4:
        st.metric("Soil type", soil_type.title())

    st.markdown("---")

    # Two columns: SHAP + probabilities
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🔍 Key Risk Drivers (SHAP)")
        st.caption("Per-prediction feature attribution — EU AI Act Article 13 compliant")

        drivers = explanation.top_drivers(k=5)
        for d in drivers:
            color = "🔴" if d.shap_value > 0 else "🟢"
            bar_width = min(abs(d.shap_value) * 20, 100)
            direction = "increases" if d.shap_value > 0 else "decreases"
            st.markdown(
                f"{color} **{d.feature}** = `{d.value}` "
                f"— {direction} risk by {abs(d.shap_value):.3f}"
            )
            if d.shap_value > 0:
                st.progress(min(d.shap_value / 3.0, 1.0))
            else:
                st.progress(min(abs(d.shap_value) / 3.0, 1.0))

    with col_right:
        st.markdown("### 📊 Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": model.class_names,
            "Probability": [float(p) for p in probs],
        })
        st.bar_chart(prob_df.set_index("Class"), color="#ff7a59")

    st.markdown("---")

    # Full remediation report
    st.markdown("### 📋 Remediation Report")
    st.caption("Generated by LangGraph agent pipeline: assess → lookup → draft → quality check")

    with st.expander("View full report", expanded=True):
        st.markdown(report)

    # Download button
    st.download_button(
        label="📥 Download Report (Markdown)",
        data=report,
        file_name=f"zakkend_report_{risk_class}_{year_built}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption(
        "Built by [Prakhar Prakarsh](https://linkedin.com/in/pprakarsh04/) · "
        "[GitHub](https://github.com/prakharprakarsh/zakkend-nederland) · "
        "XGBoost + SHAP + LangGraph + FastAPI"
    )
