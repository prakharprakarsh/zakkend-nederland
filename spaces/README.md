---
title: Zakkend Nederland
emoji: 🏚️
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.41.0
app_file: app.py
pinned: true
license: mit
short_description: ML-powered foundation subsidence risk for Dutch homes
tags:
  - machine-learning
  - xgboost
  - shap
  - langgraph
  - geospatial
  - netherlands
  - explainable-ai
---

# 🏚️ Zakkend Nederland

**ML-powered foundation subsidence risk assessment for Dutch homes.**

An estimated 1,000,000+ Dutch homes face foundation damage from soil subsidence,
with total projected damage exceeding €60 billion by 2050.

## What this app does

1. **Predicts** foundation risk class using XGBoost (trained on real PDOK BAG data)
2. **Explains** every prediction with SHAP feature attribution (EU AI Act compliant)
3. **Generates** a remediation report via a LangGraph agent pipeline
4. **Provides** links to Dutch funding schemes and inspection bodies

## Tech stack

`XGBoost` · `SHAP` · `LangGraph` · `FastAPI` · `Streamlit` · `PDOK BAG` · `Sentinel-1 InSAR`

## Author

[Prakhar Prakarsh](https://linkedin.com/in/pprakarsh04/) · [GitHub](https://github.com/prakharprakarsh/zakkend-nederland)
