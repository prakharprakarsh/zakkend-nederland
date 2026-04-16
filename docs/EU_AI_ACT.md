# EU AI Act alignment

This project is designed with EU AI Act obligations in mind from the outset.

## Risk classification

Under the EU AI Act taxonomy (Regulation (EU) 2024/1689), an automated
property-damage risk score provided to consumers sits in a **grey zone**:

- Not explicitly listed as high-risk (Annex III)
- But "creditworthiness" adjacent — if banks or insurers use the output in
  mortgage or policy decisions, the downstream use case **could** be high-risk
- The prudent position is to build to high-risk standards from day one,
  so the tool can be licensed into regulated workflows without rework

## Article 13 — Transparency & information provision

Every `/explain` call returns:

1. The predicted class + calibrated probabilities
2. The base expected value (class prior)
3. Top-5 SHAP feature contributions with direction and magnitude
4. Feature-level input echo so the caller can audit what was used

This satisfies the substantive core of Article 13: an affected person can
understand, in non-technical terms, *why* a specific output was produced.

## Article 15 — Accuracy, robustness and cybersecurity

- Training metrics are persisted alongside the model (`models/metrics.json`)
- Tests validate both schema and domain-level sanity (e.g. peat homes score
  higher on average; older wooden-pile homes score higher than new concrete)
- Pydantic range validation blocks out-of-distribution inputs

## Article 10 — Data governance

Phase 2+ work will use only open data with documented licences
(see [`DATA_SOURCES.md`](DATA_SOURCES.md)). No personal data is stored —
only address-level features derived from public registries.

## Model card (Phase 5 deliverable)

A full model card following the [Google/Mitchell template](https://arxiv.org/abs/1810.03993)
will accompany the first real-data release, covering:

- Intended use / out-of-scope use
- Factor-level performance breakdown (by province, by build era)
- Ethical considerations (risk scores affecting property values)
- Caveats & recommendations for downstream users
