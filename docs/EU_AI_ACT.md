# EU AI Act — Gap Analysis

This document is a **self-assessment gap analysis**, not a compliance certification.
It maps EU AI Act obligations (Regulation (EU) 2024/1689) against what this project
currently does, identifies gaps, and notes what closing each gap would require.

---

## Risk classification (Annex III)

An automated property-damage risk score provided to consumers sits in a **grey zone**.
The strongest arguments from the text of the Act:

**Against high-risk classification (the property-insurance carve-out):**
Annex III §5(c) explicitly covers "risk assessment and pricing in relation to natural
persons in the case of **life and health** insurance." Property insurance is not listed.
This is a deliberate legislative choice: the Act's drafters specifically contemplated
insurance risk models, enumerated the ones in scope, and excluded property. A
property-subsidence risk score, read against this, is most naturally *not* high-risk
under the Act as written.

**The residual risk path — creditworthiness adjacency:**
Annex III §5(b) covers "creditworthiness assessment and credit scoring." If a bank or
mortgage lender uses this score as an input to a lending decision, the downstream
application could be high-risk regardless of how the tool itself is classified. The
instrument would then be an ancillary system to a high-risk system, with obligations
flowing through the integrating lender.

**Prudent position:** build to high-risk standards from day one so the tool can be
licensed into regulated workflows without rework, but do not assert high-risk
classification — that determination belongs to legal counsel with full deployment context.

This project therefore treats high-risk requirements as a **design target**, not a
concluded classification. Any commercial deployment requires a formal legal determination.

---

## Gap analysis

| Article | What the Act requires | What this project does today | Gap | What closing it would take |
|---|---|---|---|---|
| **Annex III — Classification** | Determine whether the system is high-risk before deployment | Grey zone: §5(c) explicitly covers life/health insurance risk assessment and excludes property — the legislative carve-out is evidence the system is *not* high-risk as a standalone tool; §5(b) creditworthiness path remains live if the output feeds a lending decision | No authoritative legal determination; downstream integrations are unknown | Formal legal opinion before any commercial deployment; build to high-risk standards as precaution in the meantime |
| **Article 10 — Data and data governance** | Training data that is relevant, representative, free from errors, and adequate for the intended purpose; processes to examine possible biases; documented data provenance | Labels are produced by a hand-written rule engine (`synthetic._compute_risk_score`); 9 of 11 features are RNG-generated with domain-informed priors; only `year_built` and the BAG geometry centroid come from a real government registry; provenance documented in `README.md` and `docs/DATA_SOURCES.md` | Labels are a formula, not real damage records — the model cannot be shown to be relevant to its intended purpose; simulated features cannot demonstrate representativeness against the real Dutch building stock; no bias examination performed | Weak labels from KCAF / municipal *funderingsrisico* zone polygons; real feature data from BRO (soil), BodemDalingsKaart (InSAR), KNMI (drought), GLD wells (groundwater); bias analysis across provinces, construction eras, and soil types |
| **Article 13 — Transparency** | Clear instructions for use; disclosure of AI involvement; transparency about capabilities and limitations; information enabling meaningful oversight by affected persons; per-instance explanation of significant automated decisions | `/explain` returns top-5 SHAP contributors with direction and magnitude, raw class probabilities (XGBoost softmax — not calibrated frequency estimates), and input echo; `docs/LIMITATIONS.md` documents all known gaps; probabilities are explicitly labelled as relative confidence scores, not frequencies | SHAP attributions reflect the synthetic rule engine, not real foundation-damage drivers; no formal instructions-for-use document; no end-user disclosure statement; no calibrated probabilities (see `docs/LIMITATIONS.md §L4`) | Real ground-truth labels → SHAP attributions reflect actual damage drivers; `CalibratedClassifierCV` + reliability diagram; formal instructions-for-use; end-user disclosure statement |
| **Article 14 — Human oversight** | Measures enabling human oversight of the system; ability to interrupt, override, or disregard outputs; training for oversight personnel; detection and response to anomalous outputs | None implemented — this is a research demo, not a deployed system | No override mechanism; no human-in-the-loop workflow; no anomaly monitoring; no reviewer training materials | For production deployment: audit trail, review queue for borderline (`high`/`critical`) cases, override workflow logged against the prediction record, reviewer training materials, monitoring dashboard |
| **Article 15 — Accuracy, robustness, cybersecurity** | Documented accuracy levels appropriate for the purpose; robustness to input errors and adversarial manipulation; cybersecurity; performance monitored over time | Metrics persisted in `models/metrics.json` and published in `README.md`; CI regenerates metrics deterministically; tests validate schema and domain-level sanity; `UnknownCategoryError` returns HTTP 422 for unseen categorical values; Pydantic validates individual field ranges | **The model collapses on an unseen city (0.092 accuracy, well below the 4-class uniform baseline of 0.250).** The collapse has one root cause and two downstream consequences. Root cause: `_classify_soil_by_coordinates` mislabels Dordrecht as `sandy_clay` via the catch-all branch — the Dordrecht bbox misses the river-clay rule by 0.08 deg longitude and the peat-belt rule by 0.07 deg latitude. Dordrecht is in reality a peat-and-river-clay city in the Drechtsteden; the `sandy_clay` assignment is an artefact of bbox placement, not its geology. Consequence 1: `sandy_clay` is absent from the frozen training vocabulary (all three training cities are peat), so all 974 Dordrecht rows encode as NaN and are routed to the XGBoost missing-value branch. Consequence 2: the rule engine scores `sandy_clay` lower than `peat`, so Dordrecht's true label distribution (65% moderate) is the inverse of the training distribution (76% critical), producing near-zero recall on the majority class. As currently constructed the held-out-city experiment measures what happens when the soil classifier misclassifies the test city — not whether the model generalises geographically. Additionally: accuracy on the random split (0.824) measures rule-recovery from a synthetic label function; no adversarial robustness testing; no security audit of the API | Real soil data via PDOK BRO Bodemkaart WFS (`use_soil_api=True`) gives Dordrecht its actual peat/clay classification instead of the catch-all label; real hydrology data via BodemDalingsKaart / KNMI / BRO GLD breaks the municipality confound for all simulated features; adversarial input testing; penetration test before any public deployment |

---

## Status summary

The EU AI Act framework has informed design choices throughout this project — SHAP
explanations, documented limitations, open-data-only provenance, and this gap analysis
are all motivated by it. However, this is a **research prototype with synthetic labels
and simulated features**. The gaps above are structural, not cosmetic:

- **Article 10:** No real labels; no real features; cannot demonstrate representativeness.
- **Article 13:** SHAP attributions describe the rule engine, not real damage dynamics.
- **Article 14:** No oversight infrastructure exists.
- **Article 15:** The model collapses on an unseen city (0.092 accuracy, well below the 4-class uniform baseline of 0.250). The collapse traces to one root cause — `_classify_soil_by_coordinates` mislabels Dordrecht as `sandy_clay` via the catch-all branch (bbox placement artefact, not its geology: Dordrecht is a peat-and-river-clay city in the Drechtsteden) — with two downstream consequences: (1) `sandy_clay` absent from the training vocabulary encodes all 974 test rows as NaN; (2) the rule engine's lower risk scores for `sandy_clay` invert the label distribution relative to the peat training cities. As constructed, the held-out-city experiment measures classifier mislabelling of the test city, not geographic generalisation. The 0.092 figure does not change; what changes is its cause. Accuracy on the random split measures rule-recovery; cross-field validation is absent.

A reviewer using this document to evaluate readiness for production deployment should
treat the current state as **not compliant** and the architecture as **compliance-oriented**.
The design is structured to close these gaps incrementally as real data becomes available.
