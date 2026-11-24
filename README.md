# Zakkend Nederland

Foundation subsidence risk scoring for Dutch homes — XGBoost + SHAP, served through FastAPI and a Streamlit demo, with a small LangGraph agent that turns a prediction into a plain-language remediation report.

[![CI](https://github.com/prakharprakarsh/zakkend-nederland/actions/workflows/ci.yml/badge.svg)](https://github.com/prakharprakarsh/zakkend-nederland/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Why

A large share of the housing stock in the western Netherlands sits on peat and wooden piles. When groundwater drops in dry summers the peat shrinks and the piles rot, and homeowners usually find out when the cracks appear. KCAF estimates around a million homes are exposed. I wanted to see how far you can get with open Dutch geodata (PDOK, BRO, KNMI, BodemDalingsKaart) towards a per-address risk signal, and where it breaks.

## What it does

- Pulls real building footprints and construction years from **PDOK BAG** for four municipalities (Gouda, Rotterdam, Zaanstad, Dordrecht — ~3,800 buildings)
- Joins soil type from the **BRO Bodemkaart** GeoPackage (spatial join in EPSG:28992), with a coordinate-rule fallback where the 1:50,000 map has no coverage
- Builds an 11-feature matrix (soil, peat thickness, groundwater, InSAR deformation, drought exposure, foundation type, ...)
- Trains a 4-class XGBoost model (`low` / `moderate` / `high` / `critical`) with class weights and early stopping
- Explains every prediction with SHAP
- Serves it via FastAPI + a Leaflet page, and a Streamlit app for Hugging Face Spaces
- A LangGraph agent (assess → look up remediation options → draft → quality check) writes the report; it falls back to a template when no LLM key is set

## Important: the labels are synthetic

There is no public per-address dataset of Dutch foundation damage, so the labels come from a documented rule engine (`synthetic._compute_risk_score` — a weighted sum of the features plus noise, thresholded into four classes). The accuracy numbers below therefore measure how well the pipeline **recovers that rule**, not real-world predictive skill.

Only `year_built` and the building location are measured directly (PDOK BAG). `soil_type` is real for ~27% of buildings (BRO join) and rule-based for the rest. The other features are simulated from published Dutch priors (TNO, KNMI, BodemDalingsKaart). Per-feature provenance is in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

The next real step is weak supervision from KCAF / municipal *funderingsrisico* zone maps — see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Architecture

```
PDOK BAG ─┐
BRO soil ─┼─► feature engineering ─► XGBoost ─► SHAP ─► FastAPI / Streamlit
KNMI     ─┤                                       │
InSAR    ─┘                                       └─► LangGraph report agent
```

More detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start

```bash
git clone https://github.com/prakharprakarsh/zakkend-nederland.git
cd zakkend-nederland
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e ".[dev]"

python scripts/train.py                  # synthetic data + train
uvicorn zakkend.api.main:app --reload    # http://localhost:8000
```

The demo takes the 11 features as inputs and returns the risk class, class probabilities and the top SHAP drivers. The map shows where you clicked but lat/lon is not a model input.

## Results

### Random split (synthetic, 10k rows)

`python scripts/train.py` — 60/20/20 stratified split, early stopping on validation.

| | Accuracy |
|---|---|
| Most-frequent baseline | 49.5% |
| Stratified random baseline | 35.1% |
| **XGBoost** | **82.4%** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| low | 63.4% | 73.2% | 68.0% |
| moderate | 88.1% | 84.2% | 86.1% |
| high | 74.7% | 79.8% | 77.1% |
| critical | 89.6% | 85.3% | 87.4% |

Class weights lifted `low` F1 from 0.603 to 0.680 without hurting the other classes.

### Held-out city (real BAG buildings)

`python scripts/train.py --split grouped` — train on Gouda + Rotterdam + Zaanstad, test on Dordrecht.

| | Accuracy | Macro F1 |
|---|---|---|
| Coordinate-rule soil only | 0.092 | 0.142 |
| With BRO Bodemkaart join | **0.614** | **0.480** |

The two rows aren't on identical labels (changing soil changes the rule-engine labels), and a constant `moderate` predictor scores 0.657 on Dordrecht — so this is not a generalisation claim. Breakdown in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## What I learned debugging the held-out city

The first grouped-split run collapsed to 9% accuracy — far below random. Tracing it back:

1. My coordinate-based soil classifier used bounding boxes per region. Dordrecht's box missed the river-clay rule by 0.08° longitude and the peat rule by 0.07° latitude, so every building fell through to the catch-all `sandy_clay`.
2. `sandy_clay` never appeared in training (all three training cities were peat), so XGBoost routed all 974 test rows down the missing-value branch.
3. Because the label engine depends on soil, Dordrecht's label distribution was also inverted relative to training.

One bug, two symptoms. Replacing the dead BRO WFS endpoint with a local spatial join against the bulk GeoPackage fixed the soil for ~half of Dordrecht and moved accuracy to 0.614. The lesson: with geodata, check the feature distributions per group before trusting any split.

## Engineering notes

- **Model format:** saved as XGBoost `.ubj` + a JSON metadata file (feature schema, frozen category vocabulary, metrics, xgboost version). No pickle/joblib, so loading a model file can't execute code.
- **Frozen categorical vocabulary:** unseen categories at inference raise instead of silently becoming NaN.
- **API validation:** Pydantic `Literal` types for categoricals, `extra="forbid"`, `building_age` derived server-side from `year_built`.
- **Reproducibility:** `requirements.lock.txt` alongside `requirements.txt`; `RANDOM_STATE=42` throughout.
- **Tests:** `pytest -q` — 68 tests, all offline, each run trains its own small model in a temp dir. CI runs on Python 3.11 and 3.12.

## Limitations

- Synthetic labels (see above)
- Most features simulated, not measured
- No postcode geocoder — inputs are manual
- Probabilities are uncalibrated softmax scores
- Spatial leakage through `neighborhood_damage_rate` inflates the random-split number

Full list: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). Also: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), [`docs/EU_AI_ACT.md`](docs/EU_AI_ACT.md).

## Roadmap

- [x] Synthetic pipeline, XGBoost baseline, SHAP, FastAPI demo
- [x] Real PDOK BAG buildings + BRO soil join
- [x] LangGraph report agent, Hugging Face Spaces deployment
- [ ] Weak supervision from KCAF / municipal zone data
- [ ] Probability calibration
- [ ] Postcode lookup

## Data sources

[PDOK](https://www.pdok.nl/) · [BRO Bodemkaart](https://www.broloket.nl/) · [TNO DINOloket](https://www.dinoloket.nl/) · [BodemDalingsKaart](https://bodemdalingskaart.nl/) · [KNMI](https://www.knmi.nl/) · [KCAF](https://www.kcaf.nl/)

## License

MIT — Prakhar Prakarsh ([LinkedIn](https://www.linkedin.com/in/pprakarsh04/))
