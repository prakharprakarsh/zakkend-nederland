# 🏚️ Zakkend Nederland

> **ML-powered foundation subsidence risk assessment for Dutch homes.**
> An estimated **1 million Dutch homes** face foundation damage from soil subsidence,
> with total damage projected at **€60+ billion by 2050**. Homeowners currently have
> no way to know their risk until cracks appear. This project changes that.

![status](https://img.shields.io/badge/status-Phase%201%20%7C%20MVP-blue)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![eu-ai-act](https://img.shields.io/badge/EU%20AI%20Act-aware-orange)

---

## 🎯 The problem

Peat soils shrink in drought. Wooden pile foundations rot when groundwater drops.
Together, these phenomena cause slow, irreversible foundation damage across the
western Netherlands — especially in cities built on reclaimed land (Gouda, Rotterdam-Zuid,
Zaanstad, Amsterdam-Noord, Dordrecht).

Key facts that make this a real problem worth solving:

- **1,000,000+** Dutch homes at risk (KCAF, TNO estimates)
- **€60B+** projected repair cost by 2050
- **Insurers do not cover** subsidence damage
- **Mortgage banks** (ING, Rabobank, ABN AMRO) are scrambling for risk models
- **Municipalities** need prioritization tools for funding schemes
- **Homeowners** have no accessible tool to check their own address

## 🧠 The ML approach

This project combines **four data modalities** to produce a per-address risk score:

| Modality          | Source                        | Signal                                     |
| ----------------- | ----------------------------- | ------------------------------------------ |
| Satellite (InSAR) | Sentinel-1 / BodemDalingsKaart| Vertical ground deformation (mm/year)      |
| Geospatial        | PDOK BAG, TNO DINOloket       | Building footprint, year, soil composition |
| Climate           | KNMI                          | Precipitation deficit, drought indices     |
| Graph (Phase 3)   | Derived                       | Groundwater-network spatial correlations   |

**Model:** Gradient-boosted decision trees (XGBoost) with SHAP explainability for
every prediction — fully aligned with **EU AI Act Article 13** transparency
requirements for automated decisions affecting property.

## 🏗️ Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ BAG buildings│   │ TNO soil     │   │ KNMI drought │   │ InSAR        │
│ (PDOK WFS)   │   │ (DINOloket)  │   │ (KNMI API)   │   │ (BDK 2.0)    │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │                  │
       └──────────────────┴────────┬─────────┴──────────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │ Feature engineering  │
                        │ (geohash, rolling,   │
                        │  categorical encode) │
                        └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │ XGBoost risk model   │
                        │ (classification +    │
                        │  calibrated proba)   │
                        └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │ SHAP explainer       │
                        │ (per-prediction      │
                        │  feature attribution)│
                        └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │ FastAPI + Leaflet    │
                        │ (postcode → risk)    │
                        └──────────────────────┘
```

## 🚀 Quick start

```bash
# 1. Clone and enter
git clone https://github.com/prakharprakarsh/zakkend-nederland.git
cd zakkend-nederland

# 2. Create environment (Python 3.12)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 3. Generate synthetic training data (Phase 1)
python -m zakkend.data.synthetic --n 10000 --out data/processed/training.parquet

# 4. Train the baseline model
python scripts/train.py

# 5. Launch the API
uvicorn zakkend.api.main:app --reload

# 6. Open the demo
open http://localhost:8000
```

Type any Dutch postcode (e.g. `2801AB` for Gouda) and see the risk assessment.

## 📊 What the demo shows

- **Interactive map** — click any point in the Netherlands
- **Risk score** — 0–100 with calibrated probability
- **SHAP waterfall** — why the model predicted this score (year built,
  soil type, drought exposure, etc.)
- **Remediation summary** — actionable next steps for homeowners
  (inspection bodies, funding schemes like *Nationaal Fonds Funderingsherstel*)

## 🗺️ Roadmap

- [x] **Phase 1** — Synthetic-data pipeline, XGBoost baseline, SHAP, FastAPI + Leaflet demo
- [ ] **Phase 2** — Real PDOK/BAG buildings + BodemDalingsKaart InSAR ingestion
- [ ] **Phase 3** — KNMI climate features + Graph Neural Network over groundwater
- [ ] **Phase 4** — LangGraph agent drafts remediation reports; HF Spaces deployment
- [ ] **Phase 5** — Calibration study vs. KCAF known-damage dataset; model card

## 🧪 Tests

```bash
pytest -v
```

## 📚 Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design deep-dive
- [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) — all sources, licenses, refresh cadence
- [`docs/EU_AI_ACT.md`](docs/EU_AI_ACT.md) — transparency & risk-category notes

## 🤝 Contributing

This is a portfolio project by [Prakhar Prakarsh](https://www.linkedin.com/in/pprakarsh04/).
If you work in Dutch housing/insurance/municipal data and spot something wrong,
issues and PRs are very welcome.

## 📜 License

MIT. See [LICENSE](LICENSE).

## 🙏 Acknowledgments

Data sources and domain knowledge informed by:

- [KCAF (Kenniscentrum Aanpak Funderingsproblematiek)](https://www.kcaf.nl/)
- [TNO DINOloket](https://www.dinoloket.nl/)
- [BodemDalingsKaart.nl](https://bodemdalingskaart.nl/)
- [PDOK (Publieke Dienstverlening Op de Kaart)](https://www.pdok.nl/)
- [KNMI](https://www.knmi.nl/)
