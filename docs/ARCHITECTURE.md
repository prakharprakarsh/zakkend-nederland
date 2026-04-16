# Architecture

## System overview

Zakkend Nederland is a **feature-pipeline-first** system. Data from four modalities
is harmonized into a single tabular schema, a gradient-boosted tree model scores
each address, and SHAP decomposes the score into feature-level contributions
consumed by a FastAPI service.

## Components

### 1. Data layer (`src/zakkend/data/`)

| Module         | Responsibility                                        | Phase |
| -------------- | ----------------------------------------------------- | ----- |
| `synthetic.py` | Generates labelled training data encoding NL dynamics | 1     |
| `bag.py`       | PDOK BAG (buildings) loader                           | 2     |
| `soil.py`      | TNO DINOloket soil/peat thickness                     | 2     |
| `insar.py`     | BodemDalingsKaart InSAR deformation                   | 2     |
| `weather.py`   | KNMI drought & precipitation deficit                  | 3     |

### 2. Features (`src/zakkend/features/`)

The schema defined in `config.FEATURE_COLUMNS` is the contract the whole system
revolves around — training, inference, and API validation all read it.
Categorical fields (`foundation_type`, `soil_type`) are handled by XGBoost's
native `enable_categorical=True`, which avoids brittle one-hot encoding and keeps
SHAP explanations human-readable.

### 3. Model (`src/zakkend/models/baseline.py`)

XGBoost `multi:softprob` over 4 ordinal risk classes. Chosen because:

- Handles mixed numeric/categorical cleanly
- Robust to missing values (real PDOK/InSAR data will be sparse)
- Tree-SHAP is exact and fast — critical for real-time explanation API
- Widely deployed in Dutch banks' credit models, so explainability stories
  translate directly to recruiter-relevant contexts

### 4. Explainability (`src/zakkend/explain/`)

Every `/explain` call returns the top-5 SHAP drivers with direction and value.
This is the **EU AI Act Article 13** compliance layer: automated decisions
affecting property are explainable per-instance.

### 5. API (`src/zakkend/api/`)

FastAPI for two reasons: Pydantic v2 gives us range-validated inputs for free,
and automatic OpenAPI docs (`/docs`) make the system easy for recruiters to
click through.

## Phase 2 hook points

The codebase is deliberately structured so Phase 2 requires changes in exactly
three places:

1. `data/bag.py`, `data/insar.py`, `data/soil.py`, `data/weather.py`
   — implement real loaders (they already exist as stubs)
2. `scripts/train.py` — swap `generate()` for a real-data ETL
3. Nothing else. Feature engineering, model, API, and UI remain untouched.

## Why synthetic data for Phase 1

Recruiters want to see end-to-end ML systems that work. Starting with
synthetic data that encodes published domain dynamics means:

- The pipeline is fully testable before real data arrives
- A reproducible demo works on every machine without credentials
- The model learns transferable patterns (peat + wooden pile + drought → risk)
  rather than memorising a one-off dataset

Real-data calibration (vs. KCAF's known-damage dataset) is Phase 5.
