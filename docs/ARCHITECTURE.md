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
- Robust to missing values (when real InSAR and BRO soil data replace the current coordinate-based estimates, those sources are naturally sparse)
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

## Phase 2 integration points (shipped)

Phase 2 is complete. The codebase was structured so real-data integration
required changes in exactly three places:

1. `data/bag.py`, `data/insar.py`, `data/soil.py`, `data/weather.py`
   — real loaders implemented (BAG WFS; coordinate/RNG estimators for the rest)
2. `scripts/train.py` — `fetch_and_train.py` added for real-data ETL
3. Nothing else. Feature engineering, model, API, and UI were untouched.

## Why the model still trains on synthetic data

Phase 2 ETL is complete: `data/bag.py` fetches real buildings from PDOK BAG and
coordinate-based estimators fill the remaining features. The model, however, still
trains on synthetic labels because there is no public per-address ground truth for
Dutch foundation damage. Synthetic training remains the default for three reasons:

- `python scripts/train.py` is fully reproducible without credentials or API access
- CI regenerates `models/metrics.json` deterministically (RANDOM_STATE = 42)
- The synthetic labels encode documented domain patterns (peat + wooden pile + drought → risk)
  rather than memorising a one-off dataset

The transition to real supervision is the primary goal of Phase 5; see `docs/LIMITATIONS.md §L1`.

Validation against KCAF's known-damage dataset (to measure real-world predictive skill,
as opposed to the current rule-recovery metric) is Phase 5.
