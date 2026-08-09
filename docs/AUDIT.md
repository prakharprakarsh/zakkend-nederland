# zakkend-nederland — audit & fix plan

Audited at commit `f1ddfbb` (main, 3 commits, last pushed 2026-04-17). All findings below were
reproduced by cloning the repo, installing it, training a model, and calling the code — not read off
the README.

Test suite result: **55 passed**. That claim holds up.

---

## P0 — Fix before anyone else sees this repo

### 0.1 🔴 The demo silently ignores foundation type and soil type

**This is the one that ends an interview.** `features/engineering.py::build_feature_matrix` calls
`.astype("category")` on whatever DataFrame it receives. At training time the frame has all 4–5
levels. At inference time the API passes a **single row**, so the category dtype has exactly one
level → code `0`. XGBoost 2.x matches categoricals by code, not by name.

Reproduced on `xgboost==2.1.4` (which `xgboost>=2.0` in your `requirements.txt` resolves to):

```
API path (1-row frame), varying ONLY foundation_type:
  wooden_pile    critical  [0.  0.008 0.411 0.582]
  concrete_pile  critical  [0.  0.008 0.411 0.582]   ← identical
  strip          critical  [0.  0.008 0.411 0.582]   ← identical
  slab           critical  [0.  0.008 0.411 0.582]   ← identical
  "banana"       critical  [0.  0.008 0.411 0.582]   ← no error raised

Correct path (proper category dtype):
  wooden_pile    critical  [0. 0. 0.001 0.999]
  concrete_pile  high      [0. 0. 0.590 0.410]
  strip          critical  [0. 0. 0.270 0.730]
  slab           high      [0. 0. 0.602 0.398]
```

Consequences:
- Every prediction from the FastAPI demo and the Streamlit Space is wrong.
- The SHAP output is computed on the same broken encoding, so your **EU AI Act Article 13
  "transparency"** artefact is misattributing contributions.
- `wooden_pile` and `peat` are the two heaviest terms in your own risk formula (0.35 and 0.25).
  The model is deaf to exactly the features the whole project is about.

**Fix.** Freeze the category vocabulary at train time and reapply it at inference.

`src/zakkend/models/baseline.py` — add to `TrainedModel`:

```python
@dataclass
class TrainedModel:
    classifier: xgb.XGBClassifier
    feature_columns: list[str]
    categorical_features: list[str]
    class_names: list[str]
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_ready = build_feature_matrix(X, category_levels=self.category_levels)
        return self.classifier.predict_proba(X_ready)
```

In `train(...)`, before fitting:

```python
category_levels = {
    col: sorted(df[col].astype(str).unique())
    for col in config.CATEGORICAL_FEATURES
}
X = build_feature_matrix(df, category_levels=category_levels)
```

and pass `category_levels=category_levels` into the returned `TrainedModel`.

`src/zakkend/features/engineering.py`:

```python
class UnknownCategoryError(ValueError):
    """Raised when an inference-time category was never seen in training."""


def build_feature_matrix(
    df: pd.DataFrame,
    category_levels: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    features = df[config.FEATURE_COLUMNS].copy()
    for col in config.CATEGORICAL_FEATURES:
        values = features[col].astype(str)
        if category_levels and col in category_levels:
            allowed = category_levels[col]
            unseen = sorted(set(values) - set(allowed))
            if unseen:
                raise UnknownCategoryError(
                    f"{col}: {unseen} not in training vocabulary {allowed}"
                )
            features[col] = pd.Categorical(values, categories=allowed)
        else:
            features[col] = values.astype("category")
    return features
```

`src/zakkend/api/main.py` — return 422, not a 500 stack trace:

```python
from zakkend.features.engineering import UnknownCategoryError

@app.exception_handler(UnknownCategoryError)
async def unknown_category_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
```

**Regression test that must fail before the fix and pass after** — put this in `tests/test_api.py`:

```python
def test_categorical_features_actually_change_the_prediction(client, base_payload):
    """Guards against train/serve category-encoding skew."""
    results = []
    for foundation in ["wooden_pile", "concrete_pile", "strip", "slab"]:
        payload = {**base_payload, "foundation_type": foundation}
        results.append(tuple(
            round(p["probability"], 6)
            for p in client.post("/predict", json=payload).json()["class_probabilities"]
        ))
    assert len(set(results)) > 1, "foundation_type has no effect — categorical encoding is broken"
```

---

### 0.2 🔴 The model is trained to imitate your own formula

`scripts/train.py::_label_real_data` labels every "real" building by calling
`synthetic._compute_risk_score` — a hand-written weighted sum of the *same 11 features* the model
then receives as input, plus `N(0, 0.05)` noise and three fixed thresholds.

So `accuracy = 0.833` does not mean "the model predicts subsidence risk at 83%". It means
"XGBoost recovered my own if-statements 83% of the time" — which, for a deterministic function of
the inputs, is **underperformance, not a result**. The 17% gap is just the noise term straddling
your thresholds.

Any reviewer at ING or Rabobank will spot this in under two minutes, and right now the README
presents it as a model result.

You have three honest options, in ascending order of effort and payoff:

| Option | Effort | What it buys you |
|---|---|---|
| **A. Reframe** — call it a rule-engine distillation / demo harness, put the metric in context | 1 hour | Stops the credibility bleed immediately |
| **B. Weak labels** — scrape KCAF `FunderMaps` / municipal *funderingsrisico* zone polygons, label buildings by zone, treat as noisy supervision | 1–2 weeks | A real, defensible supervised setup |
| **C. Unsupervised** — drop classification, ship a transparent scored index + uncertainty | 3–4 days | Honest, and arguably the *right* answer given no ground truth |

Do **A this week** regardless. It is a two-paragraph README edit and it converts your biggest
liability into evidence of judgement:

> **On labels and metrics.** There is no public per-address ground truth for Dutch foundation
> damage. Labels here are generated by an explicit, documented rule engine
> (`synthetic._compute_risk_score`), so the reported 83% accuracy measures how well a gradient-boosted
> model recovers that rule — it is a pipeline-integrity check, **not** evidence of real-world
> predictive skill. The next milestone is weak supervision from KCAF zone data; see
> `docs/LIMITATIONS.md`.

Add a `DummyClassifier(strategy="stratified")` baseline next to it. Right now there's nothing to
compare 83% against.

---

### 0.3 🔴 "Trained on real PDOK BAG data" is 2 features out of 11

Your HF Space card says *"Predicts foundation risk class using XGBoost (trained on real PDOK BAG
data)"*. What actually comes from PDOK is `identificatie`, `bouwjaar`, and the polygon centroid.
The other nine features are generated with `np.random.default_rng` in `pipeline.py`, `soil.py`,
`insar.py`, and `weather.py`:

| Feature | Source in your code |
|---|---|
| `year_built` | ✅ real (BAG `bouwjaar`) |
| `lat` / `lon` | ✅ real (BAG geometry centroid) |
| `soil_type` | ⚠️ coordinate rules + RNG (BRO API exists but `use_soil_api=False` by default) |
| `peat_thickness_m` | ❌ `rng` |
| `groundwater_depth_m` | ❌ `rng.normal(...)` by soil type |
| `groundwater_variability` | ❌ `rng.beta(...)` |
| `insar_deformation_mm_yr` | ❌ `rng.normal(...)` by bounding box |
| `drought_exposure_index` | ❌ `rng` |
| `distance_to_water_m` | ❌ `rng.exponential(...)` |
| `neighborhood_damage_rate` | ❌ `rng.beta(...)` per spatial hash |
| `foundation_type` | ❌ `rng.choice(...)` by era + soil |

The estimators are *sensible* — the bounding boxes and priors are defensible domain work. The
problem is purely that you call them "real data" and "calibrated" where the code says
`default_rng`. Rename the module functions `simulate_*` instead of `estimate_*`/`enrich_*`,
put the table above in the README, and the same work now reads as careful synthetic-data
engineering rather than as an overclaim.

---

### 0.4 🔴 Following your own README fails

README quick start:

> Type any Dutch postcode (e.g. `2801AB` for Gouda) and see the risk assessment.

There is no postcode field anywhere in `src/zakkend/api/static/index.html`, and no geocoding module
in the repo. The UI is 11 manual numeric inputs. The Leaflet map drops a marker on click but
`lat`/`lon` aren't model features, so the marker changes nothing.

A recruiter who clones and follows step 6 finds the README lying to them in the first 30 seconds,
and then discounts everything else — including the parts that are genuinely good.

**Either** build the postcode lookup (PDOK Locatieserver free geocoding → BAG lookup → derive
`year_built` + coords → run your existing estimators for the rest) **or** delete the claim today.
Deleting takes 60 seconds. The lookup is maybe a day and would be a genuinely strong demo.

---

### 0.5 🔴 The README contradicts the repository

- Roadmap shows Phase 2, 3, 4 **unchecked**. Commits show Phase 2 and Phase 4 shipped; Phase 3 was
  skipped entirely.
- Phase 4 is titled "HF Spaces deployment" and there is **no link to the live Space anywhere** —
  not in the README, not in the GitHub sidebar. If it's live, it's invisible.
- GitHub sidebar: *"No description, website, or topics provided."* Add the description, the Space
  URL as the homepage, and topics (`machine-learning`, `xgboost`, `shap`, `geospatial`,
  `netherlands`, `explainable-ai`, `fastapi`, `mlops`). This is 2 minutes and it's how your repo
  gets found at all.
- `pyproject.toml` author email is `prakhar@example.com`.

---

### 0.6 🟠 The commit history reads as one AI session

```
4cb7386  2026-04-16 21:20  Phase 1  (1,825 lines, 33 files)
968025b  2026-04-17 11:56  Phase 2  (1,470 lines, 11 files)
f1ddfbb  2026-04-17 12:32  Phase 4  (1,299 lines,  9 files)   ← 36 minutes after Phase 2
```

4,594 lines in three commits over fifteen hours, then nothing for four months. Using Claude Code
to build this is completely fine and increasingly normal — but *this shape* signals "generated,
never lived in", which is exactly the doubt you don't want when your differentiator is engineering
judgement.

You can't rewrite history credibly, and you shouldn't try. What you can do:
- Make every fix in this document a **separate, small, well-messaged commit**. Twenty commits over
  three weeks on top of three big ones tells a fine story: prototyped fast, then hardened.
- Open GitHub Issues for the items you're not fixing yet (0.2 option B, Phase 3). An issue tracker
  with real self-filed issues is strong evidence of ownership.
- The commit fixing 0.1 is your best asset. Message it properly:
  `fix(inference): freeze categorical vocabulary — 1-row frames silently encoded every category as 0`

---

## P1 — Engineering issues a senior reviewer will list

### 1.1 Unpinned dependencies + pickled model = a Space that breaks on rebuild

`requirements.txt` is all `>=`. `TrainedModel.save()` `joblib.dump`s the whole dataclass, which
pickles the `xgb.XGBClassifier` object. Two problems compound:

- HF Spaces rebuilds resolve fresh versions. `xgboost>=2.0` gave you 2.x in April; it resolves to
  3.4 today. **Categorical handling changed between them** (2.x matches by code, 3.x by name and
  raises on unknown values). Your Space's predictions can change on a rebuild with no code change.
- Pickled sklearn/xgboost estimators are not guaranteed loadable across versions. A rebuild can
  hard-fail on `joblib.load`.

Fix:
```bash
pip freeze > requirements.lock.txt   # commit this; keep requirements.txt as loose dev ranges
```
and change persistence to the portable format:
```python
def save(self, path: Path | None = None) -> None:
    path = Path(path or config.MODEL_PATH).with_suffix(".ubj")
    self.classifier.save_model(path)                       # xgboost's own format
    path.with_suffix(".meta.json").write_text(json.dumps({
        "feature_columns": self.feature_columns,
        "categorical_features": self.categorical_features,
        "category_levels": self.category_levels,
        "class_names": self.class_names,
        "metrics": self.metrics,
        "xgboost_version": xgb.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
```
Also: pickle deserialisation is arbitrary code execution. Loading a `.joblib` you didn't produce is
a real supply-chain risk, and saying so in your README is a free EU-AI-Act-Article-15 point.

### 1.2 Test set leaks into training

`baseline.py:87` — `clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])`. You're monitoring on
the set you report. There's no validation split and no `early_stopping_rounds`, so the `eval_set`
isn't even buying you anything. Split three ways (train/val/test), early-stop on val, report test.

### 1.3 Spatial leakage in the split

`train_test_split(..., stratify=y, random_state=42)` is a random row split. But
`neighborhood_damage_rate` is assigned per spatial hash bucket (`pipeline.py:147-163`), so
neighbouring buildings share a feature value *and* land on both sides of the split. Your 83% is
optimistic for that reason too.

Use `GroupKFold` / `GroupShuffleSplit` grouped by `municipality`, or hold out one city entirely
(train Gouda + Rotterdam + Zaanstad, test Dordrecht). Report both numbers. "Random split 0.83,
held-out-city 0.71" is *far* more impressive than 0.83 alone — it shows you know which number is
the honest one.

### 1.4 "Calibrated probabilities" is claimed three times and implemented zero times

The phrase appears in `README.md` (architecture box: "classification + calibrated proba"),
`docs/EU_AI_ACT.md` (Article 13 section), and the API docstring. There is no
`CalibratedClassifierCV`, no isotonic regression, no Platt scaling, no reliability diagram in the
repo. Softmax outputs are not calibrated probabilities.

Either wrap the classifier in `CalibratedClassifierCV(method="isotonic", cv=5)` and ship a
reliability plot — genuinely a strong, cheap addition given the compliance framing — or delete the
word from all three places.

### 1.5 API accepts contradictory and unvalidated inputs

Reproduced:
```
POST /predict {"year_built": 2020, "building_age": 300, ...}  → 200 OK, "critical"
POST /predict {"foundation_type": "banana", ...}              → 500 (uncaught XGBoostError on 3.x)
```

- `building_age` should never be a user input. Derive it: `building_age = current_year - year_built`.
  Drop it from `BuildingInput` entirely.
- `foundation_type` / `soil_type` are typed `str`. Use `Literal` so FastAPI returns a clean 422 and
  the values show up in the OpenAPI schema:
  ```python
  from typing import Literal
  foundation_type: Literal["wooden_pile", "concrete_pile", "strip", "slab"]
  soil_type: Literal["peat", "clay", "sandy_clay", "sand", "loess"]
  ```
- Hardcoded `2026` in `pipeline.py:176` (`result["building_age"] = 2026 - result["year_built"]`) —
  use `datetime.now().year`, or better, a `REFERENCE_YEAR` constant so training is reproducible.

### 1.6 RNG seed collisions produce duplicate "measurements"

`insar.py:50` — `np.random.default_rng(int(abs(lat * 10000) + abs(lon * 10000)))`.
Same pattern in `weather.py:46` and `soil.py:89`.

Because it's a **sum**, distinct coordinates collide:

```
lat 52.0100, lon 4.7000 → 520100 + 47000 = 567100
lat 52.0000, lon 4.7100 → 520000 + 47100 = 567100   ← same seed, identical InSAR value
```

Two buildings 1.1 km apart get byte-identical "satellite measurements". Use a proper hash:
```python
seed = int(hashlib.blake2b(f"{lat:.6f},{lon:.6f}".encode(), digest_size=8).hexdigest(), 16) % (2**32)
```

### 1.7 No CI — nobody can see your 55 tests pass

This is the highest-value-per-minute item in the whole document. Create
`.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -r requirements.txt -e ".[dev]"
      - run: ruff check src tests
      - run: black --check src tests
      - run: pytest --cov=zakkend --cov-report=term-missing --cov-fail-under=70
```

Then put the badge at the top of the README, above the hand-written `status-Phase 1 | MVP` badge
(which is itself now stale — you're on Phase 4). A green CI badge is worth more than all four
shields.io badges you currently have combined, because it's the only one a recruiter can't fake.

### 1.8 `spaces/` as committed is not deployable

`spaces/` contains only `app.py`, `requirements.txt`, `README.md`. No `src/`, no model —
`deploy_hf.py` copies those in locally, and `.gitignore` excludes `models/*.joblib`. So nobody
(including future you on a new laptop) can reproduce the Space from this repo.

Options: commit the trained model with Git LFS, publish it to the HF Hub as a model repo and pull
it at Space startup, or have the Space train on first boot from the synthetic generator. Any of the
three; just make it reproducible and say which in the README.

### 1.9 Requirements are wrong in both directions

- `requirements.txt` ships `streamlit` and `langgraph` for the API service — the FastAPI container
  doesn't need either.
- It does **not** ship `pytest` or `httpx`, so the README's `pytest -v` step fails on a clean
  install. `pip install -e ".[dev]"` is the missing instruction.

Split into `requirements.txt` (API), `requirements-spaces.txt` (already exists as
`spaces/requirements.txt`), and the `dev` extra.

### 1.10 `models/metrics.json` is committed; the model that produced it is gitignored

Nobody can verify or reproduce those numbers, and the file will silently drift from reality. Either
regenerate it in CI from the synthetic pipeline (deterministic — you seed everything with 42, so
this works) or move it into a proper model card.

### 1.11 Performance and style
- `pipeline.py` `_estimate_groundwater` and `_estimate_foundation` loop with `.iterrows()` over up
  to 20,000 rows. Both are trivially vectorisable with `np.select` / `pd.Series.map`. Reviewers
  read `.iterrows()` as a junior tell.
- `synthetic.py::_sample_soil` and `_sample_foundation` loop `rng.choice` per row for the same
  reason.
- `_estimate_groundwater`, `_estimate_foundation`, and `_estimate_spatial_features` each call
  `default_rng(config.RANDOM_STATE)` — same seed, same stream, three times. Not a bug, but it
  creates correlated draws across supposedly independent features. Use one `rng` threaded through.

### 1.12 Class imbalance is unreported
`low` has 44 test samples and F1 = 0.50 — half of `low` buildings are misclassified. It's in
`metrics.json` but nowhere in the README, which shows only the headline. Add per-class metrics and
the confusion matrix to the README, and set `sample_weight` or `scale_pos_weight`. Showing your
weakest class *first* is a strong signal; hiding it is the opposite.

---

## P2 — `docs/EU_AI_ACT.md` is your differentiator and it's overclaiming

This document is genuinely the smartest thing in the repo — the Annex III "grey zone" reasoning
around creditworthiness adjacency is exactly the thinking a Dutch fintech wants. Which is why the
false claims in it hurt disproportionately.

| Claim in the doc | Reality |
|---|---|
| "Every `/explain` call returns... calibrated probabilities" | No calibration exists (§1.4) |
| "Pydantic range validation blocks out-of-distribution inputs" | Reproduced a 200 OK on `year_built=2020, building_age=300`, and a 500 on an unknown category (§1.5) |
| "satisfies the substantive core of Article 13" | Assertive for a self-audit; and the explanations are currently computed on broken encodings (§0.1) |
| "Article 10 — Data governance... only open data with documented licences" | The labels are a hand-written formula and 9/11 features are RNG (§0.2, §0.3) |

Rewrite as **a gap analysis, not a compliance claim**. Two columns: "what the Act requires" /
"what this project currently does, and what's missing". That's what an actual AI-governance
engineer produces, and it's strictly more impressive than a green checklist. Add the model card
(`docs/MODEL_CARD.md`) now rather than deferring it to "Phase 5" — you have everything you need for
it except the honest limitations section, which this audit just wrote for you.

---

## P3 — Domain credibility with a Dutch reviewer

Your headline numbers are correct and correctly attributed: KCAF does estimate ~1 million homes at
risk (one in four built before 1970) and up to €60 billion in damage by 2050.

But KCAF's more recent *Stand van het Land* narrows the count to roughly 487,000–537,000 buildings
at elevated foundation risk, with repair costs toward 2050 estimated around €55 billion. A reviewer
at ING or ABN AMRO may well know the newer figure and read the €60bn/1M framing as dated.

Cite both, with dates. "KCAF's 2018 estimate of ~1M homes / €60bn has since been refined to
487–537k buildings at elevated risk (~€55bn); this project targets the latter population" turns a
potential gotcha into evidence you actually follow the literature. Add inline source links — right
now the numbers are in the README and the sources are in Acknowledgments with no connection between
them.

---

## Suggested order of work

**Day 1 (~3 hours) — stop the bleeding**
1. Fix the categorical encoding bug (§0.1) + add the regression test. Retrain, redeploy the Space.
2. Add the CI workflow and badge (§1.7).
3. GitHub repo description, topics, homepage → Space URL (§0.5).
4. Fix `prakhar@example.com` (§0.5).
5. Delete the postcode claim from the README, or open an issue to build it (§0.4).

**Week 1 — honesty pass**
6. README: labels & metrics disclaimer, real-vs-simulated feature table, per-class metrics, Dummy
   baseline (§0.2, §0.3, §1.12).
7. Rewrite `EU_AI_ACT.md` as a gap analysis; add `MODEL_CARD.md` and `LIMITATIONS.md` (§P2).
8. Fix roadmap checkboxes; update the stale `status-Phase 1 | MVP` badge (§0.5).
9. Remove "calibrated" everywhere, or implement calibration (§1.4).

**Week 2 — engineering hardening**
10. Pin dependencies, switch to `.ubj` + metadata JSON persistence (§1.1).
11. Three-way split with early stopping; add held-out-city evaluation (§1.2, §1.3).
12. `Literal` types, derive `building_age`, 422 handler (§1.5).
13. Fix the seed collisions (§1.6); vectorise the `.iterrows()` loops (§1.11).
14. Make `spaces/` reproducible; split requirements; add a Dockerfile (§1.8, §1.9).

**Week 3+ — the thing that makes it a portfolio piece**
15. Weak supervision from KCAF / municipal *funderingsrisico* zone data (§0.2 option B). This is
    the single change that turns "nice pipeline" into "he found real labels for a problem with no
    labels" — which is a much better interview story than the Airflow + Terraform layer you were
    planning to add on top.

One closing note: the *architecture* here is fine. Module boundaries, the config module, the
dataclass model wrapper, the test coverage, the choice of problem — all solid, and the domain
research behind the bounding boxes and foundation-era priors is real work. Almost every finding
above is a claim-vs-code mismatch or an inference-path bug, not a design failure. That's a good
position to be fixing from.
