# Known limitations

This document is a gap analysis produced alongside the `hardening` branch audit
(`docs/AUDIT.md`). Each limitation states what is currently true, why it matters,
and what would be needed to close the gap.

---

## L0 — Model collapses on an unseen city; random-split accuracy is not evidence of generalisation

**Current state.** On the real PDOK BAG grouped split (train: Gouda/Rotterdam/Zaanstad,
test: Dordrecht), the model scores **0.092 accuracy** (macro F1 0.142) — well below the
4-class uniform random baseline of **0.250**. This result has one root cause and two
downstream consequences; they are not three independent confounds.

**Root cause — the coordinate classifier mislabels Dordrecht as `sandy_clay`.**
`_classify_soil_by_coordinates` in `data/soil.py` uses a sequence of geographic rules
with a final catch-all `return "sandy_clay"`. Dordrecht's bbox falls through every rule:
the river-clay rule requires `lon >= 4.8` (bbox lon\_max is 4.72 — gap: **0.08 deg**,
verified by running `_classify_soil_by_coordinates` on all corners and the centroid in
this session); the peat-belt rule requires `lat >= 51.9` (bbox lat\_max is 51.83 —
gap: **0.07 deg**). Every point in the bbox hits the catch-all and is labelled `sandy_clay`.

Dordrecht is in reality a peat-and-river-clay city in the Drechtsteden and one of the
more subsidence-affected municipalities in the Netherlands. The `sandy_clay` assignment
is an artefact of where the bounding boxes were drawn, not of its geology.

Bounding-box coordinates and assigned soil types, from `config.TARGET_MUNICIPALITIES`:

| Municipality | lon min | lat min | lon max | lat max | Soil type assigned |
|---|---|---|---|---|---|
| Gouda | 4.68 | 52.00 | 4.75 | 52.03 | peat (all points) |
| Rotterdam | 4.40 | 51.88 | 4.56 | 51.96 | peat (lat ≥ 51.9) / sandy\_clay (lat < 51.9) |
| Zaanstad | 4.75 | 52.43 | 4.88 | 52.50 | peat (all points) |
| Dordrecht | 4.62 | 51.78 | 4.72 | 51.83 | **sandy\_clay — catch-all** (all points) |

**Consequence 1 — NaN encoding of the entire test set.** `soil_type = sandy_clay` is absent
from the frozen training vocabulary (all three training municipalities are peat). At test
time it encodes as NaN, and all 974 Dordrecht rows are routed to the same XGBoost
missing-value branch. This is not a generalisation failure; it is the soil classifier
assigning a label that the training pipeline never saw.

**Consequence 2 — label distribution inversion.** Because the rule engine assigns lower
risk to `sandy_clay` buildings than to `peat` buildings, Dordrecht's true label
distribution (65% `moderate`, 18% `low`, 13% `high`, 4% `critical`) is the inverse of
the training distribution (76% `critical`). The missing-value branch routes most Dordrecht
rows to `critical` (the dominant training class), yielding 100% recall on the 37
true-critical buildings but near-zero recall on the 632 true-moderate buildings.

**What the experiment actually measures.** As currently constructed the grouped split does
not measure whether the model generalises geographically. It measures what happens when
the soil classifier misclassifies the test city. Nudge the Dordrecht bbox 0.09 deg east
(lon\_max 4.72 → 4.81) and the classifier returns `clay`; nudge it 0.08 deg north
(lat\_max 51.83 → 51.91) and it returns `peat`. The 0.092 figure reflects a data pipeline
error in the soil labelling, not the model's intrinsic geographic transferability.

**Why the random-split figure (0.824) does not refute this.** The 82.4% accuracy is
achieved by recovering the synthetic rule engine on held-out rows from the same
distribution. The synthetic rule is written in terms of `soil_type`, and the same
coordinate-based classifier generates `soil_type` for both training and test rows in the
random split. The model memorises the rule correctly when the feature distribution is
identical. The held-out-city result shows that when the soil-type distribution changes —
here because the bbox placement produced the wrong label — the model collapses entirely.

**Summary of the two figures** (both produced by commands run on the `hardening` branch):

| | Command | Accuracy |
|---|---|---|
| Random split (synthetic) | `python3 scripts/train.py` | 0.824 |
| Held-out city (real data) | `python3 scripts/train.py --split grouped` | **0.092** |
| 4-class uniform baseline | — | 0.250 |

The 0.824 should be reported only as a pipeline integrity check. It is not evidence that
the model has learned transferable risk assessment. The 0.092 reflects classifier
mislabelling of the test city, not geographic out-of-distribution failure — though the
distinction matters only for interpretation; the magnitude of the collapse is the same.

**What's needed.** Two independent interventions are required to make the held-out-city
evaluation meaningful:

1. **Real soil data.** Replace the coordinate-based soil classifier with the PDOK BRO
   Bodemkaart WFS (`use_soil_api=True` in `data/soil.py`). Real soil data is not
   confounded with municipality bounding boxes; Dordrecht would receive its actual
   peat/clay classification.

2. **Real InSAR, groundwater, and drought data.** The same coordinate-based shortcut
   exists for every simulated feature. Replacing them with real measurements (BDK 2.0,
   BRO GLD monitoring wells, KNMI precipitation deficit API) breaks the municipality
   confound for all features simultaneously.

Until these are in place, the grouped split is not a valid generalisation test. It is a
test of what happens when a coordinate rule assigns the wrong soil type to the test city.

---

## L1 — Labels are synthetic; accuracy measures rule-recovery

**Current state.** There is no public per-address ground truth for Dutch foundation
damage. Labels (`low`/`moderate`/`high`/`critical`) are generated by
`synthetic._compute_risk_score`: a hand-written weighted sum of the 11 input features,
thresholded at three fixed cutoffs, with a `N(0, 0.05)` noise term.

**Why it matters.** The model's reported accuracy (82.5% on the random split) measures
how faithfully XGBoost reconstructs the rule engine. The ~17% gap from perfect is almost
entirely the noise term straddling the thresholds. This is a valid pipeline integrity
check but should not be described as evidence of real-world predictive skill. See §L0
for why the held-out-city result (0.092) shows the rule-recovery does not transfer.

**What's needed.** Weak supervision from KCAF / municipal *funderingsrisico* zone
polygons would provide the first real, defensible labels. Labelling buildings by
whether they fall in a designated high-risk zone is noisy supervision, but it is
real supervision.

---

## L2 — 9 of 11 features are simulated

**Current state.** Only `year_built` (BAG `bouwjaar`) is a directly measured feature
from a real government registry. `building_age` is computed arithmetically from
`year_built` (`2026 − year_built`) — it does not come from any data source and adds
no independent information. The building centroid (`lat`/`lon`) is also a real BAG
measurement but is not a model feature; it drives the coordinate-based estimators.
The remaining 9 model features are produced by domain-informed random number generators.
See the feature table in the README.

**Why it matters.** The model is learning from simulated measurements of simulated
labels. Domain patterns (peat + wooden pile → higher risk) are encoded correctly,
but the magnitudes, correlations, and noise levels are the product of the generator,
not of the real world.

**What's needed.** For each simulated feature, there is an existing Dutch open-data
source that provides real measurements:

| Feature | Real source |
|---------|-------------|
| `soil_type` | PDOK BRO Bodemkaart WFS (`use_soil_api=True`) |
| `peat_thickness_m` | TNO DINOloket |
| `groundwater_depth_m/variability` | BRO Grondwaterstandonderzoek (GLD) monitoring wells |
| `insar_deformation_mm_yr` | BodemDalingsKaart 2.0 WMS |
| `drought_exposure_index` | KNMI daily neerslagtekort API |
| `distance_to_water_m` | PDOK Top10NL water bodies layer |
| `neighborhood_damage_rate` | KCAF/municipal damage reports |
| `foundation_type` | KCAF FunderMaps; municipal foundation surveys |

---

## L3 — No postcode lookup in the demo

**Current state.** The FastAPI UI accepts 11 manual numeric sliders. There is no
geocoder or postcode field; the Leaflet map drop-pin does not feed lat/lon into the
model.

**Why it matters.** The original README's "Type any Dutch postcode (e.g. 2801AB)"
instruction was factually wrong and is now removed.

**What's needed.** PDOK Locatieserver provides free Dutch geocoding. The flow would
be: postcode → centroid coordinates → BAG WFS lookup → derive `year_built` → run
existing estimators for the remaining features. This is roughly one day of work and
would convert the demo from a parameter editor into a genuine address lookup tool.

---

## L4 — No probability calibration

**Current state.** The model outputs raw XGBoost softmax scores. There is no
`CalibratedClassifierCV`, isotonic regression, or Platt scaling anywhere in the
codebase.

**Why it matters.** Softmax outputs are not calibrated probabilities. `P(critical) = 0.85`
means "the model is relatively confident this is critical", not "85% of buildings
with this feature profile have critical damage". The EU AI Act Article 13 section
previously claimed "calibrated probabilities" — that claim has been corrected.

**What's needed.** Wrap the classifier in `CalibratedClassifierCV(method="isotonic", cv=5)`
and ship a reliability diagram alongside the model. This is a meaningful addition
given the compliance framing; isotonic regression typically requires at least a few
hundred positive examples per class, which the synthetic pipeline can provide.

---

## L5 — Spatial leakage in the train/test split

**Current state.** `train_test_split` is a random row split. `neighborhood_damage_rate`
is assigned per spatial hash bucket, so buildings in the same ~2 km cell share a
feature value and appear on both sides of the split.

**Why it matters.** The model has implicitly seen neighbourhood-level signal during
training that it will also see at test time — the reported accuracy is mildly
optimistic as a result. In a real deployment, you would never see a test building's
neighbourhood rate at training time.

**What's needed.** A municipality-based grouped split has been added (`--split grouped`)
and reveals the problem explicitly: the held-out-city result (0.092) is well below the
4-class uniform baseline (0.250). The gap cannot be closed without real soil, groundwater,
and InSAR data that are not confounded with municipality geography — see §L0.

---

## L6 — `low` class is underrepresented and weakly predicted

**Current state.** On the 2,000-row synthetic test set, `low` has 142 samples (7%)
and F1 = 0.618. 65 of 142 true-`low` buildings are predicted as `moderate`.

**Why it matters.** If this were a real model, a homeowner whose building is
genuinely low-risk has a 45% chance of being told it is moderate-risk. The cost of
this error is probably low (unnecessary worry, not unnecessary surgery), but it
should be acknowledged.

**What's needed.** Sample weights are now applied (`compute_sample_weight("balanced")`),
improving `low` F1 from 0.603 to 0.680. `DummyClassifier(most_frequent)` scores 49.5%
on the synthetic test set — XGBoost at 82.5% is a real improvement against that baseline.
The remaining gap is the noise term in the label rule, not class imbalance alone.

---

## L7 — Model artefact is not reproducible from the published repo

**Current state.** `models/subsidence_xgb.joblib` is `.gitignore`d. `models/metrics.json`
is committed. Anyone cloning the repo cannot verify or reproduce those numbers; the
file will silently drift if the training pipeline changes.

**What's needed.** Either: (a) regenerate `metrics.json` in CI from the synthetic
pipeline (deterministic — RANDOM_STATE=42 throughout), or (b) commit the model with
Git LFS and pin the XGBoost version so the joblib is reproducible. Option (a) is
lower friction and already possible today.

---

## L8 — `spaces/` deployment is not reproducible

**Current state.** `spaces/app.py` imports `zakkend` and loads a model from disk,
but neither the package source nor the model file is included in `spaces/`. The
`deploy_hf.py` script copies them in locally, and `.gitignore` excludes
`models/*.joblib`. A fresh clone cannot reproduce the Space.

**What's needed.** Any of: commit the trained model with Git LFS, publish it as a
HF Hub model repo and pull it at Space startup, or have the Space train on first
boot from the synthetic generator. Document which approach is used.
