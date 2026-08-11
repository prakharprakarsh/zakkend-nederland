# Known limitations

This document is a gap analysis produced alongside the `hardening` branch audit
(`docs/AUDIT.md`). Each limitation states what is currently true, why it matters,
and what would be needed to close the gap.

---

## L0 — Grouped-split accuracy: 0.092 (before BRO) → 0.614 (after BRO); held-out-city evaluation is not a valid generalisation test

**Current state.** On the real PDOK BAG grouped split (train: Gouda/Rotterdam/Zaanstad,
test: Dordrecht), the model scored **0.092 accuracy** (macro F1 0.142) before BRO Bodemkaart
integration — well below the 4-class uniform random baseline of **0.250**. After integrating
the BRO Bodemkaart GeoPackage spatial join, accuracy improved to **0.614** (macro F1 0.480),
above all baselines. This result has one root cause and two downstream consequences; they are
not three independent confounds.

**Root cause — the coordinate classifier mislabels Dordrecht as `sandy_clay`.**
`_classify_soil_by_coordinates` in `data/soil.py` uses a sequence of geographic rules
with a final catch-all `return "sandy_clay"`. Dordrecht's bbox falls through every rule:
the river-clay rule requires `lon >= 4.8` (bbox lon_max is 4.72 — gap: **0.08 deg**,
verified by running `_classify_soil_by_coordinates` on all corners and the centroid in
this session); the peat-belt rule requires `lat >= 51.9` (bbox lat_max is 51.83 —
gap: **0.07 deg**). Every point in the bbox hits the catch-all and is labelled `sandy_clay`.

Dordrecht is in reality a peat-and-river-clay city in the Drechtsteden and one of the
more subsidence-affected municipalities in the Netherlands. The `sandy_clay` assignment
is an artefact of where the bounding boxes were drawn, not of its geology.

Bounding-box coordinates and assigned soil types, from `config.TARGET_MUNICIPALITIES`:

| Municipality | lon min | lat min | lon max | lat max | Soil type assigned |
|---|---|---|---|---|---|
| Gouda | 4.68 | 52.00 | 4.75 | 52.03 | peat (all points) |
| Rotterdam | 4.40 | 51.88 | 4.56 | 51.96 | peat (lat ≥ 51.9) / sandy_clay (lat < 51.9) |
| Zaanstad | 4.75 | 52.43 | 4.88 | 52.50 | peat (all points) |
| Dordrecht | 4.62 | 51.78 | 4.72 | 51.83 | **sandy_clay — catch-all** (all points) |

**The before and after figures are not on the same test labels.** The rule engine derives labels
from `soil_type`; changing Dordrecht's soil classifications shifts the test-set class distribution:
`low` 177 → 103, `moderate` 632 → 640, `high` 128 → 175, `critical` 37 → 56. The 0.092 and
0.614 figures measure different classification problems.

**Consequence 1 — NaN encoding of the entire test set [RESOLVED by BRO integration].**
Before BRO, `soil_type = sandy_clay` was absent from the frozen training vocabulary (all three
training cities peat-dominant). At test time it encoded as NaN, routing all 974 Dordrecht rows
to the XGBoost missing-value branch. After BRO, `sandy_clay` enters the training vocabulary
from Zaanstad (167 buildings) and `clay` from 132 buildings across training cities; neither
encodes as NaN.

**Consequence 2 — label distribution inversion [PARTIALLY RESOLVED].** The rule engine assigns
lower risk to `sandy_clay` than to `peat`. Before BRO, all 974 Dordrecht rows were `sandy_clay`,
producing 65% `moderate`/18% `low` labels — the inverse of the training distribution (76%
`critical`). After BRO, 483 of 974 Dordrecht buildings receive `clay` from the BRO spatial join
(Zeekleigronden / Rivierkleigronden, consistent with Dordrecht's actual Holocene geology). The
remaining 491 receive `sandy_clay` via the coordinate-rule fallback (urban hardscape not covered
by the 1:50,000 rural map). **Dordrecht's soil is approximately 50% real (483 BRO) and 50%
coordinate-rule (491 fallback); the 0.614 figure is computed on partially real soil.**

**`low` class: 14 of 103 true-`low` buildings correctly identified; 183 total buildings
predicted as `low` (precision 7.7%, recall 13.6%, F1 9.8%).** The headline macro F1 of 0.480
is carried almost entirely by `moderate` (F1 75.8%); the three remaining classes average
F1 0.387. **Majority-class reference point** (not a legitimate baseline — requires test labels):
640/974 Dordrecht buildings are `moderate` = **0.657**. A constant `moderate` predictor would
outperform the model's 0.614 on this test set; the model adds no value over the majority class
on this particular held-out city.

**What the experiment now measures.** After BRO integration the grouped split is a partial
OOD test: soil type is now partially real (BRO for 49.6% of Dordrecht buildings), but the
remaining 8 features (InSAR deformation, groundwater depth/variability, drought exposure,
distance to water, neighbourhood damage rate, foundation type, peat thickness) are still
derived from coordinate-based estimators confounded with municipality geography. The 0.614
accuracy should not be read as evidence of real-world generalisation; it reflects improved
data quality for one of eleven features.

**Why the random-split figure (0.824) does not refute this.** The 82.4% accuracy is
achieved by recovering the synthetic rule engine on held-out rows from the same
distribution. The model memorises the rule correctly when the feature distribution is
identical. The held-out-city result shows the limit of that rule-recovery when even one
feature's distribution (soil type) changes across cities.

**Summary of results** (both produced by commands run on the `hardening` branch):

| | Command | Accuracy | Macro F1 | Notes |
|---|---|---|---|---|
| Random split (synthetic) | `python3 scripts/train.py` | 0.824 | 0.797 | — |
| Held-out city — before BRO | `python3 scripts/train.py --split grouped` | 0.092 | 0.142 | All 974 rows NaN-encoded |
| Held-out city — after BRO | `python3 scripts/train.py --split grouped` | **0.614** | **0.480** | Different label set; 640/974 `moderate` (0.657) |
| 4-class uniform baseline | — | 0.250 | — | — |

The 0.824 should be reported only as a pipeline integrity check. It is not evidence that
the model has learned transferable risk assessment. The before/after grouped-split rows use
different test-set label distributions; they are not measuring the same classification problem.

**What's needed.** To make the held-out-city evaluation a valid generalisation test:

1. **Real soil data** ✅ **Partially done.** BRO Bodemkaart GeoPackage spatial join now
   provides real soil labels for 1,031 of 3,828 buildings (26.9%). The remaining 2,797
   (73.1%) fall back to the coordinate rule — building centroids in urban hardscape and
   infrastructure zones not covered by the 1:50,000 rural map. Per-municipality BRO match
   rate: Dordrecht 483/974 (49.6%), Zaanstad 383/975 (39.3%), Rotterdam 91/932 (9.8%),
   Gouda 74/947 (7.8%).

2. **Real InSAR, groundwater, and drought data.** The same coordinate-based shortcut
   exists for every other simulated feature. Replacing them with real measurements (BDK 2.0,
   BRO GLD monitoring wells, KNMI precipitation deficit API) would break the municipality
   confound for all features simultaneously.

Until these are in place, the grouped split is not a valid generalisation test. The 0.614
figure reflects partial data-quality improvement for one feature, not geographic OOD
performance.

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
for why the held-out-city result (0.614 after BRO integration) is still not a valid
generalisation test.

**What's needed.** Weak supervision from KCAF / municipal *funderingsrisico* zone
polygons would provide the first real, defensible labels. Labelling buildings by
whether they fall in a designated high-risk zone is noisy supervision, but it is
real supervision.

---

## L2 — 8 of 11 features fully simulated; soil_type is hybrid (26.9% BRO, 73.1% coord-rule)

**Current state.** Only `year_built` (BAG `bouwjaar`) is a directly measured feature
from a real government registry. `building_age` is computed arithmetically from
`year_built` (`2026 − year_built`) — it does not come from any data source and adds
no independent information. The building centroid (`lat`/`lon`) is also a real BAG
measurement but is not a model feature; it drives the coordinate-based estimators.
`soil_type` is now a hybrid: 1,031 of 3,828 buildings (26.9%) receive a real BRO
Bodemkaart classification from a spatial join against the PDOK GeoPackage; the remaining
2,797 (73.1%) fall back to the coordinate rule (urban hardscape not covered by the
1:50,000 rural map). The cache parquet (data/raw/bro_soil_cache.parquet) contains 1,128
unique coordinate entries — 97 more than matched in the pipeline, because coordinate
rounding differs slightly between the GeoPackage spatial join and the BAG WFS centroids.
Per-municipality BRO match rate: Dordrecht 483/974 (49.6%), Zaanstad 383/975 (39.3%),
Rotterdam 91/932 (9.8%), Gouda 74/947 (7.8%). The remaining 8 model features are produced
by domain-informed random number generators. See the feature table in the README.

**Why it matters.** The model is learning from simulated measurements of simulated
labels. Domain patterns (peat + wooden pile → higher risk) are encoded correctly,
but the magnitudes, correlations, and noise levels are the product of the generator,
not of the real world.

**What's needed.** For each simulated feature, there is an existing Dutch open-data
source that provides real measurements:

| Feature | Real source |
|---------|-------------|
| `soil_type` | ✅ Partially done: BRO Bodemkaart GeoPackage spatial join (26.9% of buildings, 1,031/3,828). Remainder needs BRO SoilInvestigation API or higher-resolution urban soil mapping. |
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
and reveals the soil-classifier root cause: before BRO integration the held-out-city result
(0.092) was well below the 4-class uniform baseline (0.250); after BRO integration it
improved to 0.614 (macro F1 0.480). The remaining gap requires real groundwater, InSAR, and
drought data not confounded with municipality geography — see §L0.

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
