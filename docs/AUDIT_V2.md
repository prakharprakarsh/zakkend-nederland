# zakkend-nederland — audit v2

Audited at `7dbe1d4` (main, 15 commits). Cloned fresh, installed, and ran the code — same method
as the first audit.

**Verdict on the hardening pass: it worked.** Every P0 and every P1 item I could verify without
your local `real_data.parquet` is genuinely fixed, not papered over. The categorical vocabulary is
frozen and `build_feature_matrix` has no silent fallback. `enable_categorical=True` is on all four
`DMatrix` call sites. Sample weights use the full 4-class vocabulary so an absent class gets zero
weight rather than infinite. The `evaluation_notes` block that records unseen test categories is
better than what I specified — it stores the training vocabulary alongside the unseen values, so
the record is self-describing.

Below: one finding that changes the meaning of your headline number, then what's actually left.

---

## The main finding — your 0.092 is a soil-classifier bug, not geographic shift

`soil.py::_classify_soil_by_coordinates` is a hand-written if-chain over lat/lon boxes. I ran it
across all four municipality bounding boxes from `config.TARGET_MUNICIPALITIES`:

```
Gouda      lat[52.00,52.03] lon[4.68,4.75] -> ['peat']
Rotterdam  lat[51.88,51.96] lon[4.40,4.56] -> ['peat', 'sandy_clay']
Zaanstad   lat[52.43,52.50] lon[4.75,4.88] -> ['peat']
Dordrecht  lat[51.78,51.83] lon[4.62,4.72] -> ['sandy_clay']
```

Dordrecht's entire bounding box misses **every** rule in the chain and falls through to the final
`return "sandy_clay"` — the catch-all default, not a deliberate classification. Two near-misses:

| Rule | Condition | Dordrecht | Gap |
|---|---|---|---|
| River clay (Betuwe/Rivierenland) | `51.75 <= lat <= 52.05` **and** `4.8 <= lon <= 6.0` | lat ✅, lon max 4.72 | misses by **0.08°** (~5.5 km) |
| Groene Hart peat belt | `51.9 <= lat <= 52.5` and `4.4 <= lon <= 5.2` | lon ✅, lat max 51.83 | misses by **0.07°** |

Nudge Dordrecht 0.08° east and the classifier returns `clay`. Nudge it 0.07° north and it returns
`peat`. Verified:

```
f(51.80, 4.81) -> 'clay'      # river-clay rule
f(51.91, 4.70) -> 'peat'      # peat belt rule
```

**Why this matters more than the number itself.** Dordrecht is not a sandy-clay town. It sits in
the Drechtsteden on Holocene peat and river clay and is one of the more subsidence-affected cities
in the country — which is presumably why you picked it as the held-out city in the first place.
The classifier assigns it `sandy_clay` because of where you drew a box, not because of its geology.

So the three confounds your README documents are not three independent findings. They are one
root cause and two consequences:

1. **Root cause:** the coordinate rule's catch-all branch mislabels Dordrecht.
2. Consequence: `sandy_clay` is absent from the training vocabulary → all 974 rows encode as NaN.
3. Consequence: the rule engine scores `sandy_clay` low, so Dordrecht's label distribution inverts
   relative to the peat cities.

The held-out-city experiment is currently measuring *"what happens when my soil classifier
misclassifies the test city"*, not *"does this model generalise across geography"*. Those are
different questions and only one of them is interesting.

### What to do about it

Three things, in order.

**1. Correct the framing (30 minutes, do this first).** The number stands — 0.092 is real and
correctly measured. What changes is the diagnosis. Update `README.md`, `docs/LIMITATIONS.md` L0,
and the Article 15 row in `docs/EU_AI_ACT.md` to say: the collapse traces to a single gap in
`_classify_soil_by_coordinates`, whose catch-all branch assigns Dordrecht `sandy_clay`; the other
two confounds are downstream of it. Include the bounding-box table above. This is a *better*
finding than three vague confounds, because it's specific and you found it by reading your own code.

**2. Turn on the soil API you already wrote.** `pipeline.py:191` has `use_soil_api: bool = False`,
and `soil.py::try_pdok_bro_soil` is a working PDOK BRO Bodemkaart WFS point query that's been
sitting unused. Refetch the four cities with `use_soil_api=True`, then rerun the grouped split.
If Dordrecht comes back peat/clay from the actual soil map, the confound dissolves and you get a
number that means something. Cache the responses to a parquet so the run is reproducible offline
and CI stays green.

That single flag is the difference between "9 of 11 features are simulated" and "8 of 11" — and
it converts your most important feature from a hand-drawn box to a government soil map.

**3. Then the labels.** Even with real soil, `risk_class` still comes from
`synthetic._compute_risk_score`. See the weak-supervision plan below.

---

## What's actually left from audit v1

Sessions 8–11 were never run. Verified still present:

### Still open — worth doing

| § | Issue | Evidence |
|---|---|---|
| 1.6 | RNG seed collisions | `insar.py:46`, `soil.py:89`, `weather.py:46` all still `default_rng(int(abs(lat*N) + abs(lon*N)))`. Additive, so `52.0100/4.7000` and `52.0000/4.7100` both seed 567100 — two buildings 1.1 km apart get byte-identical "measurements". |
| 1.11 | `.iterrows()` loops | 5 sites: `soil.py:195`, `insar.py:123`, `weather.py:93`, `pipeline.py:60`, `pipeline.py:106`. |
| 1.8 | `deploy_hf.py` doesn't purge | Copies into `spaces/src/` without clearing first; package grew 111 → 133 files across deploys. Also doesn't configure LFS for the `.ubj`, so the documented push is rejected. Its printed instructions still say `<YOUR_USERNAME>` and omit LFS entirely. |
| 1.10 | Metrics not regenerated in CI | `models/metrics.json` is committed but nothing verifies it. |
| — | CI installs from `requirements.txt`, not the lock | This is what let the missing `enable_categorical` pass locally and fail in CI. Point CI at `requirements.lock.txt`. |

### Closed since v1 — no action

Categorical vocabulary freeze (§0.1), CI (§1.7), README honesty (§0.2–0.5), EU AI Act gap
analysis (§P2), `.ubj` persistence (§1.1), three-way split + early stopping (§1.2), grouped split
on real data (§1.3), API validation (§1.5), `REFERENCE_YEAR` constant.

### Two small things I'd fix while you're in there

- `docs/LIMITATIONS.md:39` and `:83` say **82.5%**; everywhere else says **82.4%**. Same number,
  two roundings, and this is the one doc where numerical sloppiness costs you most.
- `docs/LIMITATIONS.md` has an escaped underscore (`sandy\_clay`) that renders literally on GitHub.

### One thing I'd leave alone

`allow_unseen=True` on the test frame in the grouped path looks like it weakens the Session 1
guard, but it doesn't — it's opt-in, defaults to False, applies only to batch evaluation, and the
API path never sets it. The docstring explains exactly why. That's the right design.

---

## The weak-supervision plan (§0.2 option B)

This is the work that changes what the project *is*. Everything so far removes reasons to reject
you; this gives someone a reason to hire you.

**The problem in one sentence:** `risk_class` is a weighted sum of the same 11 features the model
receives, so the model can only ever rediscover your arithmetic. You now have unusually sharp
evidence for this — three real peat cities produced **zero** low-risk buildings, because the rule
weights soil at 0.35 and all three are peat. The label is geography wearing a costume.

### Step 1 — find out what's actually available (a day, and it's the part not to delegate)

Candidate sources, in rough order of promise:

- **KCAF FunderMaps** (`fundermaps.com`) — the national foundation-risk platform. Has per-building
  risk indications for parts of the country. Check the licence and whether there's an API or bulk
  export; some layers are municipality-restricted.
- **Municipal open data portals** — Rotterdam, Zaanstad, Gouda and Dordrecht all publish open data.
  Search for *funderingsrisico*, *funderingsproblematiek*, or *funderingskaart* WFS/WMS layers.
  Zaanstad and Dordrecht have both run funding schemes, which usually means a published zone map.
- **Provincie Zuid-Holland / Noord-Holland** geoportals — subsidence and foundation risk zones.
- **BodemDalingsKaart 2.0** — you already cite it for InSAR priors. Check whether the underlying
  raster is downloadable rather than just illustrative.

You're looking for a polygon layer with an ordinal risk attribute. Even three classes over
partial coverage is enough.

### Step 2 — spatial join, honestly

Point-in-polygon your BAG centroids against the zone layer. Label each building with its zone's
risk class. Be explicit in the docs that this is **zone-level weak supervision**, not per-address
ground truth: every building in a polygon gets the same label, so the labels are noisy and
spatially correlated. That's a known and respectable setup — say so and cite it rather than
pretending it's clean.

### Step 3 — the evaluation that makes the story

With real labels, the held-out-city split finally answers a real question. Report:

- random split (for continuity with what you have)
- held-out city, with real soil from step 2 above
- a `most_frequent` baseline on each
- coverage: what fraction of buildings fall inside any zone polygon

If accuracy is bad, report it. A defensible bad number on real weak labels is worth more than a
good number on your own formula, and you've already proved you'll do that.

### What to keep from the current setup

The synthetic generator doesn't go away — it stays as the pipeline integrity check it currently is,
and it's genuinely useful for that. Keep both paths, label them clearly, and the repo tells a
progression: rule engine → real features → real labels. That progression *is* the portfolio piece.

---

## Suggested order

1. Reframe the 0.092 finding around the soil-classifier root cause — half an hour, biggest
   credibility gain per minute available to you right now.
2. `use_soil_api=True`, refetch, rerun grouped split, update the numbers.
3. Fix the seed collisions (§1.6) — small, and they currently produce duplicate "satellite
   measurements".
4. Point CI at `requirements.lock.txt`.
5. Then the weak-supervision work, at its own pace.

Items 6–10 from audit v1 (vectorisation, deploy purge, metrics-in-CI) remain optional. None of
them changes whether this repo survives a technical interview.
