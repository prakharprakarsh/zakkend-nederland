# CLAUDE.md — zakkend-nederland

## What this project is

ML-powered foundation subsidence risk assessment for Dutch homes. XGBoost multiclass
(`low`/`moderate`/`high`/`critical`) + SHAP explanations, served via FastAPI (local demo) and
Streamlit (Hugging Face Spaces).

This is a **portfolio project for AI/ML engineering roles in the Netherlands** (target: ING, Adyen,
Booking.com, ASML, Philips). Reviewers will be senior ML engineers who read the code. Optimise for
things a sharp reviewer would check, not for surface polish.

## Current state

The repo is being hardened following an audit — see `docs/AUDIT.md`. Read it before making changes.
It contains reproduced evidence for every issue, plus the intended fix for each. Section numbers
in `docs/AUDIT.md` (§0.1, §1.7, ...) are referenced in commit messages.

## Non-negotiable rules

1. **Never claim in docs what the code doesn't do.** The single biggest problem in this repo's
   history was README/doc claims that the code doesn't support ("real PDOK data", "calibrated
   probabilities", "postcode lookup", "Pydantic blocks OOD inputs"). If you change a claim, verify
   it against the code first. If you can't verify it, delete the claim.
2. **Labels are synthetic and derived from `synthetic._compute_risk_score`.** Reported accuracy
   measures rule-recovery, not real-world predictive skill. Never describe metrics as predictive
   performance. Any new doc text about metrics must carry this caveat.
3. **8 of 11 features are fully simulated** (`np.random.default_rng`), not measured. Only
   `year_built` and the lat/lon centroid come from PDOK BAG. `soil_type` is hybrid: BRO
   Bodemkaart GeoPackage spatial join (29.5% of buildings, cached in
   `data/raw/bro_soil_cache.parquet`) + coordinate-based rule fallback (70.5%). Functions
   that simulate should be named `simulate_*`, not `estimate_*` or `enrich_*`.
4. **Categorical encoding.** `foundation_type` and `soil_type` MUST be encoded against a frozen
   vocabulary captured at training time (`TrainedModel.category_levels`). Never call
   `.astype("category")` on an inference-time frame without passing explicit categories — a 1-row
   frame produces a 1-level dtype and XGBoost 2.x maps everything to code 0. There is a regression
   test for this; do not weaken it.
5. **Don't touch the test set during training.** No `eval_set=[(X_test, y_test)]`. Use a separate
   validation split.
6. **Determinism.** `config.RANDOM_STATE = 42` everywhere. Training on the synthetic pipeline must
   be reproducible, because CI regenerates `models/metrics.json` from it.

## Conventions

- Python 3.11+ / 3.12. `src/` layout, package `zakkend`, installed with `pip install -e ".[dev]"`.
- `ruff` + `black`, line length 100. Run `make lint` before committing.
- Type hints everywhere; `from __future__ import annotations` at the top of every module.
- NumPy-style docstrings on public functions.
- Prefer vectorised pandas/numpy. `.iterrows()` is banned in new code.
- Tests use pytest with class-based grouping (`class TestBAGParser:`). Mock all network calls —
  tests must pass offline, in CI, with no PDOK/KNMI access.

## Commit discipline

- **One logical change per commit.** This repo's history is three giant commits; every new commit
  should be small and reviewable to counteract that.
- Conventional commits: `fix(inference):`, `feat(ci):`, `docs(readme):`, `refactor(pipeline):`,
  `test(api):`, `chore(deps):`.
- Reference the audit section in the body: `Ref: docs/AUDIT.md §0.1`.
- Never `git push --force` on `main`.

## Layout

```
src/zakkend/
  config.py              constants, paths, feature schema, PDOK endpoints
  data/                  synthetic.py (generator + label rule), bag.py (real PDOK),
                         soil.py / insar.py / weather.py (simulated enrichment), pipeline.py (ETL)
  features/engineering.py build_feature_matrix, encode/decode_risk_class
  models/baseline.py      TrainedModel dataclass, build_classifier, train
  explain/shap_explainer.py
  api/main.py + static/index.html   FastAPI demo
  agent/                  LangGraph remediation-report agent
scripts/                 train.py, fetch_and_train.py, deploy_hf.py
spaces/                  Streamlit app for HF Spaces
tests/                   55 tests, all offline
```

## Verification before any commit

```bash
make lint
pytest -q
```

Both must be clean. If you changed anything under `models/` or `features/`, also retrain and
confirm the metrics didn't silently move:

```bash
python scripts/train.py
```
