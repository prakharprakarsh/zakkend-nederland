"""Tests for the baseline model + SHAP explainer."""

from __future__ import annotations

import pandas as pd
import pytest

from zakkend import config
from zakkend.data.synthetic import generate
from zakkend.explain.shap_explainer import SubsidenceExplainer
from zakkend.models.baseline import train


@pytest.fixture(scope="module")
def trained_model():
    df = generate(n=2_000)
    return train(df)


def test_model_trains(trained_model):
    assert trained_model.classifier is not None
    assert trained_model.metrics["accuracy"] > 0.60  # beats 25% random on 4 classes


def test_model_predicts_correct_shape(trained_model):
    df = generate(n=50)
    probs = trained_model.predict_proba(df)
    assert probs.shape == (50, len(config.RISK_CLASSES))
    # probabilities sum to 1
    assert (abs(probs.sum(axis=1) - 1.0) < 1e-5).all()


def test_model_predict_class_returns_valid_labels(trained_model):
    df = generate(n=20)
    preds = trained_model.predict_class(df)
    assert all(p in config.RISK_CLASSES for p in preds)


def test_save_and_load_roundtrip(tmp_path, trained_model):
    path = tmp_path / "m.ubj"
    trained_model.save(path)

    assert path.exists(), "save() must write the .ubj model file"
    assert path.with_suffix(".meta.json").exists(), "save() must write the sibling .meta.json"

    from zakkend.models.baseline import TrainedModel

    loaded = TrainedModel.load(path)
    assert loaded.category_levels == trained_model.category_levels
    df = generate(n=10)
    original = trained_model.predict_proba(df)
    reloaded = loaded.predict_proba(df)
    assert (abs(original - reloaded) < 1e-6).all()


def test_shap_explainer_returns_contributions(trained_model):
    df = generate(n=1)
    explainer = SubsidenceExplainer(trained_model)
    expl = explainer.explain(df)
    assert expl.predicted_class in config.RISK_CLASSES
    assert len(expl.contributions) == len(config.FEATURE_COLUMNS)
    assert len(expl.top_drivers(k=3)) == 3


def test_baselines_computed_and_in_metrics(trained_model):
    """DummyClassifier baselines must be present and cover all four strategies."""
    baselines = trained_model.metrics.get("baselines", {})
    for strategy in ("most_frequent", "prior", "stratified", "uniform"):
        assert strategy in baselines, f"baseline '{strategy}' missing from metrics"
        assert 0.0 < baselines[strategy] < 1.0


def test_model_beats_majority_class_baseline(trained_model):
    """XGBoost must beat most_frequent by >20 percentage points on the test split."""
    xgb_acc = trained_model.metrics["accuracy"]
    mf_acc = trained_model.metrics["baselines"]["most_frequent"]
    margin = xgb_acc - mf_acc
    assert margin > 0.20, (
        f"XGBoost ({xgb_acc:.3f}) beats most_frequent ({mf_acc:.3f}) by only "
        f"{margin:.3f} — expected >0.20"
    )


def test_three_way_split_sizes(trained_model):
    """60/20/20 split: n_train + n_val + n_test must equal the input length."""
    m = trained_model.metrics
    assert "n_val" in m, "three-way split must report n_val"
    total = m["n_train"] + m["n_val"] + m["n_test"]
    # fixture trains on n=2000
    assert total == 2_000
    # approximately 60/20/20 within rounding of stratified splits
    assert 0.55 <= m["n_train"] / total <= 0.65
    assert 0.15 <= m["n_val"] / total <= 0.25
    assert 0.15 <= m["n_test"] / total <= 0.25


def test_grouped_split_municipality_disjoint():
    """municipality_split metadata must be stored and train/test sets must not overlap."""
    from sklearn.model_selection import train_test_split as sk_split

    train_df = generate(n=1_200)
    test_df_raw = generate(n=300)

    train_core, val_df = sk_split(train_df, test_size=0.20, random_state=config.RANDOM_STATE)
    model = train(
        train_core,
        val_df=val_df,
        test_df=test_df_raw,
        municipality_split={"train": ["CityA"], "test": ["CityB"]},
    )

    ms = model.metrics.get("municipality_split", {})
    assert ms, "municipality_split must be in metrics when provided"
    train_set = set(ms["train"])
    test_set = set(ms["test"])
    assert train_set.isdisjoint(
        test_set
    ), f"Municipality overlap between train {train_set} and test {test_set}"


def test_shap_directions_labeled(trained_model):
    df = generate(n=1)
    explainer = SubsidenceExplainer(trained_model)
    expl = explainer.explain(df)
    for c in expl.contributions:
        assert c.direction in ("↑ risk", "↓ risk")


class TestSpacesAppPredictionPath:
    """Regression guard for the building_age omission in spaces/app.py.

    spaces/app.py bypasses api/main.py and calls the model directly. When
    api/main.py was refactored to derive building_age server-side, spaces/app.py
    was not updated, causing "Missing required feature columns: ['building_age']"
    on every prediction. These tests exercise the exact DataFrame-building pattern
    that spaces/app.py uses so a similar refactor cannot silently break the Space.
    """

    # Mirrors the `features` dict built in spaces/app.py — no building_age.
    SPACES_FEATURES: dict = {
        "year_built": 1925,
        "foundation_type": "wooden_pile",
        "soil_type": "peat",
        "peat_thickness_m": 2.5,
        "groundwater_depth_m": 1.1,
        "groundwater_variability": 0.45,
        "insar_deformation_mm_yr": 3.2,
        "drought_exposure_index": 0.55,
        "distance_to_water_m": 120,
        "neighborhood_damage_rate": 0.18,
    }

    def test_derive_building_age_then_predict(self, trained_model):
        """Derive building_age from year_built (as spaces/app.py does) and predict."""
        df = pd.DataFrame([self.SPACES_FEATURES])
        df["building_age"] = config.REFERENCE_YEAR - df["year_built"]

        probs = trained_model.predict_proba(df)[0]
        assert len(probs) == len(config.RISK_CLASSES)
        assert abs(sum(probs) - 1.0) < 1e-5
        risk_class = trained_model.class_names[int(probs.argmax())]
        assert risk_class in config.RISK_CLASSES

    def test_missing_building_age_raises(self, trained_model):
        """Omitting building_age must raise ValueError — documents the original bug."""
        df = pd.DataFrame([self.SPACES_FEATURES])  # no building_age column
        with pytest.raises(ValueError, match="building_age"):
            trained_model.predict_proba(df)

    def test_agent_assess_risk_with_complete_features(self, trained_model, monkeypatch):
        """assess_risk node succeeds when features dict includes building_age.

        Regression guard for spaces/app.py passing the raw sidebar dict (without
        building_age) to agent.invoke(). After the fix, building_age is derived on
        the features dict itself so both the model DataFrame and the agent receive
        a complete feature set.
        """
        from zakkend.agent.graph import assess_risk
        from zakkend.models.baseline import TrainedModel

        # Inject the fixture model so assess_risk doesn't need to load from disk
        monkeypatch.setattr(TrainedModel, "load", lambda *a, **kw: trained_model)

        features = {
            **self.SPACES_FEATURES,
            "building_age": config.REFERENCE_YEAR - self.SPACES_FEATURES["year_built"],
        }
        state = assess_risk({"building_features": features})
        assert state["risk_class"] in config.RISK_CLASSES
        assert 0.0 <= state["risk_score"] <= 1.0
        assert len(state["shap_drivers"]) == 5
