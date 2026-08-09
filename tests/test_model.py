"""Tests for the baseline model + SHAP explainer."""

from __future__ import annotations

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


def test_shap_directions_labeled(trained_model):
    df = generate(n=1)
    explainer = SubsidenceExplainer(trained_model)
    expl = explainer.explain(df)
    for c in expl.contributions:
        assert c.direction in ("↑ risk", "↓ risk")
