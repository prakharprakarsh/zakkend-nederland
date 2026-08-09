"""Baseline subsidence risk classifier: XGBoost with native categorical support."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from zakkend import config
from zakkend.features.engineering import build_feature_matrix, encode_risk_class

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """Bundle of a fitted classifier + metadata for reproducibility."""

    classifier: xgb.XGBClassifier
    feature_columns: list[str]
    categorical_features: list[str]
    class_names: list[str]
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x_ready = build_feature_matrix(df, category_levels=self.category_levels)
        return self.classifier.predict_proba(x_ready)

    def predict_class(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(df)
        idx = probs.argmax(axis=1)
        return np.array([self.class_names[i] for i in idx])

    def save(self, path: Path | str | None = None) -> None:
        """Save model in XGBoost's portable UBJSON format.

        Writes two sibling files:
          <stem>.ubj   — the XGBoost model (portable across XGBoost versions)
          <stem>.meta.json — feature schema, category levels, metrics, version stamp
        """
        ubj_path = Path(path or config.MODEL_PATH).with_suffix(".ubj")
        ubj_path.parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(ubj_path)
        meta = {
            "feature_columns": self.feature_columns,
            "categorical_features": self.categorical_features,
            "category_levels": self.category_levels,
            "class_names": self.class_names,
            "metrics": self.metrics,
            "xgboost_version": xgb.__version__,
            "trained_at": datetime.now(UTC).isoformat(),
        }
        ubj_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path | str | None = None) -> TrainedModel:
        """Load a saved model.

        Prefers the portable .ubj + .meta.json pair. Falls back to a legacy
        .joblib file with a deprecation warning if the .ubj artefact is absent.
        """
        given = Path(path or config.MODEL_PATH)
        ubj_path = given.with_suffix(".ubj")
        meta_path = given.with_suffix(".meta.json")

        if not ubj_path.exists():
            joblib_path = given.with_suffix(".joblib")
            if joblib_path.exists():
                logger.warning(
                    "Loading legacy joblib artefact %s — re-run `python scripts/train.py` "
                    "to migrate to the portable .ubj format.",
                    joblib_path,
                )
                import joblib  # noqa: PLC0415

                return joblib.load(joblib_path)
            raise FileNotFoundError(
                f"No model found at {ubj_path} (or legacy {joblib_path}). "
                "Run `python scripts/train.py` first."
            )

        clf = xgb.XGBClassifier()
        clf.load_model(ubj_path)
        meta = json.loads(meta_path.read_text())

        saved_version = meta.get("xgboost_version", "unknown")
        if saved_version != xgb.__version__:
            logger.warning(
                "Model was trained with XGBoost %s; current version is %s. "
                "Re-train if predictions look unexpected.",
                saved_version,
                xgb.__version__,
            )

        return cls(
            classifier=clf,
            feature_columns=meta["feature_columns"],
            categorical_features=meta["categorical_features"],
            category_levels=meta.get("category_levels", {}),
            class_names=meta["class_names"],
            metrics=meta.get("metrics", {}),
        )


def build_classifier() -> xgb.XGBClassifier:
    """XGBoost with native categorical handling. Tuned for ~10k rows."""
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(config.RISK_CLASSES),
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=30,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def train(
    df: pd.DataFrame,
    *,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    use_sample_weights: bool = True,
    municipality_split: dict[str, list[str]] | None = None,
) -> TrainedModel:
    """Train the baseline model on a labeled dataframe.

    Expects `df` to contain every column in `config.FEATURE_COLUMNS`
    plus the target `config.TARGET_COLUMN`.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (or full dataset when val_df/test_df are None).
    val_df : pd.DataFrame or None
        Pre-built validation split. Must be provided together with `test_df`.
        When None, a random 60/20/20 stratified split is used.
    test_df : pd.DataFrame or None
        Pre-built test split. Must be provided together with `val_df`.
    use_sample_weights : bool
        Whether to apply balanced class weights during XGBoost training.
    municipality_split : dict or None
        Metadata dict (e.g. ``{"train": [...], "test": [...]}``). When provided
        it is stored verbatim in metrics under ``"municipality_split"``.

    Returns
    -------
    TrainedModel
        Fitted model with evaluation metrics.

    Raises
    ------
    ValueError
        If exactly one of `val_df` / `test_df` is None.
    """
    if (val_df is None) != (test_df is None):
        raise ValueError("val_df and test_df must both be provided or both be None")

    # Compute category levels from the training frame only.
    category_levels = {
        col: sorted(df[col].astype(str).unique().tolist()) for col in config.CATEGORICAL_FEATURES
    }

    x = build_feature_matrix(df, category_levels=category_levels)
    y = encode_risk_class(df[config.TARGET_COLUMN])

    if val_df is None:
        # Random 60/20/20 stratified split.
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x,
            y,
            test_size=0.20,
            stratify=y,
            random_state=config.RANDOM_STATE,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval,
            y_trainval,
            test_size=0.25,
            stratify=y_trainval,
            random_state=config.RANDOM_STATE,
        )
    else:
        # Pre-supplied splits; df is the training frame.
        x_train, y_train = x, y
        x_val = build_feature_matrix(val_df, category_levels=category_levels)
        y_val = encode_risk_class(val_df[config.TARGET_COLUMN])
        x_test = build_feature_matrix(test_df, category_levels=category_levels)
        y_test = encode_risk_class(test_df[config.TARGET_COLUMN])

    weights = compute_sample_weight("balanced", y_train) if use_sample_weights else None

    clf = build_classifier()
    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        sample_weight=weights,
        verbose=False,
    )

    y_pred = clf.predict(x_test)
    report = classification_report(
        y_test, y_pred, target_names=list(config.RISK_CLASSES), output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    baselines: dict[str, float] = {}
    for strategy in ("most_frequent", "prior", "stratified", "uniform"):
        dummy = DummyClassifier(strategy=strategy, random_state=config.RANDOM_STATE)
        dummy.fit(x_train, y_train)
        baselines[strategy] = round(float(dummy.score(x_test, y_test)), 4)

    best_iter = None
    if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
        best_iter = int(clf.best_iteration)

    metrics: dict[str, object] = {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in config.RISK_CLASSES
        },
        "confusion_matrix": cm,
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_test": len(x_test),
        "best_iteration": best_iter,
        "sample_weighted": bool(use_sample_weights),
        "baselines": baselines,
    }

    if municipality_split is not None:
        metrics["municipality_split"] = municipality_split

    return TrainedModel(
        classifier=clf,
        feature_columns=config.FEATURE_COLUMNS,
        categorical_features=config.CATEGORICAL_FEATURES,
        class_names=list(config.RISK_CLASSES),
        category_levels=category_levels,
        metrics=metrics,
    )
