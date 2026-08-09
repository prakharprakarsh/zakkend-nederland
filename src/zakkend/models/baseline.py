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

from zakkend import config
from zakkend.features.engineering import build_feature_matrix, encode_risk_class

logger = logging.getLogger(__name__)

_N_CLASSES = len(config.RISK_CLASSES)


@dataclass
class TrainedModel:
    """Bundle of a fitted classifier + metadata for reproducibility."""

    classifier: xgb.Booster
    feature_columns: list[str]
    categorical_features: list[str]
    class_names: list[str]
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x_ready = build_feature_matrix(df, category_levels=self.category_levels)
        dm = xgb.DMatrix(x_ready, enable_categorical=True)
        probs = self.classifier.predict(dm)
        if probs.ndim == 1:
            probs = probs.reshape(-1, len(self.class_names))
        return probs

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

        bst = xgb.Booster()
        bst.load_model(ubj_path)
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
            classifier=bst,
            feature_columns=meta["feature_columns"],
            categorical_features=meta["categorical_features"],
            category_levels=meta.get("category_levels", {}),
            class_names=meta["class_names"],
            metrics=meta.get("metrics", {}),
        )


def _booster_params() -> dict[str, Any]:
    """XGBoost training params for the Booster API. Tuned for ~10k rows."""
    return {
        "objective": "multi:softprob",
        "num_class": _N_CLASSES,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": config.RANDOM_STATE,
        "nthread": -1,
        "verbosity": 0,
    }


def _sample_weights(y: pd.Series, use_weights: bool) -> np.ndarray | None:
    """Balanced sample weights using the full 4-class vocabulary.

    Uses _N_CLASSES (always 4) in the denominator so absent training classes
    do not inflate weights for the classes that are present.  A class with zero
    training examples gets zero weight naturally — no samples, no contribution.
    """
    if not use_weights:
        return None
    n = len(y)
    weights = np.zeros(n, dtype=np.float32)
    y_arr = np.asarray(y)
    for cls_idx in range(_N_CLASSES):
        mask = y_arr == cls_idx
        count = int(mask.sum())
        if count > 0:
            weights[mask] = n / (_N_CLASSES * count)
    return weights


def train(
    df: pd.DataFrame,
    *,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    use_sample_weights: bool = True,
    municipality_split: dict[str, list[str]] | None = None,
) -> TrainedModel:
    """Train the baseline model on a labeled dataframe.

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

    evaluation_notes: dict[str, Any] = {}

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

        # Detect categories in test_df not seen in training (e.g. held-out city has
        # different soil type). Record them as evaluation notes; encode as NaN so
        # XGBoost handles them via its missing-value branch.
        for col in config.CATEGORICAL_FEATURES:
            test_vals = set(test_df[col].astype(str).unique())
            unseen = sorted(test_vals - set(category_levels[col]))
            if unseen:
                evaluation_notes[col] = {
                    "unseen_in_test": unseen,
                    "training_vocabulary": category_levels[col],
                    "note": (
                        f"Test rows with {col} in {unseen} are not in the training vocabulary "
                        f"{category_levels[col]}. They are encoded as NaN and handled by "
                        "XGBoost's missing-value split. The reported accuracy conflates "
                        "geographic distribution shift with unseen-category coverage; "
                        "these effects cannot be separated from this evaluation alone."
                    ),
                }

        x_test = build_feature_matrix(test_df, category_levels=category_levels, allow_unseen=True)
        y_test = encode_risk_class(test_df[config.TARGET_COLUMN])

    # Document which classes are absent from training (informational — no anchor rows added).
    present_classes = {int(v) for v in np.asarray(y_train)}
    absent_classes = [config.RISK_CLASSES[i] for i in range(_N_CLASSES) if i not in present_classes]
    if absent_classes:
        evaluation_notes["absent_training_classes"] = {
            "classes": absent_classes,
            "note": (
                f"Classes {absent_classes} have zero training examples under the rule engine "
                "for the given municipalities. XGBoost (num_class=4) still outputs 4-class "
                "probabilities; absent classes receive base-score logits only. "
                "No synthetic anchor rows were injected."
            ),
        }

    weights = _sample_weights(y_train, use_sample_weights)

    dm_train = xgb.DMatrix(x_train, label=y_train, weight=weights, enable_categorical=True)
    dm_val = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
    dm_test = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)

    bst = xgb.train(
        _booster_params(),
        dm_train,
        num_boost_round=300,
        evals=[(dm_val, "val")],
        callbacks=[xgb.callback.EarlyStopping(rounds=30, save_best=True)],
    )

    raw = bst.predict(dm_test)
    if raw.ndim == 1:
        raw = raw.reshape(-1, _N_CLASSES)
    y_pred = raw.argmax(axis=1)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(_N_CLASSES)),
        target_names=list(config.RISK_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(_N_CLASSES))).tolist()

    baselines: dict[str, float] = {}
    for strategy in ("most_frequent", "prior", "stratified", "uniform"):
        dummy = DummyClassifier(strategy=strategy, random_state=config.RANDOM_STATE)
        dummy.fit(x_train, y_train)
        baselines[strategy] = round(float(dummy.score(x_test, y_test)), 4)
    # Fixed 4-class uniform: 1/4 = 0.25, regardless of training class distribution.
    baselines["uniform_4class"] = round(1.0 / _N_CLASSES, 4)

    best_iter = getattr(bst, "best_iteration", None)
    if best_iter is not None:
        best_iter = int(best_iter)

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

    if evaluation_notes:
        metrics["evaluation_notes"] = evaluation_notes

    return TrainedModel(
        classifier=bst,
        feature_columns=config.FEATURE_COLUMNS,
        categorical_features=config.CATEGORICAL_FEATURES,
        class_names=list(config.RISK_CLASSES),
        category_levels=category_levels,
        metrics=metrics,
    )
