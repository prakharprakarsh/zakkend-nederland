"""Baseline subsidence risk classifier: XGBoost with native categorical support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from zakkend import config
from zakkend.features.engineering import build_feature_matrix, encode_risk_class


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

    def save(self, path=None) -> None:
        path = path or config.MODEL_PATH
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=None) -> TrainedModel:
        path = path or config.MODEL_PATH
        return joblib.load(path)


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
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> TrainedModel:
    """Train the baseline model on a labeled dataframe.

    Expects `df` to contain every column in `config.FEATURE_COLUMNS`
    plus the target `config.TARGET_COLUMN`.
    """
    category_levels = {
        col: sorted(df[col].astype(str).unique().tolist()) for col in config.CATEGORICAL_FEATURES
    }

    x = build_feature_matrix(df, category_levels=category_levels)
    y = encode_risk_class(df[config.TARGET_COLUMN])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    clf = build_classifier()
    clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

    y_pred = clf.predict(x_test)
    report = classification_report(
        y_test, y_pred, target_names=list(config.RISK_CLASSES), output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
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
        "n_test": len(x_test),
    }

    return TrainedModel(
        classifier=clf,
        feature_columns=config.FEATURE_COLUMNS,
        categorical_features=config.CATEGORICAL_FEATURES,
        class_names=list(config.RISK_CLASSES),
        category_levels=category_levels,
        metrics=metrics,
    )
