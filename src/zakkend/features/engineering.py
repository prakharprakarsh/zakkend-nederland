"""Feature engineering for the subsidence risk model."""

from __future__ import annotations

import pandas as pd

from zakkend import config


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select + one-hot encode features for modelling.

    The XGBoost pipeline handles the categorical encoding internally when
    `enable_categorical=True` is set, but we normalize dtypes here for safety
    and to keep the API input validation aligned with training-time schema.
    """
    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    features = df[config.FEATURE_COLUMNS].copy()
    for col in config.CATEGORICAL_FEATURES:
        features[col] = features[col].astype("category")
    return features


def encode_risk_class(y: pd.Series) -> pd.Series:
    """Map ordinal class names to integers (preserving ordering)."""
    mapping = {cls: i for i, cls in enumerate(config.RISK_CLASSES)}
    return y.map(mapping).astype(int)


def decode_risk_class(y: pd.Series | int) -> str | pd.Series:
    """Inverse of `encode_risk_class`."""
    mapping = dict(enumerate(config.RISK_CLASSES))
    if isinstance(y, int):
        return mapping[y]
    return y.map(mapping)
