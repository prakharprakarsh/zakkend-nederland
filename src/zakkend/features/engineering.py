"""Feature engineering for the subsidence risk model."""

from __future__ import annotations

import pandas as pd

from zakkend import config


class UnknownCategoryError(ValueError):
    """Raised when an inference-time category value was not seen during training."""


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    category_levels: dict[str, list[str]],
    allow_unseen: bool = False,
) -> pd.DataFrame:
    """Select and encode features for modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input frame containing at least `config.FEATURE_COLUMNS`.
    category_levels : dict[str, list[str]]
        Vocabulary captured at training time, keyed by column name.
        Must be passed explicitly — callers that omit it get a TypeError,
        not a silent fallback that reproduces the 1-row encoding bug.
    allow_unseen : bool
        When False (default), raise ``UnknownCategoryError`` on any value
        not in the training vocabulary. Set to True only for batch evaluation
        of held-out splits where unseen categories are expected and intentional
        (e.g., a held-out city with a different soil type). Unknown values are
        coerced to NaN and handled by XGBoost's missing-value branch.

    Raises
    ------
    ValueError
        If required feature columns are missing or `category_levels` lacks
        an entry for a categorical column.
    UnknownCategoryError
        If an inference-time value is not in the training vocabulary and
        ``allow_unseen`` is False.
    """
    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    features = df[config.FEATURE_COLUMNS].copy()
    for col in config.CATEGORICAL_FEATURES:
        values = features[col].astype(str)
        if col not in category_levels:
            raise ValueError(
                f"category_levels is missing an entry for '{col}'. "
                "Pass the vocabulary captured at training time."
            )
        allowed = category_levels[col]
        unseen = sorted(set(values) - set(allowed))
        if unseen and not allow_unseen:
            raise UnknownCategoryError(f"{col}: {unseen} not in training vocabulary {allowed}")
        features[col] = pd.Categorical(values, categories=allowed)
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
