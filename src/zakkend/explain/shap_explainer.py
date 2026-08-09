"""Per-prediction SHAP explanations for the subsidence model.

SHAP (SHapley Additive exPlanations) is our chosen method for EU AI Act
Article 13 compliance — every score the model produces can be decomposed
into the contribution of each input feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from zakkend.features.engineering import build_feature_matrix
from zakkend.models.baseline import TrainedModel


@dataclass
class FeatureContribution:
    feature: str
    value: Any
    shap_value: float
    direction: str  # "↑ risk" / "↓ risk"


@dataclass
class Explanation:
    predicted_class: str
    predicted_class_index: int
    base_value: float
    contributions: list[FeatureContribution]

    def top_drivers(self, k: int = 5) -> list[FeatureContribution]:
        return sorted(self.contributions, key=lambda c: abs(c.shap_value), reverse=True)[:k]


class SubsidenceExplainer:
    """Thin wrapper around shap.TreeExplainer."""

    def __init__(self, model: TrainedModel) -> None:
        self.model = model
        self._explainer = shap.TreeExplainer(model.classifier)

    def explain(self, row: pd.DataFrame | dict) -> Explanation:
        if isinstance(row, dict):
            row = pd.DataFrame([row])
        if len(row) != 1:
            raise ValueError("Explain one row at a time.")

        x_feat = build_feature_matrix(row, category_levels=self.model.category_levels)
        probs = self.model.predict_proba(x_feat)[0]
        pred_idx = int(probs.argmax())

        # For multi-class XGBoost, shap_values is shape (n, n_features, n_classes)
        shap_values = self._explainer.shap_values(x_feat)
        shap_values = np.asarray(shap_values)

        sv = shap_values[0, :, pred_idx] if shap_values.ndim == 3 else shap_values[0]

        base = self._explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).flatten()[pred_idx])
        else:
            base = float(base)

        contribs = [
            FeatureContribution(
                feature=feat,
                value=row[feat].iloc[0],
                shap_value=float(sv[i]),
                direction="↑ risk" if sv[i] > 0 else "↓ risk",
            )
            for i, feat in enumerate(self.model.feature_columns)
        ]

        return Explanation(
            predicted_class=self.model.class_names[pred_idx],
            predicted_class_index=pred_idx,
            base_value=base,
            contributions=contribs,
        )
