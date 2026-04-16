"""Train the baseline subsidence model.

Generates synthetic data if none is present, then trains and saves the model
with its metrics bundle. Designed to be idempotent and safe to re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from zakkend import config
from zakkend.data.synthetic import generate
from zakkend.models.baseline import train


def main() -> None:
    data_path = config.PROCESSED_DATA_DIR / "training.parquet"

    if not data_path.exists():
        print(f"→ Generating synthetic dataset at {data_path}")
        df = generate(n=10_000)
        df.to_parquet(data_path, index=False)
    else:
        print(f"→ Loading existing dataset from {data_path}")
        df = pd.read_parquet(data_path)

    print(f"→ Training on {len(df):,} rows...")
    model = train(df)

    print(f"→ Saving model to {config.MODEL_PATH}")
    model.save()

    print("\n" + "=" * 60)
    print("  TRAINING METRICS")
    print("=" * 60)
    print(f"Accuracy:       {model.metrics['accuracy']:.4f}")
    print(f"Macro F1:       {model.metrics['macro_f1']:.4f}")
    print(f"Weighted F1:    {model.metrics['weighted_f1']:.4f}")
    print(f"Train rows:     {model.metrics['n_train']:,}")
    print(f"Test rows:      {model.metrics['n_test']:,}")
    print("\nPer-class performance:")
    for cls, m in model.metrics["per_class"].items():
        print(
            f"  {cls:10s}  "
            f"precision={m['precision']:.3f}  "
            f"recall={m['recall']:.3f}  "
            f"f1={m['f1']:.3f}  "
            f"n={m['support']}"
        )

    # Persist metrics alongside the model for the README / model card
    metrics_path = config.MODELS_DIR / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(model.metrics, f, indent=2)
    print(f"\n✓ Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
