"""Train the subsidence model on synthetic or real data.

Usage
-----
    # Phase 1: synthetic data (default)
    python scripts/train.py

    # Phase 2: real data (after running the pipeline)
    python scripts/train.py --data data/processed/real_data.parquet

    # Phase 2: combined (real data + synthetic augmentation)
    python scripts/train.py --data data/processed/real_data.parquet --augment-synthetic 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from zakkend import config
from zakkend.data.synthetic import generate as generate_synthetic
from zakkend.models.baseline import train


def _label_real_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate risk labels for real data using the domain rule engine.

    Since we don't have ground-truth labels for real buildings, we reuse
    the same rule engine from synthetic.py — this ensures consistency
    between training and the domain dynamics the model learns.
    """
    from zakkend.data.synthetic import _compute_risk_score, _score_to_class

    result = df.copy()
    result["risk_score"] = _compute_risk_score(result).round(3)
    result["risk_class"] = _score_to_class(result["risk_score"].values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to real data parquet. If omitted, uses synthetic data.",
    )
    parser.add_argument(
        "--augment-synthetic",
        type=int,
        default=0,
        help="Number of synthetic rows to add alongside real data.",
    )
    parser.add_argument(
        "--synthetic-n",
        type=int,
        default=10_000,
        help="Number of synthetic rows (when not using real data).",
    )
    args = parser.parse_args()

    # ─── Load or generate data ───
    frames = []

    if args.data and args.data.exists():
        print(f"→ Loading real data from {args.data}")
        real_df = pd.read_parquet(args.data)
        print(f"  {len(real_df):,} buildings loaded")

        # Label if not already labeled
        if config.TARGET_COLUMN not in real_df.columns:
            print("→ Generating risk labels with domain rule engine...")
            real_df = _label_real_data(real_df)
            print(f"  Class distribution:\n{real_df['risk_class'].value_counts().to_string()}")

        frames.append(real_df)

        if args.augment_synthetic > 0:
            print(f"→ Augmenting with {args.augment_synthetic:,} synthetic rows")
            syn = generate_synthetic(n=args.augment_synthetic)
            frames.append(syn)
    else:
        if args.data:
            print(f"⚠ File not found: {args.data}, falling back to synthetic data")
        synth_path = config.PROCESSED_DATA_DIR / "training.parquet"
        if synth_path.exists():
            print(f"→ Loading existing synthetic data from {synth_path}")
            frames.append(pd.read_parquet(synth_path))
        else:
            print(f"→ Generating {args.synthetic_n:,} synthetic rows")
            syn = generate_synthetic(n=args.synthetic_n)
            syn.to_parquet(synth_path, index=False)
            frames.append(syn)

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # ─── Ensure all required columns exist ───
    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        print(f"✗ Missing required columns: {sorted(missing)}")
        return

    print(f"\n→ Training on {len(df):,} total rows...")
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

    # ─── Data source breakdown ───
    if "municipality" in df.columns:
        print("\nData sources:")
        for muni, count in df["municipality"].value_counts().items():
            print(f"  {muni}: {count:,} buildings")
        synth_count = df["municipality"].isna().sum()
        if synth_count > 0:
            print(f"  Synthetic: {synth_count:,} rows")

    metrics_path = config.MODELS_DIR / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(model.metrics, f, indent=2)
    print(f"\n✓ Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
