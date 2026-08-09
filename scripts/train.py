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
from sklearn.model_selection import train_test_split

from zakkend import config
from zakkend.data.synthetic import generate as generate_synthetic
from zakkend.data.synthetic import generate_for_municipality
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


def _print_metrics(model: object, label: str = "") -> None:
    """Print a formatted metrics block for the given trained model."""
    from zakkend.models.baseline import TrainedModel  # local to avoid circular import at module top

    assert isinstance(model, TrainedModel)
    m = model.metrics
    header = f"TRAINING METRICS — {label}" if label else "TRAINING METRICS"
    print("\n" + "=" * 60)
    print(f"  {header}")
    print("=" * 60)
    print(f"Accuracy:       {m['accuracy']:.4f}")
    print(f"Macro F1:       {m['macro_f1']:.4f}")
    print(f"Weighted F1:    {m['weighted_f1']:.4f}")
    print(f"Train rows:     {m['n_train']:,}")
    print(f"Val rows:       {m['n_val']:,}")
    print(f"Test rows:      {m['n_test']:,}")
    if m.get("best_iteration") is not None:
        print(f"Best iteration: {m['best_iteration']}")
    print(f"Sample weighted:{m.get('sample_weighted', False)}")
    if "municipality_split" in m:
        ms = m["municipality_split"]
        print(f"Municipality split — train: {ms.get('train')}  test: {ms.get('test')}")
    print("\nPer-class performance:")
    for cls, pc in m["per_class"].items():
        print(
            f"  {cls:10s}  "
            f"precision={pc['precision']:.3f}  "
            f"recall={pc['recall']:.3f}  "
            f"f1={pc['f1']:.3f}  "
            f"n={pc['support']}"
        )
    print("\nBaselines (same test split):")
    for strategy, acc in m["baselines"].items():
        marker = " <- majority class" if strategy == "most_frequent" else ""
        print(f"  DummyClassifier({strategy:14s})  accuracy={acc:.4f}{marker}")
    margin = m["accuracy"] - m["baselines"]["most_frequent"]
    print(f"  XGBoost margin over most_frequent: +{margin:.4f}")


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
    parser.add_argument(
        "--split",
        choices=("random", "grouped"),
        default="random",
        help=(
            "Split strategy. 'random': 60/20/20 stratified split of the full dataset. "
            "'grouped': train on Gouda+Rotterdam+Zaanstad synthetic data, "
            "test on Dordrecht (municipality-based spatial holdout)."
        ),
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

    # ─── Train ───
    if args.split == "random":
        print(f"\n→ Training on {len(df):,} total rows (random 60/20/20 split)...")

        # Compare weighted vs unweighted to surface the effect.
        print("  Running unweighted baseline for comparison...")
        model_no_weights = train(df, use_sample_weights=False)
        print("  Running weighted training (this is the saved model)...")
        model = train(df, use_sample_weights=True)

        print("\n" + "=" * 60)
        print("  SAMPLE WEIGHT COMPARISON  (random split)")
        print("=" * 60)
        print(f"{'Class':12s}  {'F1 (no weights)':>15s}  {'F1 (weighted)':>13s}  {'Delta':>7s}")
        print("-" * 55)
        for cls in config.RISK_CLASSES:
            f1_nw = model_no_weights.metrics["per_class"][cls]["f1"]
            f1_w = model.metrics["per_class"][cls]["f1"]
            delta = f1_w - f1_nw
            sign = "+" if delta >= 0 else ""
            print(f"  {cls:10s}  {f1_nw:>15.3f}  {f1_w:>13.3f}  {sign}{delta:>6.3f}")

    else:
        # ─── Grouped / municipality-based spatial holdout ───
        print("\n→ Grouped split: train=Gouda+Rotterdam+Zaanstad, test=Dordrecht")
        train_munis = ["Gouda", "Rotterdam", "Zaanstad"]
        test_muni = "Dordrecht"

        train_frames = [generate_for_municipality(m, n=2_500) for m in train_munis]
        train_concat = pd.concat(train_frames, ignore_index=True)
        dordrecht_df = generate_for_municipality(test_muni, n=2_500)

        print(f"  Training pool: {len(train_concat):,} rows")
        print(f"  Test set:      {len(dordrecht_df):,} rows")

        # Build train/val split from the concatenated training data.
        train_core_df, val_df_df = train_test_split(
            train_concat,
            test_size=0.20,
            random_state=config.RANDOM_STATE,
        )

        model = train(
            train_core_df,
            val_df=val_df_df,
            test_df=dordrecht_df,
            municipality_split={"train": train_munis, "test": [test_muni]},
        )

    # ─── Save model ───
    print(f"\n→ Saving model to {config.MODEL_PATH}")
    model.save()

    # ─── Print metrics ───
    _print_metrics(model, label=f"{args.split} split")

    # ─── Data source breakdown ───
    if "municipality" in df.columns:
        print("\nData sources:")
        for muni, count in df["municipality"].value_counts().items():
            print(f"  {muni}: {count:,} buildings")
        synth_count = df["municipality"].isna().sum()
        if synth_count > 0:
            print(f"  Synthetic: {synth_count:,} rows")

    # ─── Write metrics.json (incremental merge) ───
    metrics_path = config.MODELS_DIR / "metrics.json"
    all_metrics: dict = {}
    if metrics_path.exists():
        try:
            with metrics_path.open() as f:
                existing = json.load(f)
            if "random_split" in existing:
                all_metrics["random_split"] = existing["random_split"]
            if "grouped_split" in existing:
                all_metrics["grouped_split"] = existing["grouped_split"]
        except json.JSONDecodeError:
            pass
    all_metrics[f"{args.split}_split"] = model.metrics
    with metrics_path.open("w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✓ Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
