"""Fetch real BAG data for target municipalities and optionally train.

This is the "one-command Phase 2" experience:
    python scripts/fetch_and_train.py --municipalities Gouda
    python scripts/fetch_and_train.py --municipalities Gouda Rotterdam --train

Equivalent to running:
    python -m zakkend.data.pipeline --municipalities Gouda Rotterdam
    python scripts/train.py --data data/processed/real_data.parquet
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from zakkend import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--municipalities",
        nargs="+",
        default=["Gouda"],
        help=f"Available: {list(config.TARGET_MUNICIPALITIES.keys())}",
    )
    parser.add_argument("--max-per-muni", type=int, default=5000)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Also train the model after fetching",
    )
    parser.add_argument(
        "--augment-synthetic",
        type=int,
        default=3000,
        help="Synthetic rows to add during training (0 to disable)",
    )
    args = parser.parse_args()

    out_path = config.PROCESSED_DATA_DIR / "real_data.parquet"

    # Step 1: Fetch real data
    print("=" * 60)
    print("  STEP 1: Fetching real building data from PDOK")
    print("=" * 60)

    fetch_cmd = [
        sys.executable,
        "-m",
        "zakkend.data.pipeline",
        "--municipalities",
        *args.municipalities,
        "--max-per-muni",
        str(args.max_per_muni),
        "--out",
        str(out_path),
    ]
    result = subprocess.run(fetch_cmd)
    if result.returncode != 0:
        print("✗ Data fetching failed!")
        sys.exit(1)

    if not out_path.exists():
        print("✗ Output file not created!")
        sys.exit(1)

    # Step 2: Train (optional)
    if args.train:
        print("\n" + "=" * 60)
        print("  STEP 2: Training model on real data")
        print("=" * 60)

        train_cmd = [
            sys.executable,
            "scripts/train.py",
            "--data",
            str(out_path),
        ]
        if args.augment_synthetic > 0:
            train_cmd += ["--augment-synthetic", str(args.augment_synthetic)]

        result = subprocess.run(train_cmd)
        if result.returncode != 0:
            print("✗ Training failed!")
            sys.exit(1)

    print("\n✓ Phase 2 pipeline complete!")
    print(f"  Data: {out_path}")
    if args.train:
        print(f"  Model: {config.MODEL_PATH}")
        print("\n  Launch demo: python -m uvicorn zakkend.api.main:app --reload")


if __name__ == "__main__":
    main()
