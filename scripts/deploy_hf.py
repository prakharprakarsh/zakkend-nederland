"""Prepare the spaces/ directory for Hugging Face Spaces deployment.

This script copies all necessary source code and the trained model into
the spaces/ directory, creating a self-contained deployable package.

Usage
-----
    python scripts/deploy_hf.py

Then push to Hugging Face:
    cd spaces/
    git init
    git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/zakkend-nederland
    git add .
    git commit -m "Deploy Zakkend Nederland"
    git push -u origin main
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPACES_DIR = PROJECT_ROOT / "spaces"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    print("=" * 60)
    print("  Preparing HF Spaces deployment")
    print("=" * 60)

    # 1. Copy source package into spaces/src/
    spaces_src = SPACES_DIR / "src"
    if spaces_src.exists():
        shutil.rmtree(spaces_src)

    print(f"→ Copying source code to {spaces_src}")
    shutil.copytree(
        SRC_DIR / "zakkend",
        spaces_src / "zakkend",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )

    # 2. Copy trained model
    model_src = MODELS_DIR / "subsidence_xgb.joblib"
    if not model_src.exists():
        print("✗ No trained model found! Run training first:")
        print("  python scripts/train.py")
        print("  — or —")
        print("  python scripts/fetch_and_train.py --municipalities Gouda --train")
        sys.exit(1)

    spaces_models = SPACES_DIR / "models"
    spaces_models.mkdir(exist_ok=True)
    dest_model = spaces_models / "subsidence_xgb.joblib"
    print(f"→ Copying model to {dest_model}")
    shutil.copy2(model_src, dest_model)

    # Also copy metrics.json if it exists
    metrics_src = MODELS_DIR / "metrics.json"
    if metrics_src.exists():
        shutil.copy2(metrics_src, spaces_models / "metrics.json")

    # 3. Create data directories (empty, for consistency)
    for d in ["data", "data/raw", "data/processed"]:
        (SPACES_DIR / d).mkdir(parents=True, exist_ok=True)
        (SPACES_DIR / d / ".gitkeep").touch()

    # 4. Summary
    total_files = sum(1 for _ in SPACES_DIR.rglob("*") if _.is_file())
    model_size_mb = dest_model.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"  ✓ Deployment package ready!")
    print(f"{'=' * 60}")
    print(f"  Directory:  {SPACES_DIR}")
    print(f"  Files:      {total_files}")
    print(f"  Model size: {model_size_mb:.1f} MB")
    print()
    print("  Next steps:")
    print("  1. Create a Space at https://huggingface.co/new-space")
    print("     Name: zakkend-nederland")
    print("     SDK: Streamlit")
    print("     Visibility: Public")
    print()
    print("  2. Push to HF:")
    print("     cd spaces/")
    print("     git init")
    print("     git remote add origin https://huggingface.co/spaces/<YOUR_USERNAME>/zakkend-nederland")
    print("     git add .")
    print('     git commit -m "Deploy Zakkend Nederland"')
    print("     git push -u origin main")
    print()
    print("  3. Wait ~2 minutes for the Space to build.")
    print("     Your demo URL: https://huggingface.co/spaces/<YOUR_USERNAME>/zakkend-nederland")


if __name__ == "__main__":
    main()
