#!/usr/bin/env python3
"""
CYO Project - End-to-End Pipeline Orchestrator
Runs all notebooks in strict sequential order so the project is fully reproducible with one command.
Keeps ALL original notebook logic, outputs, saved artifacts (models, preprocessor, figures) intact.
"""

import subprocess
import sys
from pathlib import Path

# ====================== CONFIG ======================
NOTEBOOK_ORDER = [
    "../notebooks/01_eda_and_preprocessing.ipynb",
    "../notebooks/02a_logistic_regression.ipynb",
    "../notebooks/02b_random_forest.ipynb",
    "../notebooks/02c_xgboost.ipynb",
    "../notebooks/02d_lightgbm.ipynb",
    "../notebooks/03_model_training_evaluation.ipynb",
]

RAW_DATA_PATH = Path("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DATASET_LINK = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data"

TIMEOUT_SECONDS = 900  # 15 minutes per notebook (XGBoost/LightGBM can be heavy)
# ===================================================


def check_raw_data() -> None:
    """Fail fast with helpful message if dataset is missing."""
    if not RAW_DATA_PATH.exists():
        print("❌ Raw dataset not found!")
        print(f"   Please download it from: {DATASET_LINK}")
        print(f"   and save it as: {RAW_DATA_PATH.resolve()}")
        sys.exit(1)
    print(f"✅ Raw dataset found → {RAW_DATA_PATH.name}")


def run_notebook(nb_path: str) -> None:
    """Execute a single notebook in-place using nbconvert (keeps all outputs)."""
    nb_path = Path(nb_path)
    if not nb_path.exists():
        print(f"❌ Notebook not found: {nb_path}")
        sys.exit(1)

    print(f"\n🚀 Running {nb_path.name} ...")
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={TIMEOUT_SECONDS}",
        "--ExecutePreprocessor.kernel_name=python3",
        str(nb_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {nb_path.name} executed successfully")
    else:
        print(f"❌ FAILED: {nb_path.name}")
        print(result.stderr)
        print("\n💡 Tip: Check the notebook for errors or increase TIMEOUT_SECONDS if training is slow.")
        sys.exit(1)


def main() -> None:
    print("=" * 80)
    print("🔄 CYO Telco Churn - Full Pipeline Orchestrator")
    print("=" * 80)

    check_raw_data()

    for notebook in NOTEBOOK_ORDER:
        run_notebook(notebook)

    print("\n🎉 ALL NOTEBOOKS EXECUTED SUCCESSFULLY!")
    print("   • Preprocessor & processed data saved")
    print("   • All models trained & evaluated")
    print("   • Champion model (highest F1) saved to src/models/champion_model.joblib")
    print("   • Figures saved to src/figures/")
    print("\nNext step:")
    print("python -m pytest tests/test_pipeline.py -v")



if __name__ == "__main__":
    main()