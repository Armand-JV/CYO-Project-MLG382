"""
src/download_data.py
Downloads the Telco Customer Churn dataset from Kaggle and saves it to data/raw/.

Requirements
------------
Set these environment variables in your deployment environment BEFORE starting the app:
    KAGGLE_USERNAME=<your_kaggle_username>
    KAGGLE_KEY=<your_kaggle_api_key>

You can generate a key at: https://www.kaggle.com/settings → API → "Create New Token"
"""

import os
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR    = Path(__file__).parent                                    # src/
ROOT_DIR   = SRC_DIR.parent                                           # project root
TARGET_DIR = ROOT_DIR / "data" / "raw"
TARGET_CSV = TARGET_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

KAGGLE_DATASET = "blastchar/telco-customer-churn"
KAGGLE_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def _check_credentials() -> None:
    """Fail fast with a clear message if Kaggle env vars are missing."""
    missing = [v for v in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing Kaggle credentials: {', '.join(missing)}\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY as environment variables.\n"
            "Generate them at: https://www.kaggle.com/settings → API → Create New Token"
        )


def download_data(force: bool = False) -> Path:
    """
    Download the Telco churn CSV from Kaggle into data/raw/.

    Parameters
    ----------
    force : bool
        Re-download even if the file already exists (default False).

    Returns
    -------
    Path to the downloaded CSV.
    """
    if TARGET_CSV.exists() and not force:
        log.info(f"✅ Dataset already present: {TARGET_CSV}")
        return TARGET_CSV

    _check_credentials()
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"📥 Downloading '{KAGGLE_DATASET}' from Kaggle …")

    try:
        import kagglehub  # in requirements.txt
        download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        raise RuntimeError(f"kagglehub download failed: {exc}") from exc

    # kagglehub downloads to a versioned cache dir — locate our CSV
    matches = list(download_path.rglob(KAGGLE_FILENAME))
    if not matches:
        raise FileNotFoundError(
            f"'{KAGGLE_FILENAME}' not found in downloaded archive at {download_path}"
        )

    shutil.copy(matches[0], TARGET_CSV)
    log.info(f"✅ Dataset saved → {TARGET_CSV}")
    return TARGET_CSV


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_data()