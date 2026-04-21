"""
src/bootstrap.py
One-shot setup script called by app.py on every cold start.

What it does (idempotently):
  1. Downloads the raw dataset from Kaggle  — skipped if CSV already exists.
  2. Runs the full notebook pipeline         — skipped if models + figures exist.

The two skip conditions mean a warm restart (models already trained) costs
nothing — bootstrap returns in milliseconds.
"""

import logging
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR         = Path(__file__).parent                 # src/
ROOT_DIR        = SRC_DIR.parent                        # project root
NOTEBOOKS_RUNNER = SRC_DIR / "run_all_notebooks.py"

# Sentinel files: if both exist we consider the pipeline complete
CHAMPION_MODEL  = SRC_DIR / "models" / "champion_model.joblib"
MODEL_COMPARISON_FIG = SRC_DIR / "figures" / "model_comparison.png"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pipeline_complete() -> bool:
    """True if models and key figures are already on disk."""
    return CHAMPION_MODEL.exists() and MODEL_COMPARISON_FIG.exists()


def _run_notebooks() -> None:
    """Execute run_all_notebooks.py as a subprocess (keeps its own logging)."""
    if not NOTEBOOKS_RUNNER.exists():
        raise FileNotFoundError(f"Pipeline runner not found: {NOTEBOOKS_RUNNER}")

    log.info("🚀 Running full notebook pipeline — this may take several minutes …")
    result = subprocess.run(
        [sys.executable, str(NOTEBOOKS_RUNNER)],
        cwd=str(SRC_DIR),        # run from src/ so relative paths inside match
        capture_output=False,    # stream stdout/stderr straight to the console
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Notebook pipeline failed. Check the output above for details.\n"
            "Tip: increase TIMEOUT_SECONDS in run_all_notebooks.py if training is slow."
        )
    log.info("✅ Notebook pipeline completed successfully.")


# ── Public entry point ────────────────────────────────────────────────────────
def run_bootstrap() -> None:
    """
    Called by app.py at startup.
    Guarantees that data, models, and figures are present before Dash loads.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=" * 60)
    log.info("🔧 Bootstrap starting …")

    # ── Step 1: data ──────────────────────────────────────────────
    try:
        from download_data import download_data  # sibling module in src/
        download_data()
    except EnvironmentError as exc:
        # Missing Kaggle credentials — surface clearly and abort
        log.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        log.error(f"Data download failed: {exc}")
        sys.exit(1)

    # ── Step 2: models + figures ──────────────────────────────────
    if _pipeline_complete():
        log.info("✅ Models and figures already exist — skipping notebook run.")
    else:
        log.info("⚠️  Models or figures missing — running full pipeline …")
        try:
            _run_notebooks()
        except Exception as exc:
            log.error(str(exc))
            sys.exit(1)

    log.info("✅ Bootstrap complete — starting Dash app.")
    log.info("=" * 60)


if __name__ == "__main__":
    run_bootstrap()