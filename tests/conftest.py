"""
conftest.py
-----------
Shared fixtures for the Telco Churn test suite.
pytest loads this file automatically — you never import or run it directly.

IMPORTANT: TenureGroup labels must match what the preprocessor was trained on:
  ['0-12', '13-24', '25-48', '49-72']
  NOTE: app.py engineer_features() uses DIFFERENT labels ('12-24', '24-48' etc.)
        — this is a known bug in app.py that our tests document.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Make src/ importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import load_churn_predictor


@pytest.fixture(scope="module")
def predictor():
    """
    Load ChurnPredictor once per test module (fast — model loaded from disk only once).
    Fails with a clear message if .joblib files are missing.
    """
    try:
        pred = load_churn_predictor()
        print(f"\n✅ Predictor loaded → {Path(pred.model_path).name}")
        return pred
    except Exception as e:
        pytest.fail(
            "Could not load ChurnPredictor.\n"
            "Run: python src/run_all_notebooks.py  to generate the model files.\n"
            f"Error: {e}"
        )


@pytest.fixture
def sample_customer():
    """Standard moderate-risk customer — tenure=12, DSL, Month-to-month."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 59.65,
        "TotalCharges": 715.80,
        "AvgMonthlyCharge": 59.65,
        "Has_Streaming": 0,
        "Has_OnlineSecurity": 0,
        "Has_TechSupport": 0,
        "FiberOptic": 0,
        "NoInternet": 0,
        "TenureGroup": "0-12",
    }


@pytest.fixture
def new_customer():
    """Edge case — brand new customer: tenure=0, TotalCharges=0."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 0.0,
        "AvgMonthlyCharge": 0.0,
        "Has_Streaming": 0,
        "Has_OnlineSecurity": 0,
        "Has_TechSupport": 0,
        "FiberOptic": 0,
        "NoInternet": 0,
        "TenureGroup": "0-12",
    }


@pytest.fixture
def loyal_customer():
    """Low-risk — tenure=72, Two year contract, DSL, all add-ons."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 72,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 80.00,
        "TotalCharges": 5760.00,
        "AvgMonthlyCharge": 80.00,
        "Has_Streaming": 0,
        "Has_OnlineSecurity": 1,
        "Has_TechSupport": 1,
        "FiberOptic": 0,
        "NoInternet": 0,
        "TenureGroup": "49-72",
    }


@pytest.fixture
def high_risk_customer():
    """High-risk — tenure=1, Fiber optic, Month-to-month, senior citizen."""
    return {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 110.00,
        "TotalCharges": 110.00,
        "AvgMonthlyCharge": 110.00,
        "Has_Streaming": 1,
        "Has_OnlineSecurity": 0,
        "Has_TechSupport": 0,
        "FiberOptic": 1,
        "NoInternet": 0,
        "TenureGroup": "0-12",
    }


@pytest.fixture
def sample_batch_df(sample_customer, loyal_customer):
    """Two-row DataFrame for batch prediction tests."""
    return pd.DataFrame([sample_customer, loyal_customer])
