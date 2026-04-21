# tests/test_pipeline.py
"""
Comprehensive tests for ChurnPredictor pipeline (hardened version).
Tests single/batch prediction, error handling, and the new safety fix for corrupted data.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import ChurnPredictor, load_churn_predictor


@pytest.fixture(scope="module")
def predictor():
    """Load the predictor once for all tests."""
    try:
        pred = load_churn_predictor()
        print(f"\n✅ Predictor loaded successfully → Model: {Path(pred.model_path).name}")
        return pred
    except Exception as e:
        pytest.fail(f"Failed to load ChurnPredictor.\n"
                    f"Make sure you re-ran notebooks/01_eda_and_preprocessing.ipynb "
                    f"and 02a_logistic_regression.ipynb (or champion model).\nError: {e}")


@pytest.fixture
def sample_single_record():
    """Complete raw customer record with all engineered features."""
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

        # Engineered features (required by preprocessor)
        "AvgMonthlyCharge": 59.65,
        "Has_Streaming": 0,
        "Has_OnlineSecurity": 0,
        "Has_TechSupport": 0,
        "FiberOptic": 0,
        "NoInternet": 0,
        "TenureGroup": "0-12"
    }


@pytest.fixture
def sample_batch_df(sample_single_record):
    """Create a small batch with variation."""
    record2 = sample_single_record.copy()
    record2.update({
        "tenure": 48,
        "Contract": "Two year",
        "MonthlyCharges": 110.75,
        "TotalCharges": 5318.10,
        "AvgMonthlyCharge": 110.79,
        "Has_Streaming": 1,
        "Has_OnlineSecurity": 1,
        "Has_TechSupport": 1,
        "FiberOptic": 1,
        "NoInternet": 0,
        "TenureGroup": "25-48"
    })
    return pd.DataFrame([sample_single_record, record2])


# ========================== CORE TESTS ==========================
def test_predictor_initialization(predictor):
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    print("✅ Predictor initialization passed")


def test_single_record_prediction(predictor, sample_single_record):
    pred = predictor.predict(sample_single_record)
    proba = predictor.predict_proba(sample_single_record)

    assert isinstance(pred, np.ndarray) and pred.shape == (1,)
    assert isinstance(proba, np.ndarray) and proba.shape == (1,)
    assert pred[0] in [0, 1]
    assert 0.0 <= proba[0] <= 1.0

    print(f"✅ Single record test passed → Churn: {pred[0]}, Probability: {proba[0]:.4f}")


def test_batch_prediction(predictor, sample_batch_df):
    preds = predictor.predict(sample_batch_df)
    probas = predictor.predict_proba(sample_batch_df)

    assert len(preds) == len(sample_batch_df) == 2
    assert len(probas) == len(sample_batch_df) == 2
    print(f"✅ Batch prediction test passed ({len(preds)} records)")


def test_feature_names(predictor):
    names = predictor.get_feature_names()
    assert isinstance(names, list)
    assert len(names) >= 30  # one-hot + numeric + binary features
    print(f"✅ Feature names test passed ({len(names)} features)")


# ========================== SAFETY FIX TEST ==========================
def test_safety_fix_for_corrupted_numeric_strings(predictor):
    """Test the new safety fix for scientific notation strings like '5E-1'."""
    corrupted_record = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": "1.68E+3",           # ← This would previously break
        "AvgMonthlyCharge": "7E+1",          # ← Scientific notation string
        "Has_Streaming": 0,
        "Has_OnlineSecurity": 0,
        "Has_TechSupport": 0,
        "FiberOptic": 1,
        "NoInternet": 0,
        "TenureGroup": "13-24"
    }

    # Should NOT raise ValueError or TypeError
    pred = predictor.predict(corrupted_record)
    proba = predictor.predict_proba(corrupted_record)

    assert pred[0] in [0, 1]
    assert 0.0 <= proba[0] <= 1.0
    print("✅ Safety fix for scientific notation strings (e.g. '5E-1') passed")


# ========================== ERROR HANDLING ==========================
def test_error_handling(predictor):
    """Test that invalid inputs raise clear errors."""
    with pytest.raises(TypeError):
        predictor.predict("invalid string input")
    with pytest.raises(TypeError):
        predictor.predict(12345)
    with pytest.raises(TypeError):
        predictor.predict_proba(None)
    print("✅ Error handling test passed")


def test_preprocessing_consistency(predictor, sample_single_record):
    """Verify output shape matches expected feature count."""
    X_proc = predictor._preprocess(sample_single_record)
    expected_cols = len(predictor.get_feature_names())
    assert X_proc.shape[1] == expected_cols
    print(f"✅ Preprocessing consistency verified → output shape: {X_proc.shape}")


# ========================== RUN TESTS ==========================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Running Full ChurnPredictor Pipeline Tests (Hardened Version)")
    print("=" * 80)
    pytest.main([__file__, "-v", "--tb=short"])