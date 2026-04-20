# tests/test_pipeline.py
"""
Thorough tests for ChurnPredictor pipeline.
Fixed to include all engineered features required by the preprocessor.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import ChurnPredictor, load_churn_predictor


@pytest.fixture(scope="module")
def predictor():
    """Load predictor once for the module."""
    try:
        pred = load_churn_predictor()
        print(f"\n✅ Predictor loaded successfully → Model: {Path(pred.model_path).name}")
        return pred
    except Exception as e:
        pytest.fail(f"Failed to load predictor.\n"
                    f"Make sure you re-ran notebooks/01_eda_and_preprocessing.ipynb and 02a_logistic_regression.ipynb\nError: {e}")


@pytest.fixture
def sample_single_record():
    """Complete raw record INCLUDING all engineered features created in Phase 1."""
    return {
        # Original columns
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

        # === ENGINEERED FEATURES (mandatory for preprocessor) ===
        "AvgMonthlyCharge": 59.65,                    # TotalCharges / tenure
        "Has_Streaming": 0,                           # (StreamingTV == 'Yes' or StreamingMovies == 'Yes')
        "Has_OnlineSecurity": 0,                      # OnlineSecurity == 'Yes'
        "Has_TechSupport": 0,                         # TechSupport == 'Yes'
        "FiberOptic": 0,                              # InternetService == 'Fiber optic'
        "NoInternet": 0,                              # InternetService == 'No'
        "TenureGroup": "0-12"                         # pd.cut result
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
        "FiberOptic": 0,
        "NoInternet": 0,
        "TenureGroup": "25-48"
    })
    return pd.DataFrame([sample_single_record, record2])


def test_predictor_initialization(predictor):
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    print("✅ Predictor initialization passed")


def test_single_record_prediction(predictor, sample_single_record):
    """Test single customer prediction."""
    pred = predictor.predict(sample_single_record)
    proba = predictor.predict_proba(sample_single_record)

    assert isinstance(pred, np.ndarray) and pred.shape == (1,)
    assert isinstance(proba, np.ndarray) and proba.shape == (1,)
    assert pred[0] in [0, 1]
    assert 0.0 <= proba[0] <= 1.0

    print(f"✅ Single record test passed → Churn: {pred[0]}, Probability: {proba[0]:.4f}")


def test_batch_prediction(predictor, sample_batch_df):
    """Test batch prediction with DataFrame."""
    preds = predictor.predict(sample_batch_df)
    probas = predictor.predict_proba(sample_batch_df)

    assert len(preds) == len(sample_batch_df)
    assert len(probas) == len(sample_batch_df)
    print(f"✅ Batch prediction test passed ({len(preds)} records)")


def test_feature_names(predictor):
    names = predictor.get_feature_names()
    assert isinstance(names, list)
    assert len(names) >= 30  # one-hot + numeric + binary
    print(f"✅ Feature names test passed ({len(names)} features)")


def test_error_handling(predictor):
    with pytest.raises(TypeError):
        predictor.predict("invalid string")
    with pytest.raises(TypeError):
        predictor.predict(123)
    print("✅ Error handling test passed")


def test_preprocessing_consistency(predictor, sample_single_record):
    """Verify internal preprocessing produces correct shape."""
    X_proc = predictor._preprocess(sample_single_record)
    expected_cols = len(predictor.get_feature_names())
    assert X_proc.shape[1] == expected_cols
    print(f"✅ Preprocessing consistency verified → output shape: {X_proc.shape}")


if __name__ == "__main__":
    print("=" * 75)
    print("🚀 Starting Thorough ChurnPredictor Pipeline Tests")
    print("=" * 75)
    pytest.main([__file__, "-v", "--tb=short"])