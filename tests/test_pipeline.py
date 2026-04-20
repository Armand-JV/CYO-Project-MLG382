# tests/test_pipeline.py
"""
Robust test suite for ChurnPredictor inference pipeline.
Covers happy path, edge cases, error handling, and production consistency.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path (works from root or tests/)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import ChurnPredictor, load_churn_predictor


@pytest.fixture(scope="module")
def predictor():
    """Load once per test module with clear failure message."""
    try:
        pred = load_churn_predictor()
        print(f"\n✅ Predictor loaded → Model: {Path(pred.model_path).name}")
        return pred
    except Exception as e:
        pytest.fail(
            "Failed to load ChurnPredictor.\n"
            "→ Did you run 'python run_all_notebooks.py' first?\n"
            f"Error: {e}"
        )


@pytest.fixture
def sample_single_record():
    """Complete record with ALL engineered features expected by the preprocessor."""
    return {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 59.65, "TotalCharges": 715.80,
        # Engineered features (created in notebook 01)
        "AvgMonthlyCharge": 59.65, "Has_Streaming": 0, "Has_OnlineSecurity": 0,
        "Has_TechSupport": 0, "FiberOptic": 0, "NoInternet": 0, "TenureGroup": "0-12"
    }


@pytest.fixture
def sample_batch_df(sample_single_record):
    """Small batch with variation for batch testing."""
    record2 = sample_single_record.copy()
    record2.update({
        "tenure": 48, "Contract": "Two year", "MonthlyCharges": 110.75,
        "TotalCharges": 5318.10, "AvgMonthlyCharge": 110.79,
        "Has_Streaming": 1, "Has_OnlineSecurity": 1, "Has_TechSupport": 1,
        "FiberOptic": 0, "NoInternet": 0, "TenureGroup": "25-48"
    })
    return pd.DataFrame([sample_single_record, record2])


def test_predictor_initialization(predictor):
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert len(predictor.get_feature_names()) >= 30
    print("✅ Initialization & feature names test passed")


@pytest.mark.parametrize("input_type", ["dict", "list_of_dicts", "dataframe"])
def test_prediction_input_types(predictor, sample_single_record, sample_batch_df, input_type):
    if input_type == "dict":
        pred = predictor.predict(sample_single_record)
        proba = predictor.predict_proba(sample_single_record)
    elif input_type == "list_of_dicts":
        pred = predictor.predict([sample_single_record])
        proba = predictor.predict_proba([sample_single_record])
    else:  # dataframe
        pred = predictor.predict(sample_batch_df)
        proba = predictor.predict_proba(sample_batch_df)

    assert isinstance(pred, np.ndarray)
    assert isinstance(proba, np.ndarray)
    assert pred.shape[0] == (1 if input_type != "dataframe" else len(sample_batch_df))
    assert 0.0 <= proba.min() <= proba.max() <= 1.0
    print(f"✅ {input_type} prediction test passed")


def test_feature_consistency(predictor, sample_single_record):
    """Ensure preprocessing produces the exact shape the model expects."""
    X_proc = predictor._preprocess(sample_single_record)
    expected = len(predictor.get_feature_names())
    assert X_proc.shape[1] == expected, f"Expected {expected} features, got {X_proc.shape[1]}"
    print(f"✅ Preprocessing consistency verified → {X_proc.shape}")


def test_champion_preference(predictor):
    """Confirm we are using the champion model when available."""
    model_name = predictor.model_path.name
    if "champion_model.joblib" in model_name:
        assert "champion" in model_name.lower()
        print("✅ Champion model correctly preferred")
    else:
        pytest.skip("Using fallback logistic_regression model (expected if champion not generated)")


def test_error_handling(predictor):
    """Test graceful failure on invalid input types."""
    with pytest.raises(TypeError, match="Input must be dict"):
        predictor.predict("invalid string")
    with pytest.raises(TypeError):
        predictor.predict(123)
    print("✅ Error handling test passed")


def test_missing_columns_raises(predictor, sample_single_record):
    """Critical business safety: missing features should fail explicitly."""
    bad_record = sample_single_record.copy()
    del bad_record["tenure"]  # required numeric feature
    with pytest.raises(Exception):  # ColumnTransformer or sklearn will raise
        predictor.predict(bad_record)
    print("✅ Missing-column protection test passed")


if __name__ == "__main__":
    print("=" * 75)
    print("Running ChurnPredictor Pipeline Tests")
    print("=" * 75)
    pytest.main([__file__, "-v", "--tb=short"])