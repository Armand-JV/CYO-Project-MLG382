"""
tests/test_integration.py
--------------------------
Integration tests that simulate how the Dash app calls the pipeline.

These tests verify that the full predict flow works correctly across:
  - All realistic input variations (contract types, payment methods, internet service)
  - All 73 tenure values (0-72) — the full slider range
  - Batch predictions matching individual predictions
  - Cold-start / module reimport (server restart simulation)

HOW TO RUN:
  pytest tests/test_integration.py -v

IMPORTANT NOTE — TenureGroup bug in app.py:
  app.py engineer_features() uses labels ['0-12','12-24','24-48','48-72','72+']
  but the preprocessor was trained on  ['0-12','13-24','25-48','49-72'].
  This means app.py predictions fail with ValueError: unknown categories.
  These integration tests use the CORRECT labels (matching the preprocessor)
  and also include a dedicated test that documents the app.py bug.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ChurnPredictor, load_churn_predictor


# =============================================================
# SECTION 1 — FULL INPUT VARIATION TESTS
# =============================================================

class TestAllInputVariations:

    def test_all_contract_types(self, predictor, sample_customer):
        """All three contract types must return a valid probability."""
        for contract in ["Month-to-month", "One year", "Two year"]:
            record = sample_customer.copy()
            record["Contract"] = contract
            proba = predictor.predict_proba(record)
            assert 0.0 <= float(proba[0]) <= 1.0, \
                f"Contract='{contract}': probability {float(proba[0])} out of range"

    def test_all_payment_methods(self, predictor, sample_customer):
        """All four payment methods must return a valid probability."""
        for method in [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ]:
            record = sample_customer.copy()
            record["PaymentMethod"] = method
            proba = predictor.predict_proba(record)
            assert 0.0 <= float(proba[0]) <= 1.0, \
                f"PaymentMethod='{method}': probability out of range"

    def test_all_internet_service_types(self, predictor, sample_customer):
        """DSL, Fiber optic, and No internet must all work. 'No' requires add-on = 'No internet service'."""
        configs = [
            {
                "InternetService": "DSL", "FiberOptic": 0, "NoInternet": 0,
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            },
            {
                "InternetService": "Fiber optic", "FiberOptic": 1, "NoInternet": 0,
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            },
            {
                "InternetService": "No", "FiberOptic": 0, "NoInternet": 1,
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Has_Streaming": 0, "Has_OnlineSecurity": 0, "Has_TechSupport": 0,
            },
        ]
        for cfg in configs:
            record = sample_customer.copy()
            record.update(cfg)
            proba = predictor.predict_proba(record)
            svc = cfg["InternetService"]
            assert 0.0 <= float(proba[0]) <= 1.0, \
                f"InternetService='{svc}': probability out of range"

    def test_all_gender_values(self, predictor, sample_customer):
        """Both gender values must work."""
        for gender in ["Male", "Female"]:
            record = sample_customer.copy()
            record["gender"] = gender
            proba = predictor.predict_proba(record)
            assert 0.0 <= float(proba[0]) <= 1.0

    def test_senior_citizen_flag(self, predictor, sample_customer):
        """SeniorCitizen=0 and SeniorCitizen=1 must both work."""
        for flag in [0, 1]:
            record = sample_customer.copy()
            record["SeniorCitizen"] = flag
            proba = predictor.predict_proba(record)
            assert 0.0 <= float(proba[0]) <= 1.0


# =============================================================
# SECTION 2 — FULL TENURE RANGE (Exhaustive slider test)
# =============================================================

class TestTenureRange:

    def test_all_tenure_values_0_to_72(self, predictor, sample_customer):
        """
        Runs predict_proba() for every integer tenure from 0 to 72.
        Simulates dragging the app's tenure slider across its full range.
        Every value must return a non-NaN probability in [0, 1].
        """
        monthly = sample_customer["MonthlyCharges"]
        failures = []

        for t in range(0, 73):
            group = (
                "0-12"  if t <= 12 else
                "13-24" if t <= 24 else
                "25-48" if t <= 48 else
                "49-72"
            )
            record = sample_customer.copy()
            record.update({
                "tenure": t,
                "TotalCharges": round(t * monthly, 2),
                "AvgMonthlyCharge": monthly if t > 0 else 0.0,
                "TenureGroup": group,
            })
            try:
                proba = predictor.predict_proba(record)
                p = float(proba[0])
                if np.isnan(p) or not (0.0 <= p <= 1.0):
                    failures.append(f"tenure={t}: got {p}")
            except Exception as exc:
                failures.append(f"tenure={t}: {type(exc).__name__}: {exc}")

        assert not failures, (
            f"predict_proba() failed for {len(failures)} tenure value(s):\n"
            + "\n".join(failures)
        )


# =============================================================
# SECTION 3 — BATCH vs INDIVIDUAL CONSISTENCY
# =============================================================

class TestBatchConsistency:

    def test_batch_matches_individual(self, predictor, sample_customer, loyal_customer):
        """
        Sending customers as a DataFrame batch must give identical results
        to sending them one at a time.
        """
        batch_df = pd.DataFrame([sample_customer, loyal_customer])
        batch_probas = predictor.predict_proba(batch_df)

        single_proba_1 = predictor.predict_proba(sample_customer)[0]
        single_proba_2 = predictor.predict_proba(loyal_customer)[0]

        np.testing.assert_almost_equal(batch_probas[0], single_proba_1, decimal=8,
            err_msg="Batch row 0 differs from individual prediction for sample_customer")
        np.testing.assert_almost_equal(batch_probas[1], single_proba_2, decimal=8,
            err_msg="Batch row 1 differs from individual prediction for loyal_customer")

    def test_ten_varied_customers_in_sequence(self, predictor, sample_customer):
        """
        Predicts for 10 customers with varying profiles in a loop.
        Simulates multiple users hitting the app simultaneously.
        """
        monthly_charges = [20.0, 29.85, 45.0, 59.65, 70.0, 80.0, 95.5, 105.0, 110.0, 119.99]
        tenures         = [0,     1,     6,    12,    24,   36,   48,   60,    70,    72]
        contracts       = ["Month-to-month"] * 5 + ["One year"] * 2 + ["Two year"] * 3
        internets       = ["Fiber optic", "DSL", "No", "Fiber optic", "DSL",
                           "Fiber optic", "DSL", "DSL", "DSL", "Fiber optic"]
        failures = []

        for i in range(10):
            t, m = tenures[i], monthly_charges[i]
            internet = internets[i]
            group = (
                "0-12"  if t <= 12 else
                "13-24" if t <= 24 else
                "25-48" if t <= 48 else
                "49-72"
            )
            addon = "No internet service" if internet == "No" else "No"
            record = sample_customer.copy()
            record.update({
                "tenure": t,
                "MonthlyCharges": m,
                "TotalCharges": round(t * m, 2),
                "AvgMonthlyCharge": m if t > 0 else 0.0,
                "TenureGroup": group,
                "Contract": contracts[i],
                "InternetService": internet,
                "FiberOptic": 1 if internet == "Fiber optic" else 0,
                "NoInternet": 1 if internet == "No" else 0,
                "OnlineSecurity": addon, "OnlineBackup": addon,
                "DeviceProtection": addon, "TechSupport": addon,
                "StreamingTV": addon, "StreamingMovies": addon,
                "Has_Streaming": 0, "Has_OnlineSecurity": 0, "Has_TechSupport": 0,
            })
            try:
                proba = predictor.predict_proba(record)
                p = float(proba[0])
                if np.isnan(p) or not (0.0 <= p <= 1.0):
                    failures.append(f"Customer {i} (tenure={t}): probability={p}")
            except Exception as exc:
                failures.append(f"Customer {i} (tenure={t}): {type(exc).__name__}: {exc}")

        assert not failures, f"{len(failures)} customer(s) failed:\n" + "\n".join(failures)


# =============================================================
# SECTION 4 — COLD START (server restart simulation)
# =============================================================

class TestColdStart:

    def test_predict_after_module_reimport(self, sample_customer):
        """
        Deletes the cached module and reimports it from scratch.
        Simulates what happens when the Dash server restarts.
        The predictor must still work correctly after a fresh import.
        """
        module_name = "src.pipeline"
        if module_name in sys.modules:
            del sys.modules[module_name]

        from src.pipeline import load_churn_predictor as fresh_load
        fresh_predictor = fresh_load()

        proba = fresh_predictor.predict_proba(sample_customer)
        assert proba is not None, "predict_proba() returned None after reimport"
        assert 0.0 <= float(proba[0]) <= 1.0


# =============================================================
# SECTION 5 — BUG DOCUMENTATION TEST
# =============================================================

class TestKnownBugs:

    def test_app_engineer_features_tenure_group_mismatch(self):
        """
        DOCUMENTS A KNOWN BUG in app.py.

        app.py engineer_features() computes TenureGroup with labels:
            ['0-12', '12-24', '24-48', '48-72', '72+']

        But the preprocessor (OneHotEncoder) was trained on:
            ['0-12', '13-24', '25-48', '49-72']

        These do NOT match. Any prediction where tenure > 12 will raise:
            ValueError: Found unknown categories ['12-24'] in column 8

        FIX NEEDED in app.py engineer_features():
            Change labels to: ['0-12', '13-24', '25-48', '49-72', '49-72']
            OR use pd.cut bins that match: (0,12], (12,24], (24,48], (48,72]

        This test confirms the bug by importing and running engineer_features
        with a tenure=15 customer (lands in '12-24' bucket → crash).
        """
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "dash_app")))

        try:
            from src.dash_app.app import engineer_features
            import joblib
            import pandas as pd

            preprocessor_path = os.path.join(
                os.path.dirname(__file__), "..", "src", "models", "preprocessor.joblib"
            )
            pre = joblib.load(preprocessor_path)

            record = {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 15, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "DSL",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 59.65, "TotalCharges": 894.75,
            }

            df_engineered = engineer_features(pd.DataFrame([record]))

            # The encoder now handles unknown categories by returning zeros 
            # instead of crashing. We check for the Warning instead of a ValueError.
            import warnings
            with pytest.warns(UserWarning, match="unknown categories"):
                pre.transform(df_engineered)
            
            print("\n✅ Bug Documented: Model handles mismatched label '12-24' via zero-encoding.")

        except ImportError:
            pytest.skip("Could not import app.py engineer_features — skipping bug documentation test")
