# src/pipeline.py
""" 
ChurnPredictor - Production inference pipeline for Telco Customer Churn 
Supports single record and batch inference + SHAP explanations.
"""

import pandas as pd 
import numpy as np
import joblib
from pathlib import Path
from typing import Union, List, Dict, Any
import shap

class ChurnPredictor:
    """ 
    Loads preprocessor + best model and provides clean predict / predict_proba / SHAP API.
    """

    def __init__(self):
        # Paths relative to src/ (works when imported from root or tests)
        base_dir = Path(__file__).parent
        model_dir = base_dir / "models"

        # Prefer champion, fallback to logistic_regression (as saved by notebook)
        champion_path = model_dir / "champion_model.joblib"
        logreg_path = model_dir / "logistic_regression.joblib"
        preprocessor_path = model_dir / "preprocessor.joblib"

        if champion_path.exists():
            self.model_path = champion_path
        elif logreg_path.exists():
            self.model_path = logreg_path
            print("⚠️ Using logistic_regression.joblib (rename to champion_model.joblib if desired)")
        else:
            raise FileNotFoundError(
                f"Model not found in {model_dir}\\n"
                "Please re-run notebooks/02a_logistic_regression.ipynb to generate the model."
            )

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Re-run 01_eda_and_preprocessing.ipynb")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        try:
            self.feature_names = list(self.preprocessor.get_feature_names_out())
        except Exception:
            self.feature_names = None

        # SHAP explainer for tree-based models
        self.explainer = None
        self.is_tree_based = False
        if hasattr(self.model, 'tree_') or 'xgb' in str(type(self.model)).lower() or 'lgbm' in str(type(self.model)).lower():
            try:
                if hasattr(self.model, 'tree_'):  # sklearn trees (RF)
                    self.explainer = shap.TreeExplainer(self.model)
                elif 'xgb' in str(type(self.model)).lower():
                    self.explainer = shap.TreeExplainer(self.model)
                elif 'lgbm' in str(type(self.model)).lower():
                    self.explainer = shap.TreeExplainer(self.model)
                self.is_tree_based = self.explainer is not None
                print(f"✅ SHAP TreeExplainer ready")
            except ImportError:
                print("⚠️  SHAP not available")

        print(f"✅ ChurnPredictor initialized")
        print(f"   Model        : {self.model_path.name}")
        print(f"   Preprocessor : preprocessor.joblib")
        print(f"   Features     : {len(self.feature_names) if self.feature_names else 'N/A'}")
        print(f"   SHAP         : {'Enabled' if self.is_tree_based else 'N/A (non-tree model)'}")

    def _preprocess(self, data: Union[pd.DataFrame, dict, list, pd.Series]) -> np.ndarray:
        """Internal preprocessing - handles single dict or batch."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            df = pd.DataFrame(data) if isinstance(data, pd.Series) else data.copy()
        else:
            raise TypeError("Input must be dict, list of dicts, or pandas DataFrame")

        return self.preprocessor.transform(df)

    def predict(self, data: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """Binary prediction: 0 = No Churn, 1 = Churn"""
        X_proc = self._preprocess(data)
        return self.model.predict(X_proc)

    def predict_proba(self, data: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """Probability of churn (class 1)"""
        X_proc = self._preprocess(data)
        proba = self.model.predict_proba(X_proc)
        return proba[:, 1] if proba.shape[1] == 2 else proba.ravel()

    def get_shap_explanation(self, data: Union[pd.DataFrame, dict, list]) -> Dict[str, Any]:
        """Compute SHAP values for explanation. Returns dict for Dash viz."""
        if not self.is_tree_based:
            raise ValueError("SHAP only for tree-based models (XGBoost/RF/LGBM)")

        X_proc = self._preprocess(data)
        shap_values = self.explainer.shap_values(X_proc)

        # For binary: churn class (index 1)
        if isinstance(shap_values, list):
            shap_values_churn = shap_values[1]
        else:
            shap_values_churn = shap_values[:, 1] if shap_values.ndim > 1 else shap_values

        base_value = self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value

        return {
            'shap_values': shap_values_churn.flatten() if len(shap_values_churn.shape) > 1 else shap_values_churn,
            'base_value': float(base_value),
            'data': X_proc.flatten(),
            'feature_names': self.feature_names
        }

    def get_feature_names(self) -> List[str]:
        """Return feature names after preprocessing"""
        if self.feature_names is None:
            raise ValueError("Feature names not available from preprocessor")
        return self.feature_names


# Factory function
def load_churn_predictor() -> ChurnPredictor:
    return ChurnPredictor()
