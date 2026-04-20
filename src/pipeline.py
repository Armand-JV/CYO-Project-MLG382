# src/pipeline.py
"""
ChurnPredictor - Production inference pipeline for Telco Customer Churn
Supports single record and batch inference.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, List, Dict, Any

class ChurnPredictor:
    """
    Loads preprocessor + best model and provides clean predict / predict_proba API.
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
                f"Model not found in {model_dir}\n"
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
        
        print(f"✅ ChurnPredictor initialized")
        print(f"   Model        : {self.model_path.name}")
        print(f"   Preprocessor : preprocessor.joblib")
        print(f"   Features     : {len(self.feature_names) if self.feature_names else 'N/A'}")
    
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
    
    def get_feature_names(self) -> List[str]:
        """Return feature names after preprocessing"""
        if self.feature_names is None:
            raise ValueError("Feature names not available from preprocessor")
        return self.feature_names


# Factory function
def load_churn_predictor() -> ChurnPredictor:
    return ChurnPredictor()