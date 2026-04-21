from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        base_dir = Path(__file__).parent
        model_dir = base_dir / "models"
        
        # Champion → fallback to logistic_regression
        champion_path = model_dir / "champion_model.joblib"
        logreg_path = model_dir / "logistic_regression.joblib"
        preprocessor_path = model_dir / "preprocessor.joblib"
        
        if champion_path.exists():
            self.model_path = champion_path
        elif logreg_path.exists():
            self.model_path = logreg_path
        else:
            raise FileNotFoundError("No model found. Please re-run notebooks/01_... and 02a_...")
        
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        # Load with extra safety
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        try:
            self.feature_names = list(self.preprocessor.get_feature_names_out())
        except:
            self.feature_names = None

    def predict(self, data):
        X = self._preprocess(data)
        return self.model.predict(X)

    def predict_proba(self, data):
        X = self._preprocess(data)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.ravel()

    def _preprocess(self, data):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data) if isinstance(data, pd.Series) else data.copy()
        
        # Force numeric conversion on any numeric columns (safety net)
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
                
        return self.preprocessor.transform(df)

def load_churn_predictor():
    return ChurnPredictor()