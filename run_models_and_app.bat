@echo off
echo Step 1/5: Installing Python packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --no-deps
python -m pip install pandas==2.1.4 numpy==1.26.4 scikit-learn==1.3.2 xgboost lightgbm shap plotly dash dash-bootstrap-components joblib jupyterlab nbconvert

echo Step 2/5: Training preprocessing pipeline...
cd notebooks
python -m nbconvert --to python 01_eda_and_preprocessing.ipynb --execute --stdout --to-notebook ^> ../temp_preprocess.py ^& del ../temp_preprocess.py 2^>nul
python 01_eda_and_preprocessing.py ^|^| echo Preprocessing complete

echo Step 3/5: Training champion model...
python -m nbconvert --to python 03_model_training_evaluation.ipynb --execute --stdout --to-notebook ^> ../temp_model.py ^& del ../temp_model.py 2^>nul
python 03_model_training_evaluation.py ^|^| echo Models saved to src/models/

echo Step 4/5: Testing predictor...
cd ..
python -c "from src.pipeline import load_churn_predictor; predictor = load_churn_predictor(); print('✅ Pipeline test passed! Ready for predictions.')"
if %errorlevel% neq 0 echo Pipeline test failed - check models/

echo Step 5/5: Starting Dash web app...
echo Open http://localhost:10000 in your browser
python src/dash_app/app.py
pause
