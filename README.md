# Telco Customer Churn Prediction

**Predict whether a customer will churn (cancel their subscription) in the next month**  
**Project Type**: End-to-end Machine Learning + Interactive Web Application (CYO Capstone)

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-FF6600?logo=plotly&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?logo=render&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

---

## Business Objective

Predict customer churn for a telecommunications company using demographics and service usage data.  
**Target Variable**: `Churn` (Yes/No – binary classification)  
**Key Business Metric**: **F1-score** (balanced precision/recall) + **ROC-AUC** + **PR-AUC**  
**Critical Risk**: High cost of **False Negatives** (missing customers who will churn).

**Dataset**: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Rows**: 7,043 | **Columns**: 21 (original) → 26 (after feature engineering)

---

## Tech Stack

| Layer              | Technology                                                              | Purpose                        |
|--------------------|-------------------------------------------------------------------------|--------------------------------|
| **Language**       | Python 3.11+                                                            | Core                           |
| **Data Handling**  | pandas, numpy                                                           | EDA & preprocessing            |
| **Visualization**  | matplotlib, seaborn, SHAP                                               | Exploratory analysis & explainability |
| **Preprocessing**  | scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder)         | Pipeline                       |
| **Modeling**       | scikit-learn, XGBoost, LightGBM                                         | Classification                 |
| **Web App**        | Plotly Dash + HTML/CSS/JS                                               | Interactive UI & predictions   |
| **Model Serving**  | Joblib                                                                  | Model persistence              |
| **Deployment**     | Render (Free tier)                                                      | Live web service               |
| **Data Download**  | Kaggle API                                                              | Automated dataset retrieval    |
| **Environment**    | requirements.txt + virtualenv                                           | Reproducibility                |
| **Notebook**       | Jupyter (nbconvert)                                                     | Model training pipeline        |

---

## Models Trained

Four models are trained and compared in the pipeline. Each was chosen for specific strengths:

| Model                  | Strength                                                    |
|------------------------|-------------------------------------------------------------|
| **Logistic Regression**| Baseline, high interpretability, fast inference             |
| **Random Forest**      | Handles non-linear relationships & feature interactions     |
| **XGBoost**            | Best overall performance on tabular data                    |
| **LightGBM**           | Fastest training on large datasets, memory efficient        |

### Evaluation Criteria (in order of priority)

1. F1-score (primary business metric)
2. PR-AUC (handles class imbalance ~26.5% churn rate)
3. ROC-AUC
4. Training time & inference latency

The **champion model** is selected based on stratified 5-fold cross-validation + final test set performance. Class imbalance is handled via `class_weight='balanced'` / `scale_pos_weight`.

---

## How It Works — Full Pipeline

The application is fully automated. Starting `app.py` triggers the entire pipeline end-to-end:

```
app.py
  └── bootstrap.py          ← Checks if data, models & figures exist
        └── download_data.py    ← Downloads dataset from Kaggle API if missing
        └── run_all_notebooks.py ← Runs notebooks sequentially if models/figures missing
              ├── 01_eda_and_preprocessing.ipynb
              ├── 02a_logistic_regression.ipynb
              ├── 02b_random_forest.ipynb
              ├── 02c_xgboost.ipynb
              ├── 02d_lightgbm.ipynb
              └── 03_model_training_evaluation.ipynb
  └── pipeline.py           ← Provides prediction functions to the Dash frontend
```

### Step-by-step breakdown

1. **`app.py`** starts the Dash web application. Before serving any pages it calls `bootstrap.py`.
2. **`bootstrap.py`** checks whether the raw dataset, trained model files, and figures already exist on disk.
   - If anything is missing, it triggers the steps below automatically.
   - If everything is present, it skips straight to launching the app.
3. **`download_data.py`** connects to the Kaggle API using credentials stored in `.env` and downloads the Telco Customer Churn dataset into `data/raw/`.
4. **`run_all_notebooks.py`** executes all Jupyter notebooks in sequential order using `nbconvert`, producing:
   - Processed training/test data in `data/processed/`
   - Trained model `.joblib` files in `src/models/`
   - Evaluation figures in `src/figures/`
   - A champion model (`champion_model.joblib`) selected by highest F1-score
5. **`pipeline.py`** exposes the preprocessing and prediction logic used by the Dash frontend to score new customer inputs in real time.

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [Kaggle account](https://www.kaggle.com/) with an API key

### 1. Clone the repository

```bash
git clone https://github.com/your-username/CYO-Project-MLG382.git
cd CYO-Project-MLG382
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv311
# Windows
venv311\Scripts\activate
# macOS/Linux
source venv311/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API credentials

Create a `.env` file in the project root with your Kaggle credentials:

```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

> You can find your API key at [kaggle.com](https://www.kaggle.com/) → Account → API → Create New Token.

### 5. Run the application

```bash
python src/dash_app/app.py
```

That's it. The bootstrap process will automatically download the dataset, train all models, generate figures, and launch the web app. On first run this will take several minutes. Subsequent runs skip straight to the app.

---

## Running Components Individually

If you prefer to run the pipeline steps manually:

```bash
# Step 1 – Download dataset only
python src/download_data.py

# Step 2 – Run all training notebooks
python src/run_all_notebooks.py

# Step 3 – Run tests
pytest tests/test_pipeline.py

# Step 4 – Launch the web app
python src/dash_app/app.py
```

---

## Testing

```bash
pytest tests/test_pipeline.py
```

The test suite validates that the preprocessing pipeline and prediction functions in `pipeline.py` behave correctly before the app is served.

---

## Project Structure

```
CYO-Project-MLG382/
├── data/
│   ├── raw/                        ← Downloaded by download_data.py (git-ignored)
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/                  ← Generated by notebook 01 (git-ignored)
│       ├── X_train_processed.csv
│       ├── X_test_processed.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── model_comparison.csv
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb      ← EDA + feature engineering + train/test split
│   ├── 02a_logistic_regression.ipynb       ← Logistic Regression training
│   ├── 02b_random_forest.ipynb             ← Random Forest training
│   ├── 02c_xgboost.ipynb                   ← XGBoost training
│   ├── 02d_lightgbm.ipynb                  ← LightGBM training
│   └── 03_model_training_evaluation.ipynb  ← Model comparison + champion selection + SHAP
│
├── src/
│   ├── bootstrap.py                ← Checks environment; triggers download + training if needed
│   ├── download_data.py            ← Kaggle API integration; downloads raw dataset
│   ├── run_all_notebooks.py        ← Executes notebooks sequentially via nbconvert
│   ├── pipeline.py                 ← Preprocessing + prediction functions for the frontend
│   │
│   ├── dash_app/
│   │   ├── app.py                  ← Dash application entry point
│   │   └── assets/
│   │       └── style.css
│   │
│   ├── models/                     ← Trained model files (git-ignored)
│   │   ├── preprocessor.joblib
│   │   ├── champion_model.joblib
│   │   ├── feature_names.joblib
│   │   ├── logistic_regression.joblib
│   │   ├── random_forest.joblib
│   │   ├── xgboost.joblib
│   │   └── lightgbm.joblib
│   │
│   └── figures/                    ← Generated plots (git-ignored)
│       ├── shap_global_importance.png
│       ├── shap_summary.png
│       ├── model_comparison.png
│       └── ...
│
├── tests/
│   └── test_pipeline.py
│
├── .env                            ← Kaggle API credentials (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note**: `.gitignore` excludes all datasets, trained models, and generated figures to keep the repository lightweight and always current. These are regenerated automatically on first run.

---

## Important Notes

- **First run takes time.** Training four models with hyperparameter search (RandomizedSearchCV, 5-fold CV) across the full dataset takes several minutes. Subsequent runs are instant as long as model files are present.
- **Do not manually place files in `data/raw/` or `src/models/`** unless you intend to skip the automated pipeline — `bootstrap.py` checks for their existence to decide whether to regenerate them.
- **Kaggle credentials are required** for the automated dataset download. Without a valid `.env`, `download_data.py` will fail and you will need to manually download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) and place it in `data/raw/`.