# Telco Customer Churn Prediction

**Predict whether a customer will churn (cancel their subscription) in the next month**  
**Project Type**: End-to-end Machine Learning + Interactive Web Application (CYO Capstone)

![Project Banner](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Project Banner](https://img.shields.io/badge/Dash-FF6600?logo=plotly&logoColor=white)
![Project Banner](https://img.shields.io/badge/Render-46E3B7?logo=render&logoColor=white)
![Project Banner](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

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

| Layer              | Technology                                      | Purpose |
|--------------------|-------------------------------------------------|---------|
| **Language**       | Python 3.10+                                    | Core |
| **Data Handling**  | pandas, numpy                                   | EDA & preprocessing |
| **Visualization**  | matplotlib, seaborn                             | Exploratory analysis |
| **Preprocessing**  | scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder) | Pipeline |
| **Modeling**       | scikit-learn, XGBoost, LightGBM                 | Classification |
| **Web App**        | Plotly Dash + HTML/CSS/JS                       | Interactive UI & predictions |
| **Model Serving**  | Joblib                                          | Model persistence |
| **Deployment**     | Render (Free tier)                              | Live web service |
| **Environment**    | requirements.txt + virtualenv                   | Reproducibility |
| **Notebook**       | Jupyter                                         | Phase 1 & 2 |

---

## Models Used & Their Purpose
0X_model_training_evaluation.ipynb will train and compare four models. Each was chosen for specific strengths: 

- **Logistic Regression** (Baseline, high interpretability, fast inference)
- **Random Forrest** (Handles non-linear relationships & interactions)
- **XGBoost** (Best overall performance on tabular data)
- **LightGBM** (Fastest training on large datasets, memory efficient)

### Evaluation Criteria (in order of priority):

- F1-score (primary business metric)
- PR-AUC (handles class imbalance ~26.5% churn)
- ROC-AUC
- Training time & inference latency

Champion Model will be selected based on stratified 5-fold CV + final test set performance. Class imbalance will be handled via class_weight='balanced' / scale_pos_weight / SMOTE.

---
## Dev guide
****IMPORTANT: The .gitignore doesn't commit any datasets.  Also .gitignore doesn't commit any models or figures as to stay current and not overbloat the repo.****

### Dependancies
To make sure you have all the dependancies before trying to run and make sure you are on Python 3.10+
RUN:

npm install -r requirements.txt

### Dataset Download
Download the dataset at the link below and put it in ***/data/raw***.

https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data
#### Model details and creation
Look inside notebooks directory and run 01_eda_and_preprocessing.ipynb first it will create the **training data for you to use** to train the other models listed above. 
The idea goes to run each notebook sequentially acording to its number 1, 2, 3 and so on. 
Each model notebook needs to create a **.joblib** model.
Then the last notebook will run the models against each other to test which one is the most accurate it will be decided the winner:
0X_model_training_evaluation.ipynb

---

## Project Structure

```bash
└── CYO-Project-MLG382
    └── .github
        └── workflows
    └── data
        └── processed
            ├── X_test_processed.csv
            ├── X_train_processed.csv
            ├── y_test.csv
            ├── y_train.csv
        └── raw
            ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
    └── notebooks
        ├── 01_eda_and_preprocessing.ipynb
        ├── 02a_logistic_regression.ipynb
    └── src
        └── dash_app
        └── figures
            ├── churn_by_category.png
            ├── correlation_with_churn.png
            ├── logreg_evaluation.png
            ├── num_distributions.png
            ├── tenure_vs_churn.png
        └── models
            ├── logistic_regression.joblib
            ├── logreg_search.joblib
            ├── preprocessor.joblib
        └── utils
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```
