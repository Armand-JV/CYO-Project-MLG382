import os
import sys
import joblib
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import logging
from flask import send_from_directory

#Runs boostrap to download raw dataset, and run each notebook for deployment purposes.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
sys.path.insert(0, BASE_DIR)   # make src/ importable
 
from bootstrap import run_bootstrap  # noqa: E402
run_bootstrap()

#APP SETUP
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, "/assets/style.css"],
    title="Telco Customer Churn Predictor - MLG382",
    suppress_callback_exceptions=True,
)
#Server setup for deployment (e.g., Render)
server = app.server

# Paths (relative to src/dash_app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "champion_model.joblib")


# Flask route — serves every file inside src/figures/
@app.server.route("/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)


#MODEL LOAD
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(CHAMPION_MODEL_PATH)
    logging.info("✅ Champion model + preprocessor loaded successfully")
except Exception as e:
    logging.error(f"❌ Model load failed: {e}")
    preprocessor = model = None


#CHAMPION FIGURE RESOLUTION
# Maps sklearn/xgb/lgbm class names → the corresponding saved figure filename.
FEATURE_IMPORTANCE_MAP = {
    "LogisticRegression":     "logreg_coefficients.png",
    "RandomForestClassifier": "randomforest_feature_importance.png",
    "XGBClassifier":          "xgboost_feature_importance.png",
    "LGBMClassifier":         "lightgbm_feature_importance.png",
}
FEATURE_IMPORTANCE_FALLBACK = "lightgbm_feature_importance.png"

def resolve_champion_figure(loaded_model) -> tuple[str, str]:
    """
    Returns (figure_filename, model_display_name) for the champion model.
    Falls back to LightGBM if the class isn't in the map.
    """
    if loaded_model is None:
        return FEATURE_IMPORTANCE_FALLBACK, "Unknown"
    class_name = type(loaded_model).__name__
    figure = FEATURE_IMPORTANCE_MAP.get(class_name, FEATURE_IMPORTANCE_FALLBACK)
    logging.info(f"Champion class: {class_name} → figure: {figure}")
    return figure, class_name

CHAMPION_FEATURE_FIGURE, CHAMPION_MODEL_NAME = resolve_champion_figure(model)


#FEATURE ENGINEERING
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['NoInternet'] = (df['InternetService'] == 'No').astype(int)
    df['FiberOptic'] = (df['InternetService'] == 'Fiber optic').astype(int)

    df['Has_TechSupport'] = (df['TechSupport'] == 'Yes').astype(int)
    df['Has_OnlineSecurity'] = (df['OnlineSecurity'] == 'Yes').astype(int)

    df['Has_Streaming'] = (
        (df['StreamingTV'] == 'Yes') |
        (df['StreamingMovies'] == 'Yes')
    ).astype(int)

    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72, float('inf')],
        labels=['0-12', '12-24', '24-48', '48-72', '72+'],
        right=False
    ).astype(str)

    df['AvgMonthlyCharge'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

    return df


#INPUT FIELDS
def create_input(label, input_id, options=None, value_type="text", min_val=None, max_val=None, value=None):
    if options:
        return dbc.Row([
            dbc.Col(html.Label(label, className="form-label col-form-label"), width=5),
            dbc.Col(
                dcc.Dropdown(
                    id=input_id,
                    options=[{"label": o, "value": o} for o in options],
                    value=value or options[0],
                    clearable=False,
                    className="mb-2",
                ),
                width=7,
            )
        ], className="mb-1")
    else:
        return dbc.Row([
            dbc.Col(html.Label(label, className="form-label col-form-label"), width=5),
            dbc.Col(
                dbc.Input(
                    id=input_id,
                    type=value_type,
                    value=value or (0 if value_type == "number" else ""),
                    min=min_val,
                    max=max_val,
                    className="mb-2",
                ),
                width=7,
            )
        ], className="mb-1")


form_inputs = [
    create_input("Gender", "gender", ["Female", "Male"]),
    create_input("Senior Citizen (0=No, 1=Yes)", "SeniorCitizen", ["0", "1"]),
    create_input("Partner", "Partner", ["Yes", "No"]),
    create_input("Dependents", "Dependents", ["Yes", "No"]),
    create_input("Tenure (months)", "tenure", value_type="number", min_val=0, max_val=72, value=12),
    create_input("Phone Service", "PhoneService", ["Yes", "No"]),
    create_input("Multiple Lines", "MultipleLines", ["No phone service", "No", "Yes"]),
    create_input("Internet Service", "InternetService", ["DSL", "Fiber optic", "No"]),
    create_input("Online Security", "OnlineSecurity", ["No", "Yes", "No internet service"]),
    create_input("Online Backup", "OnlineBackup", ["No", "Yes", "No internet service"]),
    create_input("Device Protection", "DeviceProtection", ["No", "Yes", "No internet service"]),
    create_input("Tech Support", "TechSupport", ["No", "Yes", "No internet service"]),
    create_input("Streaming TV", "StreamingTV", ["No", "Yes", "No internet service"]),
    create_input("Streaming Movies", "StreamingMovies", ["No", "Yes", "No internet service"]),
    create_input("Contract", "Contract", ["Month-to-month", "One year", "Two year"]),
    create_input("Paperless Billing", "PaperlessBilling", ["Yes", "No"]),
    create_input("Payment Method", "PaymentMethod", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]),
    create_input("Monthly Charges ($)", "MonthlyCharges", value_type="number", min_val=0, max_val=200, value=70.0),
    create_input("Total Charges ($)", "TotalCharges", value_type="number", min_val=0, max_val=10000, value=800.0),
]

sidebar = dbc.Card(
    dbc.CardBody([
        html.H4("📋 Customer Profile", className="card-title text-primary mb-3"),
        html.Hr(),
        *form_inputs,
        dbc.Button("🚀 Predict Churn", id="predict-btn", color="primary", size="lg", className="w-100 mt-4"),
    ]),
    className="shadow",
)

#LAYOUT
app.layout = dbc.Container([
    dbc.Row(dbc.Col(
        html.H1("Telco Customer Churn Predictor", className="text-center text-primary my-4"),
        width=12
    )),

    dbc.Row([
        dbc.Col(sidebar, width=5),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5(f"📊 Prediction Result: {CHAMPION_MODEL_NAME}", className="mb-0")),
                dbc.CardBody(id="prediction-output", children=[
                    html.Div(
                        "Fill the form on the left and click 'Predict Churn'",
                        className="text-muted text-center py-5"
                    )
                ]),
            ], className="shadow mb-4"),

            dbc.Card([
                dbc.CardBody([
                    dcc.Tabs([
                        dcc.Tab(
                            label="Model Comparison",
                            children=html.Img(
                                src="/figures/model_comparison.png",
                                style={"width": "100%", "max-height": "auto"}
                            )
                        ),
                        # Dynamically resolves to the champion model's own figure
                        dcc.Tab(
                            label=f"Feature Importance ({CHAMPION_MODEL_NAME})",
                            children=html.Img(
                                src=f"/figures/{CHAMPION_FEATURE_FIGURE}",
                                style={"width": "100%", "max-height": "auto"}
                            )
                        ),
                        dcc.Tab(
                            label="SHAP Summary",
                            children=html.Img(
                                src="/figures/shap_summary.png",
                                style={"width": "100%", "max-height": "auto"}
                            )
                        ),
                    ]),
                ])
            ], className="shadow"),
        ], width=7),
    ], className="g-4"),

    html.Footer(
        html.P(
            f"MLG382 CYO Project • Champion: {CHAMPION_MODEL_NAME} • Deployed with Dash",
            className="text-center text-muted small mt-5"
        ),
    ),
], fluid=True, className="py-4")


#CALLBACK
@callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [
        State("gender", "value"), State("SeniorCitizen", "value"), State("Partner", "value"),
        State("Dependents", "value"), State("tenure", "value"), State("PhoneService", "value"),
        State("MultipleLines", "value"), State("InternetService", "value"),
        State("OnlineSecurity", "value"), State("OnlineBackup", "value"),
        State("DeviceProtection", "value"), State("TechSupport", "value"),
        State("StreamingTV", "value"), State("StreamingMovies", "value"),
        State("Contract", "value"), State("PaperlessBilling", "value"),
        State("PaymentMethod", "value"), State("MonthlyCharges", "value"),
        State("TotalCharges", "value"),
    ],
    prevent_initial_call=True,
)
def predict_churn(n_clicks, *args):
    if preprocessor is None or model is None:
        return html.Div("❌ Model not loaded. Check console logs.", className="text-danger")

    input_dict = {
        "gender": args[0],
        "SeniorCitizen": int(args[1]),
        "Partner": args[2],
        "Dependents": args[3],
        "tenure": float(args[4]),
        "PhoneService": args[5],
        "MultipleLines": args[6],
        "InternetService": args[7],
        "OnlineSecurity": args[8],
        "OnlineBackup": args[9],
        "DeviceProtection": args[10],
        "TechSupport": args[11],
        "StreamingTV": args[12],
        "StreamingMovies": args[13],
        "Contract": args[14],
        "PaperlessBilling": args[15],
        "PaymentMethod": args[16],
        "MonthlyCharges": float(args[17]),
        "TotalCharges": float(args[18]),
    }

    try:
        X_raw = pd.DataFrame([input_dict])
        X_engineered = engineer_features(X_raw)
        X_processed = preprocessor.transform(X_engineered)

        proba = model.predict_proba(X_processed)[0, 1]
        prediction = "CHURN" if proba >= 0.5 else "NO CHURN"
        risk = "🔴 High" if proba >= 0.7 else "🟡 Medium" if proba >= 0.4 else "🟢 Low"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba * 100,
            title={"text": "Churn Probability"},
            delta={"reference": 50, "increasing": {"color": "red"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if proba >= 0.5 else "forestgreen"},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"},
                ],
            }
        ))
        gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=10))

        return [
            html.H3(prediction, className=f"text-{'danger' if 'CHURN' in prediction else 'success'} display-4 text-center"),
            html.H5(f"Probability: {proba:.1%}  •  Risk Level: {risk}", className="text-center"),
            dcc.Graph(figure=gauge, config={"displayModeBar": False}),
            html.Hr(),
            html.P("✅ Feature engineering applied • Matches trained pipeline", className="text-success small"),
        ]

    except Exception as e:
        return html.Div(f"❌ Prediction error: {str(e)}", className="text-danger")


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )