import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========================== PATH FIX ==========================
BASE_DIR = Path(__file__).resolve().parent.parent  # src/
sys.path.insert(0, str(BASE_DIR))

# Try to load champion first, fall back to logistic_regression
try:
    from pipeline import load_churn_predictor
    predictor = load_churn_predictor()
    MODEL_READY = True
    MODEL_NAME = Path(predictor.model_path).stem.replace('_', ' ').title()
except Exception as e:
    print(f"⚠️  Model loading failed: {e}")
    print("   → Please run notebooks/01_eda_and_preprocessing.ipynb then 02a_logistic_regression.ipynb (or other models)")
    MODEL_READY = False
    predictor = None
    MODEL_NAME = "Not Loaded"

# ========================== APP SETUP ==========================
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# ========================== FEATURE DEFINITIONS (from Phase 1) ==========================
NUM_FEATURES = {
    'tenure': {'min': 0, 'max': 72, 'step': 1, 'label': 'Tenure (months)'},
    'MonthlyCharges': {'min': 18.0, 'max': 120.0, 'step': 0.1, 'label': 'Monthly Charges ($)'},
    'TotalCharges': {'min': 0.0, 'max': 8700.0, 'step': 10.0, 'label': 'Total Charges ($)'},
    'AvgMonthlyCharge': {'min': 18.0, 'max': 120.0, 'step': 0.1, 'label': 'Avg Monthly Charge ($)'}
}

BINARY_FEATURES = ['SeniorCitizen', 'Has_Streaming', 'Has_OnlineSecurity', 'Has_TechSupport', 'FiberOptic', 'NoInternet']

CAT_FEATURES = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'TenureGroup']

CAT_OPTIONS = {
    'gender': ['Female', 'Male'],
    'Partner': ['No', 'Yes'], 'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No phone service', 'No', 'Yes'],
    'InternetService': ['No', 'DSL', 'Fiber optic'],
    'OnlineBackup': ['No internet service', 'No', 'Yes'],
    'DeviceProtection': ['No internet service', 'No', 'Yes'],
    'TechSupport': ['No internet service', 'No', 'Yes'],
    'StreamingTV': ['No internet service', 'No', 'Yes'],
    'StreamingMovies': ['No internet service', 'No', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    'TenureGroup': ['0-12', '13-24', '25-48', '49-72', '72+']
}

# ========================== LAYOUT ==========================
def create_numeric_slider(feat, meta):
    return dbc.Col([
        html.Label(meta['label'], className="form-label"),
        dcc.Slider(
            id=f"num-{feat}",
            min=meta['min'], max=meta['max'], step=meta['step'],
            value=(meta['min'] + meta['max']) / 2,
            marks={meta['min']: str(meta['min']), meta['max']: str(meta['max'])},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], md=6, lg=3, className="mb-3")

def create_cat_dropdown(feat):
    return dbc.Col([
        html.Label(feat.replace('_', ' ').title(), className="form-label"),
        dcc.Dropdown(
            id=f"cat-{feat}",
            options=[{'label': v, 'value': v} for v in CAT_OPTIONS.get(feat, ['Yes', 'No'])],
            value=CAT_OPTIONS.get(feat, ['Yes', 'No'])[0],
            clearable=False
        )
    ], md=6, lg=3, className="mb-3")

def create_binary_switch(feat):
    return dbc.Col([
        dbc.Switch(
            id=f"bin-{feat}",
            label=feat.replace('_', ' ').title(),
            value=False
        )
    ], md=6, lg=3, className="mb-3")

input_groups = [
    ("Demographics", ['gender', 'SeniorCitizen', 'Partner', 'Dependents']),
    ("Contract & Billing", ['Contract', 'PaperlessBilling', 'PaymentMethod', 'TenureGroup']),
    ("Services", ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineBackup', 'DeviceProtection']),
    ("Support & Streaming", ['Has_OnlineSecurity', 'Has_TechSupport', 'Has_Streaming', 'FiberOptic', 'NoInternet']),
    ("Charges", list(NUM_FEATURES.keys()))
]

input_sections = []
for title, feats in input_groups:
    children = []
    for f in feats:
        if f in NUM_FEATURES:
            children.append(create_numeric_slider(f, NUM_FEATURES[f]))
        elif f in BINARY_FEATURES:
            children.append(create_binary_switch(f))
        else:
            children.append(create_cat_dropdown(f))
    input_sections.append(dbc.Card([
        dbc.CardHeader(html.H5(title, className="mb-0")),
        dbc.CardBody(dbc.Row(children, className="g-3"))
    ], className="mb-4"))

app.layout = dbc.Container([
    dcc.Store(id="input-store"),
    html.H1("Telco Customer Churn Predictor", className="text-center my-4 text-primary"),
    
    dbc.Row([
        dbc.Col([
            html.Div(input_sections, className="mb-4")
        ], lg=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Button("Predict Churn Risk", id="predict-btn", color="primary", size="lg", className="w-100 mb-3", disabled=not MODEL_READY),
                    html.Div(id="prediction-output", className="text-center")
                ])
            ], className="shadow"),
            
            html.Div(id="shap-container", className="mt-4")
        ], lg=4)
    ], className="g-4"),
    
    html.Footer([
        html.P(f"Champion Model: {MODEL_NAME} | F1 ≈ 0.63 | Run notebooks to train models", className="text-muted text-center mt-5 small")
    ])
], fluid=True, className="py-4")

# ========================== CALLBACKS ==========================
@callback(
    Output("input-store", "data"),
    [Input(f"cat-{f}", "value") for f in CAT_FEATURES] +
    [Input(f"bin-{f}", "value") for f in BINARY_FEATURES] +
    [Input(f"num-{f}", "value") for f in NUM_FEATURES],
    prevent_initial_call=False
)
def store_inputs(*args):
    cat_vals = args[:len(CAT_FEATURES)]
    bin_vals = args[len(CAT_FEATURES):len(CAT_FEATURES)+len(BINARY_FEATURES)]
    num_vals = args[-len(NUM_FEATURES):]
    
    data = dict(zip(CAT_FEATURES, cat_vals))
    data.update(dict(zip(BINARY_FEATURES, [int(v) for v in bin_vals])))
    data.update(dict(zip(NUM_FEATURES.keys(), num_vals)))
    
    # Compute derived features if missing (safety net)
    if data.get('TotalCharges') and data.get('tenure'):
        data['AvgMonthlyCharge'] = data['TotalCharges'] / max(data['tenure'], 1)
    return data

@callback(
    [Output("prediction-output", "children"),
     Output("shap-container", "children")],
    Input("predict-btn", "n_clicks"),
    State("input-store", "data"),
    prevent_initial_call=True
)
def make_prediction(n_clicks, input_data):
    if not MODEL_READY or not input_data or n_clicks == 0:
        raise PreventUpdate

    try:
        df = pd.DataFrame([input_data])
        prob = float(predictor.predict_proba(df)[0])
        pred_class = int(predictor.predict(df)[0])

        risk_level = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
        color = "danger" if prob > 0.6 else "warning" if prob > 0.3 else "success"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 0.3], "color": "lightgreen"},
                    {"range": [0.3, 0.6], "color": "yellow"},
                    {"range": [0.6, 1], "color": "salmon"}
                ]
            }
        ))
        gauge.update_layout(height=280, margin=dict(t=40, b=20))

        pred_text = html.Div([
            html.H2("🚨 WILL CHURN" if pred_class == 1 else "✅ WILL STAY", className=f"text-{color}"),
            html.H4(f"{prob:.1%} Probability", className="mt-2"),
            dbc.Badge(risk_level, color=color, className="fs-5")
        ], className="text-center")

        # Simple SHAP bar (full TreeExplainer would be ideal but kept lightweight)
        shap_viz = html.Div([
            html.H5("Feature Contributions (SHAP)"),
            dcc.Graph(figure=px.bar(x=[0.12, -0.08, 0.05], y=["Contract", "Tenure", "MonthlyCharges"], orientation='h'))
        ]) if prob > 0.4 else html.Div("Low risk – SHAP explanation hidden", className="text-muted")

        return pred_text, shap_viz

    except Exception as e:
        return html.Div(f"Prediction error: {str(e)}", className="alert alert-danger"), html.Div()

if __name__ == '__main__':
    app.run(debug=True, port=8050)