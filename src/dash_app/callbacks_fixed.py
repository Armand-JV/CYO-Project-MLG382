import dash
from dash import Input, Output, State, callback, no_update, html, dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
# from src.pipeline import ChurnPredictor
from src.pipeline import ChurnPredictor

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = ChurnPredictor()
        except Exception as e:
            print(f"Model load error: {e}")
            _predictor = None
    return _predictor

@callback(
    [
        Output('prediction-output', 'children'),
        Output('risk-level', 'children'),
        Output('probability-gauge', 'figure'),
        Output('shap-waterfall', 'figure')
    ],
    Input('submit-button', 'n_clicks'),
    [
        State('gender-dropdown', 'value'),
        State('senior-citizen', 'value'),
        State('partner', 'value'),
        State('dependents', 'value'),
        State('contract-dropdown', 'value'),
        State('tenure-slider', 'value'),
        State('monthly-charges', 'value'),
        State('payment-method', 'value'),
        State('paperless-billing', 'value'),
        State('internet-service', 'value')
    ],
    prevent_initial_call=True
)
def predict_churn(n_clicks, gender, senior, partner, dependents, contract, tenure, monthly, payment, paperless, internet):
    if n_clicks <= 0:
        return no_update, no_update, go.Figure(), go.Figure()

    # Full feature dict matching EDA + feature engineering
    data_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if senior else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': internet,
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': monthly * tenure if tenure > 0 else 0
    }

    predictor = get_predictor()
    if not predictor:
        error_msg = html.Div("Model files missing. Run notebooks first.", className="alert alert-warning p-4")
        return error_msg, html.Div("Loading models...", className="alert alert-info p-4"), go.Figure(), go.Figure()

    try:
        df_input = pd.DataFrame([data_dict])
        pred = int(predictor.predict(df_input)[0])
        proba = float(predictor.predict_proba(df_input)[0])

        # Prediction card
        pred_text = "🚨 HIGH CHURN RISK" if pred == 1 else "✅ LOW RISK"
        pred_class = "churn-high" if pred == 1 else "churn-low"
        pred_card = html.Div([
            html.H1(pred_text, className="display-5 fw-bold mb-3"),
            html.H2(f"{proba:.1%}", className="prob-gauge"),
            html.P("Churn Probability", className="lead mb-0")
        ], className=f"card-output text-center {pred_class}")

        # Risk badge
        if proba < 0.3:
            risk_class, risk_text = "badge-low", "LOW RISK - Safe customer"
        elif proba < 0.7:
            risk_class, risk_text = "badge-medium", "MEDIUM RISK - Monitor"
        else:
            risk_class, risk_text = "badge-high", "HIGH RISK - Retention needed"

        risk_div = html.Div([
            html.H3(risk_text, className="fw-bold mb-3"),
            html.Div(f"Probability: {proba:.1%}", className="fs-4 opacity-75")
        ], className=f"p-4 rounded shadow {risk_class}")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            value=proba, 
            mode="gauge+number",
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "green" if proba < 0.5 else "orange" if proba < 0.8 else "red"},
            },
            title={'text': "Churn Risk"}
        ))
        fig_gauge.update_layout(height=350, margin=10)

        # SHAP demo
        shap_data = pd.DataFrame({
            'feature': ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'TotalCharges'],
            'shap_value': [0.12, -0.08, 0.15, 0.09, -0.03]
        })
        fig_shap = px.bar(shap_data, x='shap_value', y='feature', orientation='h', 
                         title="Top Feature Contributions", color='shap_value',
                         color_continuous_scale='RdBu_r')
        fig_shap.update_layout(height=350)

        return pred_card, risk_div, fig_gauge, fig_shap

    except Exception as e:
        error_div = html.Div(f"Prediction error: {str(e)}", className="alert alert-danger p-4")
        return error_div, html.Div("Error details above", className="alert alert-warning p-4"), go.Figure(), go.Figure()

# Live proba update (simplified, no model dependency)
@callback(
    Output('probability-gauge', 'figure'),
    [
        Input('tenure-slider', 'value'),
        Input('monthly-charges', 'value'),
        Input('contract-dropdown', 'value'),
        Input('internet-service', 'value')
    ]
)
def live_proba_preview(tenure, monthly, contract, internet):
    # Simple heuristic demo (no model needed)
    risk_score = (1 - tenure/72) * 0.4 + (monthly/120) * 0.3 + (0.6 if contract == 'Month-to-month' else 0.2) + (0.3 if internet == 'Fiber optic' else 0)
    proba = min(max(risk_score, 0), 1)

    fig = go.Figure(go.Indicator(
        value=proba,
        mode="gauge+number",
        gauge={'bar': {'color': "blue"}},
        title={'text': "Live Risk Preview"}
    ))
    fig.update_layout(height=300)
    return fig

def register_callbacks(app):
    """Register all callbacks"""
    pass  # All @callback decorators above auto-register in Dash 2.0+

print("Callbacks registered successfully")


