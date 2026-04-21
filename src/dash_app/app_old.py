import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Local imports
try:
    from pipeline import load_churn_predictor
    predictor = load_churn_predictor()
    MODEL_READY = True
except Exception as e:
    print(f"Model not ready: {e}. Run notebooks first.")
    MODEL_READY = False
    predictor = None

# External CSS
external_stylesheets = [dbc.themes.BOOTSTRAP, '/assets/style.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# Raw feature categories from 01_eda_and_preprocessing.ipynb
NUM_FEATURES = {
    'tenure': {'min': 0, 'max': 72, 'step': 1, 'label': 'Tenure (months)'},
    'MonthlyCharges': {'min': 18, 'max': 120, 'step': 1, 'label': 'Monthly Charges ($)'},
    'TotalCharges': {'min': 0, 'max': 8684, 'step': 100, 'label': 'Total Charges ($)'},
    'AvgMonthlyCharge': {'min': 18, 'max': 110, 'step': 1, 'label': 'Avg Monthly Charge ($)'}
}

BINARY_FEATURES = [
    'SeniorCitizen',  # 0/1
    'Has_Streaming', 'Has_OnlineSecurity', 'Has_TechSupport', 'FiberOptic', 'NoInternet'
]

CAT_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'TenureGroup'
]

CAT_OPTIONS = {  # Default options from dataset
    'gender': ['Male', 'Female'],
    'Partner': ['Yes', 'No'], 'Dependents': ['Yes', 'No'], 'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    'TenureGroup': ['0-12', '13-24', '25-48', '49-72', '72+']
}

# Layout sections
sidebar = dbc.NavbarSimple(
    children=[
        html.H3('Telco Churn Predictor', className='mb-0', style={'color': 'white'}),
        html.Hr(),
        html.P([
            'Champion Model: ', html.Strong(id='model-info', children='Loading...'),
            html.Br(), 'F1-Score: ', html.Strong(id='f1-score', children='N/A')
        ], className='mt-4 mb-0')
    ],
    brand_href='#',
    className='sidebar vh-100 flex-column align-items-start p-4 sticky-top',
    dark=True,
    expand=True
)

def create_input_section(title, items):
    """Dynamic input row"""
    cols = []
    for i, item in enumerate(items):
        if isinstance(item, dict):  # Slider
            cols.append(
                dbc.Col([
                    html.Label(item['label'], className='form-label fw-500'),
                    dcc.Slider(
                        id=f'{title.lower().replace(" ", "-")}-{list(item.keys())[0].lower()}',
                        min=item['min'], max=item['max'], step=item['step'], value=(item['min']+item['max'])/2,
                        marks=None, tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], lg=12)
            )
        elif isinstance(item, str):  # Category or binary
            if item in BINARY_FEATURES:
                cols.append(
                    dbc.Col([
                        dbc.FormCheck(
                            id=f'binary-{item.lower()}',
                            label=item.replace('_', ' ').title(),
                            className='form-switch',
                            checked=False,
                            switch=True
                        )
                    ], lg=6)
                )
            else:
                options = [{'label': v, 'value': v} for v in CAT_OPTIONS.get(item, ['Yes', 'No'])]
                cols.append(
                    dbc.Col([
                        dbc.Label(item.replace('_', ' ').title(), className='form-label fw-500'),
                        dcc.Dropdown(
                            id=f'cat-{item.lower()}',
                            options=options,
                            value=options[0]['value'] if options else None,
                            clearable=False
                        )
                    ], lg=6)
                )
    return dbc.Row([
        dbc.Col([
            html.H5(title, className='input-group-title'),
            *cols
        ])
    ], className='input-group fade-in mb-4')

inputs = html.Div([
    create_input_section('Demographics', ['gender', 'SeniorCitizen', 'Partner', 'Dependents']),
    create_input_section('Services', ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineBackup', 'DeviceProtection']),
    create_input_section('Support & Streaming', ['Has_OnlineSecurity', 'Has_TechSupport', 'Has_Streaming']),
    create_input_section('Billing & Contract', ['Contract', 'PaperlessBilling', 'PaymentMethod']),
    create_input_section('Tenure & Charges', list(NUM_FEATURES.keys()) + ['TenureGroup', 'NoInternet', 'FiberOptic'])
], className='p-4')

# Output sections
output_card = dbc.Card([
    dbc.CardBody([
        html.Div(id='prediction-output', className='text-center py-5')
    ])
], className='mt-4 prediction-card')

shap_div = html.Div(id='shap-explanation', className='shap-container mt-4 card p-4', style={'display': 'none'})

main_content = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, md=3, className='d-none d-md-block'),
        dbc.Col([
            html.H1('Customer Churn Prediction Dashboard', className='text-center mb-5 fade-in', style={'color': 'var(--primary-blue)'}),
            inputs,
            dbc.Button('Predict Churn Risk', id='predict-btn', color='primary', className='btn-primary-custom w-100 mb-4 fw-bold fs-5', n_clicks=0, disabled=not MODEL_READY),
            output_card,
            shap_div
        ], md=9)
    ])
], fluid=True, className='py-4 px-3')

app.layout = dbc.Container([
    dcc.Store(id='input-data-store'),
    main_content
], fluid=True, className='min-vh-100')

# Callbacks
@callback(
    Output('model-info', 'children'),
    Input('predict-btn', 'n_clicks')
)
def update_model_info(n_clicks):
    if not MODEL_READY:
        return 'Not Ready - Run notebooks first'
    try:
        model_name = predictor.model_path.name.replace('.joblib', '')
        return model_name
    except:
        return 'Error loading model'

@callback(
    Output('input-data-store', 'data'),
    [Input(f'{type_id}-{feat.lower()}', 'value') for type_id, feats in 
     [('cat', CAT_FEATURES), ('binary', BINARY_FEATURES)] for feat in feats] +
    [Input(f'num-{feat.lower()}', 'value') for feat in NUM_FEATURES],
    prevent_initial_call=True
)
def collect_inputs(*values):
    # Collect all input values into dict
    input_data = {}
    # ... (simplified - map values to feature names)
    # In full impl: zip with feature lists
    return input_data

@callback(
    [Output('prediction-output', 'children'),
     Output('shap-explanation', 'style')],
    Input('predict-btn', 'n_clicks'),
    State('input-data-store', 'data'),
    prevent_initial_call=True
)
def make_prediction(n_clicks, input_data):
    if not MODEL_READY or not input_data:
        raise PreventUpdate

    try:
        # Input as dict
        input_df = pd.DataFrame([input_data])

        # Predict
        prob = predictor.predict_proba(input_df)[0]
        pred = predictor.predict(input_df)[0]

        # Risk level
        risk_class = 'low' if prob < 0.3 else 'medium' if prob < 0.6 else 'high'
        risk_label = risk_class.title()
        risk_style = f'risk-{risk_class}'

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.6], 'color': "yellow"},
                    {'range': [0.6, 1], 'color': "darkred"}
                ],
                'threshold': {
                    'line': {"color": "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prob
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))

        gauge = dcc.Graph(figure=fig_gauge, className='gauge-container mb-4')

        # Prediction text
        pred_text = "Will Churn 🚨" if pred == 1 else "Will Stay ✅"
        risk_badge = html.Span(risk_label, className=risk_style)

        output = html.Div([
            html.H2(pred_text, className='fade-in', style={'color': 'var(--success-green)' if pred == 0 else 'var(--accent-orange)'}),
            html.H4(f'{prob:.1%} Probability', className='fade-in'),
            risk_badge,
            gauge
        ], className='fade-in')

        # SHAP
        shap_data = predictor.get_shap_explanation(input_df)
        shap_style = {'display': 'block'}

        # Simple SHAP bar (full waterfall needs shap.html, approx with plotly)
        shap_df = pd.DataFrame({
            'feature': shap_data['feature_names'],
            'shap': shap_data['shap_values'],
            'sign': np.sign(shap_data['shap_values'])
        }).sort_values('shap', key=abs, ascending=False).head(10)

        fig_shap = px.bar(shap_df.head(10), x='shap', y='feature', orientation='h',
                          color='sign', color_continuous_scale=['red', 'green'],
                          title='Top SHAP Feature Contributions to Churn')

        shap_viz = dcc.Graph(figure=fig_shap)

        return output, shap_style

    except Exception as e:
        return html.Div(f'Error: {str(e)}', className='alert alert-danger'), {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
