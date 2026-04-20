import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback

# Layout components
sidebar = dbc.Navbar(
    [
        dbc.NavLink("Predict Churn", href="#", className="nav-link fw-bold", id="predict-tab"),
        dbc.NavLink("Model Info", href="#", className="nav-link", id="info-tab"),
    ],
    className="sidebar flex-column p-3",
    id="sidebar",
    style={"height": "100vh"}
)

header = dbc.Row(
    [
        dbc.Col(
            [
                html.H1("🏢 Telco Churn Predictor", className="brand-title mb-1"),
                html.P("Powered by Champion ML Model (XGBoost/LightGBM)", 
                       className="brand-subtitle lead")
            ],
            className="col-12"
        )
    ],
    className="dash-header justify-content-center text-center"
)

# Input form fields - based on key Telco features
input_fields = dbc.Row([
    # Left column: Demographics & Contract
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H5("Customer Profile", className="input-label mb-3"),
                dbc.Label("Gender", className="input-label"),
                dcc.Dropdown(
                    id="gender-dropdown",
                    options=[
                        {"label": "Male", "value": "Male"},
                        {"label": "Female", "value": "Female"}
                    ],
                    value="Male",
                    className="form-control mb-3"
                ),
                dbc.Label("Senior Citizen", className="input-label"),
                dbc.Switch(id="senior-citizen", value=False, className="mb-3"),
                dbc.Label("Partner", className="input-label"),
                dbc.Switch(id="partner", value=False, className="mb-3"),
                dbc.Label("Dependents", className="input-label"),
                dbc.Switch(id="dependents", value=False, className="mb-3"),
                dbc.Label("Contract", className="input-label"),
                dcc.Dropdown(
                    id="contract-dropdown",
                    options=[
                        {"label": "Month-to-month", "value": "Month-to-month"},
                        {"label": "One year", "value": "One year"},
                        {"label": "Two year", "value": "Two year"}
                    ],
                    value="Month-to-month",
                    className="form-control mb-3"
                ),
            ])
        ], className="input-group-custom h-100")
    ], width=6),

    # Right column: Services & Billing
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H5("Services & Billing", className="input-label mb-3"),
                dbc.Label("Tenure (months)", className="input-label"),
                dcc.Slider(
                    id="tenure-slider",
                    min=0, max=72, value=12,
                    marks={i: str(i) for i in range(0, 73, 12)},
                    className="mb-4"
                ),
                dbc.Label("Monthly Charges ($)", className="input-label"),
                dcc.Slider(
                    id="monthly-charges",
                    min=18, max=120, value=70,
                    marks={18: '$18', 70: '$70', 120: '$120'},
                    className="mb-4"
                ),
                dbc.Label("Payment Method", className="input-label"),
                dcc.Dropdown(
                    id="payment-method",
                    options=[
                        {"label": "Electronic check", "value": "Electronic check"},
                        {"label": "Mailed check", "value": "Mailed check"},
                        {"label": "Bank transfer (automatic)", "value": "Bank transfer (automatic)"},
                        {"label": "Credit card (automatic)", "value": "Credit card (automatic)"}
                    ],
                    value="Electronic check",
                    className="form-control mb-3"
                ),
                dbc.Label("Paperless Billing", className="input-label"),
                dbc.Switch(id="paperless-billing", value=True, className="mb-3"),
                dbc.Label("Internet Service", className="input-label"),
                dcc.Dropdown(
                    id="internet-service",
                    options=[
                        {"label": "DSL", "value": "DSL"},
                        {"label": "Fiber optic", "value": "Fiber optic"},
                        {"label": "No", "value": "No"}
                    ],
                    value="DSL",
                    className="form-control"
                ),
            ])
        ], className="input-group-custom h-100")
    ], width=6)
])

predict_button = dbc.Button(
    "🔮 Predict Churn Risk",
    id="submit-button",
    color="success",
    className="btn-predict mt-4 mb-5",
    n_clicks=0
)

# Output section
output_row = dbc.Row([
    # Prediction & Probability
    dbc.Col([
        html.Div(id="prediction-output", className="card-output churn-medium text-center p-4 mb-4"),
        dcc.Graph(id="probability-gauge", className="prob-gauge", config={"displayModeBar": False})
    ], width=6),

    # Risk & SHAP
    dbc.Col([
        html.Div(id="risk-level", className="card-output p-4 mb-4"),
        dcc.Graph(id="shap-waterfall", className="shap-container", config={"displayModeBar": False})
    ], width=6)
])

main_content = dbc.Container([
    header,
    dbc.Row([
        dbc.Col(sidebar, width=2, className="d-none d-lg-block"),
        dbc.Col([
            input_fields,
            predict_button,
            output_row
        ], width={"size": 10, "offset": 0})
    ])
], fluid=True, className="p-4")

def create_layout():
    return dbc.Container([
        dcc.Store(id="customer-inputs"),
        main_content
    ], fluid=True)

