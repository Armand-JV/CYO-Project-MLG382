import os
import dash
import dash_bootstrap_components as dbc
from .layout import create_layout
from .callbacks_fixed import register_callbacks

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Telco Churn Predictor | ML-Powered Risk Assessment"

# Layout
app.layout = create_layout()

# Callbacks
register_callbacks(app)

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
