import dash
import dash_bootstrap_components as dbc

# Define the external stylesheets
external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
    "https://fonts.googleapis.com/icon?family=Material+Icons",
    dbc.themes.LITERA  # Applying the "LITERA" theme
]

# Additional CSS for dbc components
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"

# Create a Dash app instance
app = dash.Dash(__name__, external_stylesheets=external_stylesheets + [dbc_css])

# Suppress callback exceptions and serve scripts locally
app.config['suppress_callback_exceptions'] = True
app.scripts.config.serve_locally = True

# Define the server
server = app.server