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

# Define the layout for your app here
app.layout = dbc.Container(
    [
        # Add your Dash components here
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.NavItem(dbc.NavLink("Page 1", href="/page-1")),
                dbc.NavItem(dbc.NavLink("Page 2", href="/page-2")),
            ],
            brand="Your App",
            brand_href="/",
            color="primary",
            dark=True,
        ),
        # Additional content can be added here
    ],
    fluid=True,
)