import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Output, Input, State


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

# Connect to your app pages
from pages import Visão_Geral, Escola_Especifica

#========= SideBar ============#
offcanvas = html.Div(
    [
        dbc.Button('Explore', id='open-offcanvas', n_clicks=0),
        dbc.Offcanvas(
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(pages['name'], href=pages['path'])
                    for pages in dash.page_registry.values()
                    if pages['module'] != 'pages.not_found_404'
                ]
            ),
            id='offcanvas',
            is_open=False,
        ),
    ],
    className='my-3'
)

#=========  Layout  =========== #
app.layout = dbc.Container(html.Div([
    offcanvas,dash.page_container
]), fluid=True)

@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

if __name__ == "__main__":
    app.run_server(debug=True, port=8001)