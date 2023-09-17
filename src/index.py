import dash
from dash import dcc, html, Output, Input, State
import dash_labs as dl
import dash_bootstrap_components as dbc

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from pages import Visão_Geral, Escola_Especifica

#========= SideBar ============#
offcanvas = html.Div(
    [
        dbc.Button('Explore', id='open-offcanvas', n_clicks=0),
        dbc.Offcanvas(
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(page['name'], href=page['path'])
                    for page in dash.page_registry.values()
                    if page['module'] != 'pages.not_found_404'
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