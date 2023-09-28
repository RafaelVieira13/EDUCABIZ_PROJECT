import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

FONT_AWESOME = (
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
)

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB, FONT_AWESOME])
server = app.server

sidebar = dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.Div(page["name"], className="ms-2"),
                    ],
                    href=page["path"],
                    active="exact",
                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
            className="bg-light",
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div(html.Img(src='assets/logo_educabiz3.png', style={'display': 'block', 'margin': '0 auto', 'height':'8vh'}),
                         style={'textAlign':'center'}))
    ]),

    html.Hr(),

    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        ]
    )
], fluid=True)


if __name__ == "__main__":
    app.run(debug=True)