import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

dash.register_page(__name__,
                   path='/escola',  # represents the url text
                   name='Escola',  # name of page, commonly used as name of link
                   title='Escola'  # epresents the title of browser's tab
)

# ========== Styles ============ #
tab_card = {'height': '100%'}

main_config = {
    "hovermode": "x unified",
    "legend": {
        "yanchor": "top",
        "y": 0.9,
        "xanchor": "left",
        "x": 0.1,
        "title": {"text": None},
        "font": {"color": "white"},
        "bgcolor": "rgba(0,0,0,0.5)"
    },
    "margin": {"l": 10, "r": 10, "t": 10, "b": 10}
}

config_graph = {"displayModeBar": False, "showTips": False}

card_icon = {
    'color': 'white',
    'textAlign': 'center',
    'fontSize': 30,
    'margin': 'auto'
}

# Indicators Height
indicator_height = 150

#======== Reading and Cleaning The Data =========#
url = 'https://raw.githubusercontent.com/RafaelVieira13/EDUCABIZ_PROJECT/main/datasets/EDUCABIZ.csv'
df = pd.read_csv(url)

# Missing Values is equal to 0 so let´s replace all the missing values by 0
df.fillna(0,inplace=True)

# Let´s create a column called 'total_interacoes'. I t will be the sum of every numeric column, just to be our 'y'
df['interacoes_totais'] = df.iloc[:,3:12].sum(axis=1)

# Let´s create a column called 'dimension'.
df['dimensao'] = df['tutores'] + df['second_tutor'] + df['mensagens (7_dias)']

#=== Clustering ===#
df1 = df.copy()

# Feature Scalling
sc = StandardScaler()
df2=sc.fit_transform(df1.iloc[:,3:])

# Applying PCA
pca = PCA(n_components=4)
pca_results = pca.fit_transform(df2)

# Training the K-Means Model
np.random.seed(42)
kmeans = KMeans(n_clusters = 3, init = 'k-means++',n_init=10)
y_kmeans = kmeans.fit_predict(pca_results)

# We called the df, that's why we need to refer to previous df to add cluster numbers
df1['Cluster'] = y_kmeans
df1['Cluster'].value_counts()
df1.drop(columns=['dimensao'],inplace=True)

# Rename Columns and Replace Values
df1.rename(columns={'Cluster':'nivel_interacao'},inplace=True)
df1['nivel_interacao'] = df1['nivel_interacao'].replace(0, 'Menor Interação')
df1['nivel_interacao'] = df1['nivel_interacao'].replace(2, 'Interação Intermédia')
df1['nivel_interacao'] = df1['nivel_interacao'].replace(1, 'Elevada Interação')

# Replca the months by numbers in order to apply the renge slider
df1['month'] = df1['month'].replace('Jan',1)
df1['month'] = df1['month'].replace('Fev',2)
df1['month'] = df1['month'].replace('Mar',3)
df1['month'] = df1['month'].replace('Abri',4)
df1['month'] = df1['month'].replace('Mai',5)
df1['month'] = df1['month'].replace('Jun',6)
df1['month'] = df1['month'].replace('Jul',7)
df1['month'] = df1['month'].replace('Ago',8)
df1['month'] = df1['month'].replace('Set',9)
df1['month'] = df1['month'].replace('Out',10)
df1['month'] = df1['month'].replace('Nov',11)
df1['month'] = df1['month'].replace('Dez',12)

df1['month'] = df1['month'].astype(int)


# Let´s Create a dictionary to change the range slider to the month names
months = {
    1: 'Jan',
    2: 'Fev',
    3: 'Mar',
    4: 'Abri',
    5: 'Mai',
    6: 'Jun',
    7: 'Jul',
    8: 'Ago',
    9: 'Set',
    10: 'Out',
    11: 'Nov',
    12: 'Dez'
}

# Data To the line-plot. Use pd.melt() to convert kpi columns into rows
df_line_plot = pd.melt(df1, id_vars=['month','escola','slug','nivel_interacao'], var_name='kpi', value_name='interacoes')
df_line_plot[df_line_plot['escola']=='ABLA']

# =========  Layout  =========== #

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("Escolha Uma Escola:", style={'font-weight': 'bold'}),
                        dcc.Dropdown(options=[{'label': escola, 'value': escola} for escola in df1['escola'].unique()],
                                     id='escola_dropdown')
                    ], xs=10, sm=10, md=8, lg=4, xl=4, xxl=4,
                    style={'margin-bottom': '15px'}
                )
            ]
        ),

        # Indicador Nivel de Interação e Line Plot (Numero de Interacoes)

        dbc.Row(
            [
                dbc.Col([
                    dbc.CardGroup([
                        dbc.Card(
                            dcc.Graph(id='indicador_escola', className='dbc'),
                            style={'maxWidth': '450px', 'height': indicator_height, 'padding-left': '20px',
                                   'padding-top': '10px', 'margin-bottom': '40px'}
                        ),
                        dbc.Card(
                            html.Div(),
                            color='lightsteelblue',
                            style={'maxWidth': '65px', 'height': indicator_height, 'margin-left': '-10px'}
                        )
                    ]),
                    dbc.Card(
                        dcc.Graph(id='kpi_use_escola'),
                        style={'width': '100%', 'height': '450px', 'padding': '10px', 'padding-top': '10px'}
                    )
                ], width=6),

                dbc.Col([
                    dbc.CardGroup([
                        dbc.Card(
                            dbc.CardBody([
                                html.H2('Interações ao Longo do Tempo', className='card-title',
                                        style={'font-weight': 'bold', 'color': '#343a40'}),
                                dcc.RangeSlider(
                                    id='month_slider_escola',
                                    min=df1['month'].min(),
                                    max=df1['month'].max(),
                                    step=1,
                                    marks={i: months[i] for i in range(1, 13)},
                                    value=[df1['month'].min(), df1['month'].max()],
                                ),
                                html.Br(),
                                html.Div('Escolha Um KPI:',style={'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    options=[{'label': kpi, 'value': kpi} for kpi in df1.iloc[:, 3:13].columns],
                                    id='kpi_dropdown_escola', style={'maxWidth': '200px', 'margin-left': '10px'}),
                                dcc.Graph(id='line_kpi_use_escola', className='dbc')
                            ])
                        )
                    ])

                ], width=6)
            ]
        )
    ]
)

# =========  Callbacks  =========== #

# Indicador


@callback(
    Output('indicador_escola', 'figure'),
    [Input('escola_dropdown', 'value')]
)
def indicador(escola):
    df_filtered = df1[df1['escola']==escola]

    df_filtered = df_filtered.groupby(['escola', 'nivel_interacao']).count().reset_index()
    if not df_filtered.empty:

        indicador_escola = go.Figure(go.Indicator(title={'text': f"<span>{df_filtered['escola'].iloc[0]}"}))
        indicador_escola.add_annotation(
            text=df_filtered['nivel_interacao'].iloc[0],
            x=0.5,
            y=0.8,
            showarrow=False,
            font={'size': 24})

        indicador_escola.update_layout(width=400, height=100)

    else:
        indicador_escola = go.Figure(go.Indicator(title=''))
        indicador_escola.add_annotation(
            text="Escolha uma Escola",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={'size': 16})

        indicador_escola.update_layout(width=400, height=100)

    return indicador_escola


# KPI Use

@callback(
    Output('kpi_use_escola', 'figure'),
    [Input('escola_dropdown', 'value')]
)
def indicador_kpi(escola):
    df_filtered = df1[df1['escola'] == escola]

    # BAR CHART -KPI USE
    df_kpi_use = df_filtered.groupby('nivel_interacao').agg(tutores=('tutores', 'sum'),
                                                            second_tutor=('second_tutor', 'sum'),
                                                            docs_fiscais=('docs_fiscais (15_dias)', 'sum'),
                                                            mensagens=('mensagens (7_dias)', 'sum'),
                                                            atividades=('atividades (7_dias)', 'sum'),
                                                            relatorios_diarios=('relatorios_diarios (7_dias)', 'sum'),
                                                            avaliacoes=('avaliacoes (7_dias)', 'sum'),
                                                            menus=('menus (7_dias)', 'sum'),
                                                            eventos=('eventos (15_dias)', 'sum')).reset_index()

    df_kpi_use = pd.melt(df_kpi_use, id_vars=['nivel_interacao'], value_vars=df_kpi_use.iloc[:, 1:])
    df_kpi_use.rename(columns={'variable': 'kpi', 'value': 'numero_interacoes'}, inplace=True)
    df_kpi_use.sort_values(by='numero_interacoes', ascending=False, inplace=True)

    # Define custom colors based on 'nivel_interacao'

    fig_kpi_use = px.bar(df_kpi_use,
                         x='kpi',
                         y='numero_interacoes',
                         color='nivel_interacao',
                         color_discrete_map={'Menor Interação': 'red',
                                             'Interação Intermédia': 'lightgrey',
                                             'Elevada Interação': 'blue'},
                         template='simple_white')

    fig_kpi_use.update_layout(width=700, height=420,
                              title={'text': '<b>Utilização de KPIs<b>',
                                     'font': {'size': 22}},
                              xaxis_title=None,
                              legend_title='Nivel de Interação')

    fig_kpi_use.update_traces(marker=dict(line=dict(color='#000000', width=2)))

    fig_kpi_use.update_yaxes(title_text='Numero de Interações', title_font=dict(size=16), tickfont=dict(size=15))

    fig_kpi_use.update_xaxes(tickfont=dict(size=15))

    return fig_kpi_use


# Line Plot - Kpi by month
@callback(
    Output('line_kpi_use_escola', 'figure'),
     [Input('escola_dropdown', 'value')],
    [Input('month_slider_escola','value')],
    [Input('kpi_dropdown_escola', 'value')]
)

def line_plot_escola(escola,month, kpi):
    df_filtered = df_line_plot[
        (df_line_plot['month'] >= month[0]) & 
        (df_line_plot['month'] <= month[1]) &
        (df_line_plot['escola'] == escola) &
        (df_line_plot['kpi'] == kpi)
    ]
    
    df_plot = df_filtered.groupby('month')['interacoes'].sum().reset_index().sort_values(by='month')
    
    by_month = go.Figure(go.Scatter(
    x=df_plot['month'], 
    y=df_plot['interacoes'], 
    mode='lines',
    line=dict(color='royalblue'),
    fill='tonexty',
    hovertext=[f'Mês: {month}<br>Interaçôes: {interacoes}' for month, interacoes in zip(df_plot['month'], df_plot['interacoes'])]
))
    by_month.update_xaxes(title_text='Mês',title_font={'size': 17},tickfont=dict(size=16))
    by_month.update_yaxes(title_text='Número de Interações',title_font={'size': 17}, tickfont=dict(size=16))

    by_month.update_traces(marker=dict(line=dict(color='#000000', width=2)))

    by_month.update_layout(width=715,
                       height=400,
                       template='simple_white')
    
    return by_month



