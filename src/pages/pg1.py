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

# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/',  # '/' is home page and it represents the url
                   name='Visão Geral',  # name of page, commonly used as name of link
                   title='Visão Geral',  # title that appears on browser's tab
                   image='logo_educabiz3.png',  # image in the assets folder
                   
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
indicator_height = 180

#======== Reading and Cleaning The Data =========#
df = pd.read_csv('EDUCABIZ.csv')

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


# =========  Layout  =========== #

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(options= [{'label':nivel, 'value':nivel} for nivel in df1['nivel_interacao'].unique()],
                                     id='nivel_int_dropdown')
                    ], xs=10, sm=10, md=8, lg=4, xl=4, xxl=4,
                    style={'margin-bottom': '15px'}
                )
            ]
        ),
        
        # Indicador Nivel de Interação - Numero de Escolas
        
        dbc.Row(
            [
              dbc.Col([
                  dbc.CardGroup([
                      dbc.Card(
                          dcc.Graph(id='indicador_numero_escolas', className='dbc'),
                          style={'maxWidth':'380px','height':indicator_height,'padding-left':'20px','padding-top': '10px'}
                      ),
                      dbc.Card(
                          html.Div(),
                          color='lightsteelblue',
                          style={'maxWidth':'65px','height':indicator_height, 'margin-left':'-10px'}
                      )
                  ])
              ], width=4),
              
              dbc.Col([
                  dbc.Card(
                      dcc.Graph(id='kpi_use'),style={'height':'100%','padding':'10px'}
                  )
              ], width=8)
            ]
        )
    ]
)



# =========  Functions To Filter the Data  =========== #
def nivel_inter_filter(nivel_interacao):
    if nivel_interacao is None or len(nivel_interacao) == 0:
        mask = df1['nivel_interacao'].isin(df1['nivel_interacao'].unique())
    else:
        if not isinstance(nivel_interacao, list):
            nivel_interacao = [nivel_interacao]  # Convert single value to a list
        mask = df1['nivel_interacao'].isin(nivel_interacao)
    return mask
    
# =========  Callbacks  =========== #
@callback(
    Output('indicador_numero_escolas', 'figure'),
    [Input('nivel_int_dropdown', 'value')]
)
def indicador(nivel_interacao):
    
    # Indicador Numero de Escolas
    if nivel_interacao is None or len(nivel_interacao) == 0:
        # When no nivel_interacao is selected, show the total number of escolas
        total_escolas = df1['escola'].count()
        indicador_escolas = go.Figure(go.Indicator(
            mode='number',
            title={'text': f"<span>Total de Escolas"},
            value=total_escolas,
            number={'valueformat': '.0f','font':{'size':60}}
        ))
        
        indicador_escolas.update_layout(
            width=350,
            height=150)
    else:
        mask = nivel_inter_filter(nivel_interacao)
        df_filtered = df1.loc[mask]

        if not df_filtered.empty:
            # Indicator Nivel Interacao Número de Escolas
            df_filtered = df_filtered.groupby('nivel_interacao')['escola'].count().reset_index()
            indicador_escolas = go.Figure(go.Indicator(
                mode='number',
                title={'text': f"<span>{df_filtered['nivel_interacao'].iloc[0]} e Número de Escolas"},
                value=df_filtered['escola'].iloc[0],
                number={'valueformat': '.0f', 'font':{'size':60}}
            ))
            indicador_escolas.update_layout(
            width=350,
            height=150)
            
        else:
            indicador_escolas = go.Figure()
            indicador_escolas.update_layout(
            width=350,
            height=150)
            

    return indicador_escolas

@callback(
    Output('kpi_use','figure'),
    [Input('nivel_int_dropdown', 'value')]
)
def indicador(nivel_interacao):
    nivel_mask = nivel_inter_filter(nivel_interacao)
    df_filtered = df1.loc[nivel_mask]
    
     # BAR CHART -KPI USE
    df_kpi_use = df_filtered.groupby('nivel_interacao').agg(tutores=('tutores','sum'),
                                                      second_tutor=('second_tutor','sum'),
                                                      docs_fiscais=('docs_fiscais (15_dias)','sum'),
                                                      mensagens=('mensagens (7_dias)','sum'),
                                                      atividades=('atividades (7_dias)','sum'),
                                                      relatorios_diarios=('relatorios_diarios (7_dias)','sum'),
                                                      avaliacoes=('avaliacoes (7_dias)','sum'),
                                                      menus=('menus (7_dias)','sum'),
                                                      eventos=('eventos (15_dias)','sum')).reset_index()

    df_kpi_use = pd.melt(df_kpi_use, id_vars=['nivel_interacao'], value_vars=df_kpi_use.iloc[:,1:])
    df_kpi_use.rename(columns={'variable':'kpi', 'value':'numero_interacoes'}, inplace=True)
    df_kpi_use.sort_values(by='numero_interacoes',ascending=False, inplace=True)
    
    # Define custom colors based on 'nivel_interacao'
        
    fig_kpi_use = px.bar( df_kpi_use,
                     x='kpi',
                     y='numero_interacoes',
                     color='nivel_interacao',
                     color_discrete_map={
                         'Menor Interação': 'royalblue',
                         'Interacção Intermédia': '#000000',
                         'Elevada Interação': 'grey'},
                     labels={'numero_interacoes': 'Number de Interações', 'kpi': 'KPI'},
                     template='simple_white')
        
    fig_kpi_use.update_layout(width=900,height=550,
                          title={'text': '<b>Utilização de KPIs<b>',
                                 'font': {'size': 22}},
                          xaxis_title=None,
                          legend_title='Nivel de Interação')
        
    fig_kpi_use.update_traces(marker=dict(line=dict(color='#000000', width=2)))
        
    fig_kpi_use.update_yaxes(title_text='Numero de Interações',title_font=dict(size=16), tickfont=dict(size=15))
        
    fig_kpi_use.update_xaxes(tickfont=dict(size=15))
    
    return fig_kpi_use