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
df = pd.read_csv('src/datasets/EDUCABIZ.csv')

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
                        dcc.Dropdown(options= [{'label':escola, 'value':escola} for escola in df1['escola'].unique()],
                                     id='escola_dropdown')
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
                          dcc.Graph(id='indicador_escola', className='dbc'),
                          style={'maxWidth':'1000px','height':indicator_height,'padding-left':'20px','padding-top': '10px'}
                      ),
                      dbc.Card(
                          html.Div(),
                          color='lightsteelblue',
                          style={'maxWidth':'65px','height':indicator_height, 'margin-left':'-10px'}
                      )
                  ])
              ], width=4)
            ]
        )
    ]
)


# =========  Functions To Filter the Data  =========== #
def escola_filter(escola):
    if escola is None or len(escola) == 0:
        mask = df1['escola'].isin(df1['escola'].unique())
    else:
        if not isinstance(escola, list):
            escola = [escola]  # Convert single value to a list
        mask = df1['escola'].isin(escola)
    return mask

# =========  Callbacks  =========== #
@callback(
    Output('indicador_escola', 'figure'),
    [Input('escola_dropdown', 'value')]
)
def indicador(escola):
    escola_mask = escola_filter(escola)
    df_filtered = df1.loc[escola_mask]
    
    df_filtered = df_filtered.groupby(['escola','nivel_interacao']).count().reset_index()
    if not df_filtered.empty:
        
        indicador_escola = go.Figure(go.Indicator(title={'text': f"<span>{df_filtered['escola'].iloc[0]}"}))
        indicador_escola.add_annotation(
        text=df_filtered['nivel_interacao'].iloc[0],
        x=0.5,
        y=0.8,
        showarrow=False,
        font={'size': 24})
        
        indicador_escola.update_layout(width=400,height=100)
    
    else:
        indicador_escola = go.Figure(go.Indicator(title=''))
        indicador_escola.add_annotation(
        text="Escolha uma Escola",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={'size': 16})
        
        indicador_escola.update_layout(width=400,height=100)
    
    return indicador_escola
            