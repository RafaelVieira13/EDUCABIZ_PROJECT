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
        
        # ROW1: COL1 (Indicador e Pie Chart), Col2 (KPI Use -Bar Chart)
        dbc.Row(
            [
              dbc.Col([
                  dbc.CardGroup([
                      dbc.Card(
                          dcc.Graph(id='indicador_numero_escolas', className='dbc'),
                          style={'maxWidth':'480px','height':indicator_height,'padding-left':'20px','padding-top': '10px','margin-bottom': '40px'}
                      ),
                      dbc.Card(
                          html.Div(),
                          color='lightsteelblue',
                          style={'maxWidth':'75px','height':indicator_height, 'margin-left':'-10px'}
                      )
                  ]),
                  dbc.Card(
                        dcc.Graph(id='pie-chart'),
                        style={'width':'100%','height':'370px','padding':'10px','padding-top': '10px'})
              ], width=4,style={'margin-right': '-60px'}),
              
              dbc.Col([
                  dbc.Card(
                      dcc.Graph(id='kpi_use'),style={'height':'100%','padding':'10px','margin-left':'80px'}
                  )
              ], width=8),
            ]
        ),
        
        # ROW2: Table
        
         dbc.Row([
            dbc.Col(
                dbc.CardGroup([
                    dbc.Card(
                        dbc.CardBody([
                            html.H2('Tabela', className='card-title',style={'font-weight':'bold','color':'#343a40'}),
                            dcc.RangeSlider(
                                id='month_slider',
                                min=df1['month'].min(),
                                max=df1['month'].max(),
                                step=1,
                                marks={i: months[i] for i in range(1, 13)},
                                value=[df1['month'].min(), df1['month'].max()],
                            ),
                            dcc.Graph(id='table', className='dbc')
                        ]),
                        style={'margin-left':'-30px', 'padding-top':'5px'}
                    )
                ]), width=12,
            )
        ], style={'margin':'20px'}),
         
         # ROW3: Line Plot - KPI USE
         dbc.Row([
            dbc.Col(
                dbc.CardGroup([
                    dbc.Card(
                        dbc.CardBody([
                            html.H2('Line Plot', className='card-title',style={'font-weight':'bold','color':'#343a40'}),
                            dcc.RangeSlider(
                                id='month_slider',
                                min=df1['month'].min(),
                                max=df1['month'].max(),
                                step=1,
                                marks={i: months[i] for i in range(1, 13)},
                                value=[df1['month'].min(), df1['month'].max()],
                            ),
                            html.Br(),
                            dcc.Dropdown(options= [{'label':kpi, 'value':kpi} for kpi in df1.iloc[:,3:13].columns],
                                     id='kpi_dropdown',style={'maxWidth': '200px','margin-left':'10px'}),
                             html.Br(),
                            dcc.Graph(id='line_kpi_use', className='dbc')
                        ]),
                        style={'margin-left':'-30px', 'padding-top':'5px'}
                    )
                ]), width=12,
            )
        ], style={'margin':'20px'}),
         
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

def month_nivel_filter(selected_nivel, selected_range):
    if selected_nivel is None and selected_range is None:
        return df1
    filtered_df = df1
    if selected_nivel:
        filtered_df = filtered_df[filtered_df['nivel_interacao'] == selected_nivel]
    if selected_range:
        filtered_df = filtered_df[(filtered_df['month'] >= selected_range[0]) & (filtered_df['month'] <= selected_range[1])]
    return filtered_df
    
# =========  Callbacks  =========== #

# Indicador

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

# KPI USE

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

# Pie Chart Numero de Interações

@callback(
    Output('pie-chart','figure'),
    [Input('nivel_int_dropdown','value')]
)

def indicador(nivel_interacao):
    nivel_mask = nivel_inter_filter(nivel_interacao)

    # Use o DataFrame original (df1) em vez do DataFrame filtrado (df_filtered)
    nivel_interacao_pie_chart = px.pie(df1,
                                   names='nivel_interacao',
                                   color='nivel_interacao',
                                   color_discrete_map={'Menor Interação': 'royalblue',
                                                       'Interacção Intermédia': 'pink',
                                                       'Elevada Interação': 'grey'},
                                   labels={'nivel_interacao': 'Nivel Interação',
                                           'value': 'Número de Escolas'}
                                   )

    nivel_interacao_pie_chart.update_traces(hoverinfo='value', textinfo='percent',
                                            textfont_size=20, marker=dict(line=dict(color='#000000', width=2)))

    nivel_interacao_pie_chart.update_layout(width=450,
                                            height=350,
                                            title={
                                                'text': '<b>Número de Escolas Por Nível de Interação<b>',
                                                'font': {'size': 20}
                                            },
                                            template='simple_white',
                                            legend=dict(
                                                title=dict(text="Nível de Interação", font=dict(size=18)),
                                                title_font=dict(size=16))
                                            )

    return nivel_interacao_pie_chart

# Tabela
@callback(
    Output('table', 'figure'),
    [Input('nivel_int_dropdown', 'value')],
    [Input('month_slider','value')],
)
def tabel(selected_nivel, selected_range):
    df_filtered = month_nivel_filter(selected_nivel, selected_range)
    
    df_table = df_filtered[['escola','nivel_interacao','slug','interacoes_totais','tutores','second_tutor','docs_fiscais (15_dias)','mensagens (7_dias)','atividades (7_dias)','relatorios_diarios (7_dias)','avaliacoes (7_dias)','menus (7_dias)','eventos (15_dias)']]
    df_table.rename(columns={'escola':'Escola',
                         'slug':'slug',
                         'nivel_interacao':'Nível de Interação',
                         'interacoes_totais':'Interações Totais',
                         'tutores':'Tutores',
                         'second_tutor':'Second Tutores',
                         'docs_fiscais (15_dias)':'Docs. Fiscais',
                         'mensagens (7_dias)':'Mensagens',
                         'atividades (7_dias)':'Atividades',
                         'relatorios_diarios (7_dias)':'Relatórios Diários',
                         'avaliacoes (7_dias)':'Avaliações',
                         'menus (7_dias)':'Menus',
                         'eventos (15_dias)':'Eventos'}, inplace=True)

    df_table.sort_values(by='Interações Totais', ascending=True, inplace=True)

    
    table = go.Figure(data=[go.Table(
        header=dict(values=list(df_table.columns),
                    fill_color='paleturquoise',
                    align='center',
                    font=dict(size=12)),
        cells=dict(values=[df_table['Escola'],df_table['Nível de Interação'],df_table['slug'],df_table['Interações Totais'],df_table['Tutores'], df_table['Second Tutores'],df_table['Docs. Fiscais'], df_table['Mensagens'], df_table['Atividades'], df_table['Relatórios Diários'], df_table['Avaliações'], df_table['Menus'],df_table['Eventos'] ],
                   fill_color='lavender',
                   align='center',
                   font=dict(size=12)))
                            ])
    
    table.update_layout(width=1400,height=650)
    
    return table

# Line Plot - Kpi by month
    
    