# IMPORT LIBRARIES
from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import requests
from io import StringIO
import math
import dash_leaflet as dl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as plt
from scipy.stats import gaussian_kde
import plotly.express as px

# --------------------------------------------------------------------------
# DICTIONARIES AND FUNCTIONS TO STYLE THE DASHBOARD
# --------------------------------------------------------------------------

aep_cf_style ={'fontFamily': 'Arial', # --------------------------- results
       'fontSize': '35px', 
       'fontWeight': 'bold',
        'textAlign': 'center', 
        'color': '#333333'}

input_style ={'fontFamily': 'Segoe UI', # ------------------------- inputs
       'fontSize': '20px',
       'fontWeight': 'normal',
        # 'textAlign': 'center', 
        'color': "#4B4A4A"}

def aep_title_style(color):             # ------------------------- distribution's title
    title_style ={'fontFamily': 'Arial', 
       'fontSize': '30px', 
       'fontWeight': 'bold',
        'textAlign': 'center',
        'color': color}
    return title_style

aepcf_title_style ={'fontFamily': 'Arial', # --------------------------- title for AEP and Capacity factor
       'fontSize': '20px', 
       'fontWeight': 'normal',
        'textAlign': 'center', 
        'color': '#737373'}

notes_style ={'fontFamily': 'Arial', # --------------------------- notes
       'fontSize': '20px', 
       'fontWeight': 'normal',
        # 'textAlign': 'center', 
        'color': '#737373'}

# --------------------------------------------------------------------------
# DEFAULT VARIABLES
# --------------------------------------------------------------------------

cols = ['P_EML', 'P_EPFM', 'P_Rayl', 'P_Observed']
columns_PDF = ['P_EML', 'P_EPFM', 'P_Rayl']
columns_CDF = ['EML_CDF','EPFM_CDF','Rayl_CDF']
n_clases = 21
limites = [0 + i * 1 for i in range(n_clases + 1)]
h1 = 10 # [m]
h2 = 50 # [m]
rho = 1.225
Cp = 16/27 # Betz limit

# --------------------------------------------------------------------------
# FUNCTIONS FOR THE CALCULOUS
# --------------------------------------------------------------------------
def generaracion_tabla(df, Hub_Height, Diameter): # -------------------- class table with the speed intervals
    area = math.pi*(Diameter/2)**2
    df = df.copy()

    num_rows = df['WS10M'].count()

    v1_safe = df['WS10M'].replace(0, 1e-6)
    v2_safe = df['WS50M'].replace(0, 1e-6)

    df['alpha'] = [(math.log(v2_safe[x]) - math.log(v1_safe[x]))/(math.log(h2) - math.log(h1)) for x in range(num_rows)]
    df['v_hub'] = [v2_safe.iloc[x]*(Hub_Height/h2)**df['alpha'][x] for x in range(num_rows)]
    # WEIBULL DISTRIBUTION
    v_hub_mean = df['v_hub'].mean()
    v_hub_desv = df['v_hub'].std()

    def weibull(v,c,k):
        out = (k / c) * (v / c)**(k - 1) * np.exp(-(v / c)**k)
        return (out)

    df['v_hub_int'] = [int(df['v_hub'][x]) for x in range(num_rows)]

    rango = df['v_hub'].max() - df['v_hub'].min()
    n_clases = 21
    limites = [0 + i * 1 for i in range(n_clases + 1)]

    tabla_clases = pd.DataFrame({
        'Speed Range': [
            f'[{limites[i]:.2f} - {limites[i + 1]:.2f})' if i < n_clases - 1 else
            f'[{limites[i]:.2f} - {limites[i + 1]:.2f}]'  # Último intervalo incluye el valor máximo
            for i in range(n_clases)
        ],
        'Frequency': [
            ((df['v_hub'] >= limites[i]) & (df['v_hub'] < limites[i + 1])).sum() if i < n_clases - 1 else
            ((df['v_hub'] >= limites[i]) & (df['v_hub'] <= limites[i + 1])).sum()  # Incluye valores máximos
            for i in range(n_clases)
        ],
        })

    # MÉTODO 1: Empirical Method of Lysen (EML)
    k = (v_hub_desv/v_hub_mean)**(-1.086)
    c = v_hub_mean * (0.568 + (0.433/k))**(-1/k)

    tabla_clases['Weib_EML'] = [weibull((x+1),c,k) for x in range(n_clases)]
    tabla_clases['EML_CDF'] = [1 - math.exp(-((x+1)/c)**k) for x in range(n_clases)]

    # MÉTODO 2: Energy Pattern Factor Method (EPFM)
    Epf = np.mean(df['v_hub']**3) / (df['v_hub'].mean())**3
    k = 1 + 3.69/Epf**2
    c = v_hub_mean/math.gamma(1+1/k)

    tabla_clases['Weib_EPFM'] = [weibull((x+1),c,k) for x in range(n_clases)]
    tabla_clases['EPFM_CDF'] = [1 - math.exp(-((x+1)/c)**k) for x in range(n_clases)]

    # MÉTODO 3: Distribución de Rayleigh
    def rayleigh(x, v_prom):
        out = (math.pi/2) * (x / v_prom**2) * math.exp(-(math.pi/4) * (x /v_prom)**2)
        return(out)

    tabla_clases['Rayl'] = [rayleigh((x+1),v_hub_mean) for x in range(n_clases)]
    tabla_clases['Rayl_CDF'] = [1 - math.exp(-(math.pi/4)*((x+1)/v_hub_mean)**2) for x in range(n_clases)]

    # Annual Production of Energy
    df['P_wind'] = [((0.5)*rho*((df['v_hub'][x])**3)*area/1e6) for x in range(num_rows)]
    df['P_rotor'] = [(Cp*df['P_wind'][x]) for x in range(num_rows)]

    tabla_clases['P_wind'] = [((0.5)*rho*((x+1)**3)*area/1e6) for x in range(n_clases)]
    tabla_clases['P_rotor'] = [(Cp*tabla_clases['P_wind'][x]) for x in range(n_clases)]

    # Calcula la Potencia estimada para ese intervalo
    tabla_clases['P_EML'] = [(tabla_clases['P_rotor'][x]*tabla_clases['Weib_EML'][x]*24*365) for x in range(n_clases)]
    tabla_clases['P_EPFM'] = [(tabla_clases['P_rotor'][x]*tabla_clases['Weib_EPFM'][x]*24*365) for x in range(n_clases)]
    tabla_clases['P_Rayl'] = [(tabla_clases['P_rotor'][x]*tabla_clases['Rayl'][x]*24*365) for x in range(n_clases)]

    tabla_clases['P_Observed']= df.groupby(['v_hub_int'])['P_rotor'].sum()

    return tabla_clases


def crear_mapa_dash(lat1, long1, lat2, long2, lat3, long3): # ----------------------- 3 location map generator

    ubicaciones = [
    [float(lat1), float(long1)],
    [float(lat2), float(long2)],
    [float(lat3), float(long3)]
    ]

    etiquetas = ["Location A", "Location B", "Location C"]
    
    promlat = sum([ubi[0] for ubi in ubicaciones]) / len(ubicaciones)
    promlon = sum([ubi[1] for ubi in ubicaciones]) / len(ubicaciones)
    marcadores = [
        dl.Marker(
            position=[lat, lon],
            children=[
                dl.Tooltip(etiqueta)
            ]
        ) for (lat, lon), etiqueta in zip(ubicaciones, etiquetas)
    ]
    return dl.Map(
        center=[promlat, promlon],
        bounds = [[lat1, long1], [lat2, long2], [lat3, long3]],
        style={"width": "100%", "height": "600px"},
        children=[
            dl.TileLayer()
        ] + marcadores
    )

# --------------------------------------------------------------------------
# CREATING THE DASHBOARD
# --------------------------------------------------------------------------
app = Dash(
    external_stylesheets=[dbc.themes.MATERIA]
)

app.layout = html.Div([
    html.Div(
        children = [
            # --------------------------------------------------------------------------
            # TITLE 
            # --------------------------------------------------------------------------
            html.H1(children="Wind Energy Analysis Dashboard", 
                    style = {'color':'white',
                             'margin':'3',
                             'fontFamily': 'Arial, sans-serif',
                             'fontWeight': 'bold'}),
            html.P("Annual Energy Production Estimation using Weibull and Rayleigh Distributions", 
                   style = {'color':'white',
                            'margin':'2',
                            'fontFamily': 'Arial, sans-serif',
                            'fontWeight': 'normal'})],
        style={
            'backgroundColor': "#112243",
            'padding': '25px', 
            'marginBottom': '20px',
            'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.1)'
        }),
    
    dcc.Tabs (id="mis-pestanas", value="pestana-1", children= [
        dcc.Tab(label="Parameters and AEP", value="pestana-1", children=[

        html.Br(),
        html.Div([
        # --------------------------------------------------------------------------
        # TAB # 1: PARAMETERS AND RESULTS FOR AEP AND CF
        # --------------------------------------------------------------------------
            dbc.Card(
                dbc.CardBody([
                    html.Div ([
        
                        html.H4("📍 Enter the location", style = {'fontFamily':'Arial, sans-serif'}),
                        dbc.Row([
                            dbc.Col([
                                dcc.Store(id='memoria-latitude'),
                                dbc.Input(id="latitude", value="16.5700", placeholder="Type the latitude", type="text", style = input_style)
                            ], width=6), 

                            dbc.Col([
                                dcc.Store(id='memoria-longitude'),
                                dbc.Input(id="longitude", value="-94.7242", placeholder="Type the longitude", type="text", style = input_style)
                            ], width=6)
                        ]),
                        html.Br(),

                        html.H4("📆 Enter the year", style = {'fontFamily':'Arial, sans-serif'}),
                        dbc.Input(id="anio", value="2025", placeholder="Type the year", type="text", style = input_style),
                        dbc.Alert(
                            [
                                # Encabezado de la alerta
                                html.H4("Warning!", className="alert-heading"),
                                html.P(
                                    [
                                        "Remember not to use the current year or the results will be incomplete!"
                                    ],
                                    className="mb-0",
                                ),
                            ],
                            id="alert-fade",
                            color="warning",
                            dismissable=True,
                            is_open=True,
                        ),
                        html.Br(),

                        dbc.Row([
                            dbc.Col([
                                html.H4("📐 Enter the diameter", style = {'fontFamily':'Arial, sans-serif'}),
                                dbc.InputGroup([
                                    dbc.Input(id="diameter", value=169, placeholder="Type the diameter", type="number", style = input_style),
                                    dbc.InputGroupText("m", style = input_style),
                                ]),
                            ], width=6),

                            dbc.Col([
                                html.H4("📐 Enter the Hub height", style = {'fontFamily':'Arial, sans-serif'}),
                                dcc.Store(id='memoria-hub_height'),
                                dbc.InputGroup([
                                    dbc.Input(id="hub_height", value=111.5, placeholder="Type the Hub Height", type="number", style = input_style),
                                    dbc.InputGroupText("m", style = input_style),
                                ]),
                            ], width=6)
                        ]),

                        html.Br(),
                        html.H4("🪫 Enter the rated power", style = {'fontFamily':'Arial, sans-serif'}),
                        dbc.Col([
                            dcc.Store(id='memoria-rated_power'),
                            dbc.InputGroup([
                                        dbc.Input(id="rated_power", value=5.5, placeholder="Type the rated power", type="number", style = input_style),
                                        dbc.InputGroupText("MW", style = input_style),
                                    ]),
                        ], width=6),

                        html.Br(),

                        dcc.Store(id='memoria-df_nasa'), # ---------------------------- saving the dataframes for multiple uses
                        dcc.Store(id='memoria-df'),
                        dcc.Store(id='memoria-tabla_clases'),

                        dbc.Button("Submit", id="my-button", n_clicks=0, className = 'w-100'),
                    ]),
                    ]),
                    style={
                            "width":"40%",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "12px",
                            "backgroundColor": "#f8f9fa",
                        },
                        className="shadow-sm p-2"
                    ),

            html.Div ([                                                     # ------------------ creating the map of 1 location
                dl.Map(id="mapa", center=[16.5700, -94.7242], zoom=8, style={"width": "100%", "height": "90vh"},
                    children=[
                        dl.TileLayer(
                            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                            attribution="© OpenStreetMap contributors © CARTO"
                        ),
                        dl.Marker(id="marker", position=[16.5700, -94.7242], children=[dl.Tooltip("Ubicación")]
                        ),
                    ],
                )
            ], style={"width": "60%", "padding": "10px"}),

            ],
            style={"display": "flex","alignItems": "flex-start","gap": "20px"}),

            html.Br(),

            html.H3("Annual Energy Production (AEP) and Factor Capacity for one Turbine", 
                    style = {'fontFamily': 'Arial, sans-serif', 
                             'fontSize': '28px',  
                             'fontWeight': 'bold',
                             'color': "#333333"}),
            html.H3("Results using different statistical methods", 
                    style = {'fontFamily': 'Arial, sans-serif', 
                             'fontSize': '22px', 
                             'fontWeight': 'normal',     
                             'color': '#333333'}),
            
            dbc.Row([
                    dbc.Col ([
                        dbc.Card(
                            dbc.CardBody ([ # ------------------------------- Displaying the results
                                html.H3("Weibull EML", style=aep_title_style("#004aad")),
                                html.H3("-----------------------------------", style=aep_title_style("#004aad")),
                                html.H3("Annual Energy Production", style = aepcf_title_style),
                                html.Div(id ='aep_eml', style= aep_cf_style),
                                html.H3("Capacity Factor", style = aepcf_title_style),
                                html.Div(id ='cap_eml', style= aep_cf_style),
                                ]),
                                style={
                                "border": "1px solid #dee2e6",
                                "borderBottom": "6px solid #004aad",
                                "borderRadius": "12px",
                                "backgroundColor": "#f8f8fc",        
                                },
                            className="shadow-sm p-2")
                    ]),

                    dbc.Col ([
                        dbc.Card(
                            dbc.CardBody ([
                                html.H3("Weibull EPFM", style= aep_title_style("#198448")),
                                html.H3("-----------------------------------", style = aep_title_style("#198448")),
                                html.H3("Annual Energy Production", style = aepcf_title_style),
                                html.Div(id ='aep_epfm', style= aep_cf_style),
                                html.H3("Capacity Factor", style = aepcf_title_style),
                                html.Div(id ='cap_epfm', style= aep_cf_style),
                            ]),
                            style={
                                "border": "1px solid #dee2e6",
                                "borderBottom": "6px solid #198448",
                                "borderRadius": "12px",
                                "backgroundColor": "#f8f8fc",        
                                },
                            className="shadow-sm p-2")
                    ]),

                    dbc.Col ([
                        dbc.Card(
                            dbc.CardBody ([
                                html.H3("Rayleigh", style = aep_title_style("#3e1984")),
                                html.H3("-----------------------------------", style = aep_title_style("#3e1984")),
                                html.H3("Annual Energy Production", style = aepcf_title_style),
                                html.Div(id ='aep_rayl', style= aep_cf_style),
                                html.H3("Capacity Factor", style = aepcf_title_style),
                                html.Div(id ='cap_rayl', style= aep_cf_style),
                            ]),
                            style={
                                "border": "1px solid #dee2e6",
                                "borderBottom": "6px solid #3e1984",
                                "borderRadius": "12px",
                                "backgroundColor": "#f8f8fc",        
                                },
                            className="shadow-sm p-2",
                            )
                    ]),
                ]),

                html.Br(),
                html.Br(),
                dbc.Row([
                    dbc.Col ([
                        dbc.Card(
                            dbc.CardBody ([
                                html.H3("Author's notes", style = aep_title_style("#112243")),
                                html.H3("----------------------------------------------------------------------------------------------------------------------------", style = notes_style),
                                html.H3("For the calculation of the AEP, a cut-in of 3m/s and the cut-out were taken into" \
                                " account, where the maximum value of the energy generated per hour cannot exceed the rated " \
                                "power entered by the user.", style = notes_style),
                                html.H3("Data collected fron the NASA POWER (Prediction of Worldwide Energy Resources) Data Access Viewer (DAV) API " \
                                "database (https://power.larc.nasa.gov/data-access-viewer/)", style = notes_style),
                            ]),
                            style={
                                "border": "1px solid #dee2e6",
                                "borderRadius": "12px",
                                "backgroundColor": "#f8f8fc",        
                                },
                            className="shadow-sm p-2"),
                    ], width = 6),

                    dbc.Col ([
                        dbc.Card(
                            dbc.CardBody ([
                                html.H3("Methodology", style = aep_title_style("#112243")),
                                html.H3("--------------------------------------------------------", style = notes_style),
                                html.H3("EML ---> Empirical Method of Lysen", style = notes_style),
                                html.H3("EPFM ---> Energy Pattern Factor Method", style = notes_style),
                            ]),
                            style={
                                "border": "1px solid #dee2e6",
                                "borderRadius": "12px",
                                "backgroundColor": "#f8f8fc",        
                                },
                            className="shadow-sm p-2")
                    ], width = 3),

                    dbc.Col ([
                        dbc.Card(
                            dbc.CardBody ([
                                html.H3("Wind Power Density (WPD)", style = aep_title_style("#7E1919")),
                                html.H3("-----------------------------------------", style = notes_style),
                                html.Div(id ='wpd', style= aep_cf_style),
                            ]),style={
                                "border": "1px solid #dee2e6",
                                "borderBottom": "6px solid #7E1919",
                                "borderRadius": "12px",
                                "backgroundColor": "#f8f8fc",        
                                },
                            className="shadow-sm p-2",)
                    ], width = 3),
                    
                    ]),

                html.Br(),

                    dbc.Card(
                            dbc.CardBody ([
                                html.H3(" ⓘ All values are estimated for the selected site and input parameters.", style = notes_style),
                            ]),
                            style={
                                "border": "1px solid #dee2e6",
                                "borderRadius": "9px",
                                "backgroundColor": "#f8f8fc",        
                                },
                            className="shadow-sm p-2"),
                html.Br(),
        ]),

        dcc.Tab(label="Tables and graphs", value="pestana-2", children=[
            # --------------------------------------------------------------------------
            # TAB #2: TABLES AND GRAPHS FOR ONE LOCATION
            # --------------------------------------------------------------------------

        # ----------------------------------------------- Tables
        html.Div([
                html.Div ([
                    html.H3("Data collected from: NASA POWER | DAV 🌎"),# ------------ NASA dataframe
                    dash_table.DataTable(
                        id="tabla-dinamica",
                        page_size=10,
                        style_table={"overflowX": "auto"}
                    ),
                    # html.Button("Descargar CSV", id="btn_nasa"),
                    dbc.Col(
                        dbc.Button("Download CSV", id="btn_nasa", n_clicks=0, className = 'w-100', color="#112243"),
                        width = 3,
                    ),
                    dcc.Download(id="descarga_nasa_csv"),
                ], style={"width": "48%","padding": "10px"}),

                html.Div([
                    html.H3("Distributions and Power divided by intervals of speed [0m/s-21m/s]"), # ---------- class speed interval dataframe
                    dash_table.DataTable(
                        id="tabla_clases",
                        page_size=10,
                        style_table={"overflowX": "auto"}
                    ),
                    dbc.Col(
                        dbc.Button("Download CSV", id="btn_calculos", n_clicks=0, className = 'w-100', color="#112243"),
                        width = 3,
                    ),
                    dcc.Download(id="descarga_calculos_csv"),
                ], style={
                    "width": "48%",
                    "padding": "10px"
                }),

            ], style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "flex-start"
            }),
                dbc.Row ([ # ----------------------------------------- Graph: Comparison between distributions AEP
                    dbc.Col ([
                        html.Div([
                            html.H3("Comparison between distributions"),
                            dbc.Checklist(id="checklist1", options=cols, value=["P_EPFM", "P_Observed"], 
                                          inline=True,
                                          switch = True,
                                        style = {"fontSize":"1rem"},
                                        input_style={"cursor":"pointer"},
                                        label_style = {"marginRight":"15px", "cursor":"pointer"}),
                            dcc.Graph(id="graph1"),
                            ]),
                    ], width = 6),

                dbc.Col([ # ------------------------------------------ Graph: Power Curve
                        html.Div([
                            html.Div([
                                html.H3("Power Curve"),
                                dcc.Graph(id="graph5")
                            ]),
                        ]),
                    ], width = 6),
                ], className="g-2"),
                html.Br(),

                dbc.Row([ # --------------------------------------------- Graph: Comparison between distributions PDF & CDF
                    dbc.Col([
                    html.Div([
                        html.H3("Comparation between distributions PDF & CDF"),
                        dbc.Checklist(id="checklist2", options=columns_PDF, value=["P_EML", "P_Rayl"], 
                                      inline=True, 
                                      switch = True,
                                      style = {"fontSize":"1rem"},
                                      input_style={"cursor":"pointer"},
                                      label_style = {"marginRight":"15px", "cursor":"pointer"}),
                        dcc.Graph(id="graph2")
                    ]),
                ], width = 6),

                dbc.Col([ # -------------------------------------------- Graph: speed at different heights
                    html.Div([
                        html.H3("Speed at different heights"),
                        dcc.Graph(id="graph3")
                    ]),
                ], width = 6),
            ]),

            html.Div([ # -------------------------------------------------- Graph: Wind Rose
                html.H3("Wind Rose at 50m height"),
                dcc.Graph(id="graph4")
            ]),

            dbc.Card(
                dbc.CardBody ([
                    html.H3(" ⓘ All values are estimated for the selected site and input parameters.", style = notes_style),
                    ]),
                            style={
                                "border": "1px solid #dee2e6",
                                "borderRadius": "9px",
                                "backgroundColor": "#f8f8fc",        
                                },
                            className="shadow-sm p-2"),
                html.Br(),
        ]),

        # --------------------------------------------------------------------------
        # TAB #3: 3 LOCATION COMPARISON
        # --------------------------------------------------------------------------

        dcc.Tab(label="3 locations comparison", value="pestana-3", children=[
            html.Div([ # --------------------------------- Input data
            # ------------------------------------
            # LOCATION 1
            # ------------------------------------
            dbc.Row ([
                dbc.Card(
                dbc.CardBody([
                    html.Div ([
                        html.H3("Location A", style = aep_title_style("#112243")),
        
                        html.H4("📍 Enter the coordinates", style = {'fontFamily':'Arial, sans-serif'}),
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(id="latitude1", value="16.5474", placeholder="Type the latitude", type="text", style = input_style),
                            ], width=12), # width=6 significa que ocupará la mitad del espacio disponible

                            dbc.Col([
                                dbc.Input(id="longitude1", value="-94.8295", placeholder="Type the longitude", type="text", style = input_style),
                            ], width=12)
                        ]),
                        html.Br(),

                            dbc.Col([
                                html.H4("📐 Enter the diameter", style = {'fontFamily':'Arial, sans-serif'}),
                                dbc.InputGroup([
                                    dbc.Input(id="diameter1", value=169, placeholder="Type the diameter", type="number", style = input_style),
                                    dbc.InputGroupText("m", style = input_style),
                                ]),
                            ], width=12),

                            html.Br(),

                            dbc.Col([
                                html.H4("📐 Enter the Hub height", style = {'fontFamily':'Arial, sans-serif'}),
                                dbc.InputGroup([
                                    dbc.Input(id="hub_height1", value=111.5, placeholder="Type the Hub Height", type="number", style = input_style),
                                    dbc.InputGroupText("m", style = input_style),
                                ]),
                            ], width=12),

                        html.Br(),
                        html.H4("🪫 Enter the rated power", style = {'fontFamily':'Arial, sans-serif'}),
                        dbc.Col([
                            dbc.InputGroup([
                                        dbc.Input(id="rated_power1", value=5.5, placeholder="Type the rated power", type="number", style = input_style),
                                        dbc.InputGroupText("MW", style = input_style),
                                    ]),
                        ], width=12),
                    ]),
                    ]),
                    style={
                            "width":"20%",
                            "padding": "10px",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "12px",     
                            "backgroundColor": "#f8f9fa",       
                        },
                        className="shadow-sm p-2",
                    ),

                # ------------------------------------
                # LOCATION 2
                # ------------------------------------

                dbc.Card(
                dbc.CardBody([
                    html.Div ([
                        html.H3("Location B", style = aep_title_style("#112243")),
        
                        html.H4("📍 Enter the coordinates", style = {'fontFamily':'Arial, sans-serif'}),
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(id="latitude2", value="31.2973", placeholder="Type the latitude", type="text", style = input_style),
                            ], width=12), # width=6 significa que ocupará la mitad del espacio disponible

                            dbc.Col([
                                dbc.Input(id="longitude2", value="-98.4968", placeholder="Type the longitude", type="text", style = input_style),
                            ], width=12)
                        ]),
                        html.Br(),

                            dbc.Col([
                                html.H4("📐 Enter the diameter", style = {'fontFamily':'Arial, sans-serif'}),
                                dbc.InputGroup([
                                    dbc.Input(id="diameter2", value=169, placeholder="Type the diameter", type="number", style = input_style),
                                    dbc.InputGroupText("m", style = input_style),
                                ]),
                            ], width=12),

                            html.Br(),

                            dbc.Col([
                                html.H4("📐 Enter the Hub height", style = {'fontFamily':'Arial, sans-serif'}),
                                dbc.InputGroup([
                                    dbc.Input(id="hub_height2", value=111.5, placeholder="Type the Hub Height", type="number", style = input_style),
                                    dbc.InputGroupText("m", style = input_style),
                                ]),
                            ], width=12),

                        html.Br(),
                        html.H4("🪫 Enter the rated power", style = {'fontFamily':'Arial, sans-serif'}),
                        dbc.Col([
                            dbc.InputGroup([
                                        dbc.Input(id="rated_power2", value=5.5, placeholder="Type the rated power", type="number", style = input_style),
                                        dbc.InputGroupText("MW", style = input_style),
                                    ]),
                        ], width=12),
                    ]),
                    ]),
                    style={
                            "width":"20%",
                            "padding": "10px",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "12px",     
                            "backgroundColor": "#f8f9fa",       
                        },
                        className="shadow-sm p-2"
                    ),

                    # ------------------------------------
                    # LOCATION 3
                    # ------------------------------------

                dbc.Card(
                dbc.CardBody([
                    html.Div ([
                        html.H3("Location C", style = aep_title_style("#112243")),
        
                        html.H4("📍 Enter the coordinates", style = {'fontFamily':'Arial, sans-serif'}),
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(id="latitude3", value="32.6208", placeholder="Type the latitude", type="text", style = input_style),
                            ], width=12),

                            dbc.Col([
                                dbc.Input(id="longitude3", value="-116.0371", placeholder="Type the longitude", type="text", style = input_style),
                            ], width=12)
                        ]),
                        html.Br(),

                            dbc.Col([
                                html.H4("📐 Enter the diameter", style = {'fontFamily':'Arial, sans-serif'}),
                                dbc.InputGroup([
                                    dbc.Input(id="diameter3", value=169, placeholder="Type the diameter", type="number", style = input_style),
                                    dbc.InputGroupText("m", style = input_style),
                                ]),
                            ], width=12),

                            html.Br(),

                            dbc.Col([
                                html.H4("📐 Enter the Hub height", style = {'fontFamily':'Arial, sans-serif'}),
                                dbc.InputGroup([
                                    dbc.Input(id="hub_height3", value=111.5, placeholder="Type the Hub Height", type="number", style = input_style),
                                    dbc.InputGroupText("m", style = input_style),
                                ]),
                            ], width=12),

                        html.Br(),
                        html.H4("🪫 Enter the rated power", style = {'fontFamily':'Arial, sans-serif'}),
                        dbc.Col([
                            dbc.InputGroup([
                                        dbc.Input(id="rated_power3", value=5.5, placeholder="Type the rated power", type="number", style = input_style),
                                        dbc.InputGroupText("MW", style = input_style),
                                    ]),
                        ], width=12),
                    ]),
                    ]),
                    style={
                            "width":"20%",
                            "padding": "10px",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "12px",     
                            "backgroundColor": "#f8f9fa",       
                        },
                        className="shadow-sm p-2"
                    ),

                    html.Div ([
                        html.H2("📍 Selected locations", style={"fontFamily": "Arial"}),
                        html.Div(id="contenedor-mapa")
                    ], style={
                        "width": "40%",
                        "padding": "10px"
                    }),



                    ], className="g-4"),
                        

            dbc.Row ([
                dbc.Col ([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("📆 Enter the year to analize", style = {'fontFamily':'Arial, sans-serif'}),
                            dbc.Input(id="anio_3loc", value="2025", placeholder="Type the year", type="text", style = input_style),
                            html.Br(),

                            dbc.Alert(
                            [
                                # Encabezado de la alerta
                                html.H4("Warning!", className="alert-heading"),
                                html.P(
                                    [
                                        "Remember not to use the current year or the results will be incomplete!"
                                    ],
                                    className="mb-0",
                                ),
                            ],
                            id="alert-fade2",
                            color="warning",
                            dismissable=True,
                            is_open=True,
                        ),

                            html.H4("📆 Enter the project lifespan", style = {'fontFamily':'Arial, sans-serif'}),
                            dbc.Input(id="lifespan_3loc", value="25", placeholder="Type the project lifespan", type="number", style = input_style),
                            html.Br(),

                            html.H4("💲Enter the discount rate", style = {'fontFamily':'Arial, sans-serif'}),
                            dbc.Input(id="discount_3loc", value="0.03", placeholder="Type the discount rate", type="number", style = input_style),
                            html.Br(),

                            dbc.Button("Submit", id="my-button_3loc", n_clicks=0),
                            ]),
                            style={
                                "padding": "10px",
                                "border": "1px solid #dee2e6",
                                "borderRadius": "12px",     
                                "backgroundColor": "#f8f9fa",       
                            },
                            className="shadow-sm p-2",
                        )], width=4),

                dbc.Col ([ # ------------------------------------------------------ Comparative dataframe
                    html.Div([
                    html.H3("Comparation between 3 locations"),
                    dash_table.DataTable(id="tabla_3loc", page_size=10, style_table={"overflowX": "auto"}),
                        ]),

                html.Br(),
                html.Br(),
                    
                dbc.Col([ # ------------------------------------------------------- Result of the best option
                    dbc.Card(
                        dbc.CardBody ([
                            html.H3("The best option", style=aep_title_style("#e3c53c")),
                            html.H3("------------------------", style=aep_title_style("#e3c53c")),
                            html.H3("Best Option", style = aepcf_title_style),
                            html.Div(id ='best_location', style= aep_cf_style),
                            html.H3("AEP", style = aepcf_title_style),
                            html.Div(id ='best_aep', style= aep_cf_style),
                            html.H3("LCOE", style = aepcf_title_style),
                            html.Div(id ='best_lcoe', style= aep_cf_style),
                            ]),
                            style={
                            "border": "1px solid #dee2e6",
                            "borderBottom": "6px solid #e3c53c",
                            "borderRadius": "12px",
                            "backgroundColor": "#f8f8fc",        
                            },
                        className="shadow-sm p-2"),
                        ], width=5),                    

                    ], width=8),
                ], className="g-4 mb-4"),
            ]),

            dbc.Card(
                dbc.CardBody ([
                    html.H3("The Annual Energy Production (AEP) in 3 locations was calculated through the Energy Pattern Factor Method (EPFM)" \
                    " which literature identifies it as the most reliable and precise approach.", style = notes_style),
                    html.H3("For the Levelized Cost of Energy (LCOE):", style = notes_style),
                    html.H3("capex per MW = 2.1 * 10 **6", style = notes_style),
                    html.H3("annual cost per MW = 15 * 10 **3", style = notes_style),
                    ]),
                    style={
                        "border": "1px solid #dee2e6",
                        "borderRadius": "9px",
                        "backgroundColor": "#f8f8fc",        
                        },
                        className="shadow-sm p-2"),
            html.Br(),

            dbc.Card(
                            dbc.CardBody ([
                                html.H3(" ⓘ All values are estimated for the selected site and input parameters.", style = notes_style),
                            ]),
                            style={
                                "border": "1px solid #dee2e6",
                                "borderRadius": "9px",
                                "backgroundColor": "#f8f8fc",        
                                },
                            className="shadow-sm p-2"),
            html.Br(),
        ]),
    ]),
]),

# --------------------------------------------------------------------------
# CALLBACKS AND THEIR FUNCTIONS
# --------------------------------------------------------------------------

@callback( # ---------------------------------------------- 1 location calculous
    Output('tabla-dinamica', 'data'),
    Output('tabla-dinamica', 'columns'),
    Output('mapa', 'center'),
    Output('marker', 'position'),
    Output('tabla_clases', 'data'),
    Output('tabla_clases', 'columns'),
    Output('memoria-latitude', 'data'),
    Output('memoria-longitude', 'data'),
    Output('memoria-hub_height', 'data'),
    Output('memoria-rated_power', 'data'),
    Output('memoria-df_nasa', 'data'),
    Output('memoria-tabla_clases', 'data'),
    Output('memoria-df', 'data'),
    Input('my-button', 'n_clicks'),
    State('latitude', 'value'),
    State('longitude', 'value'),
    State('anio', 'value'),
    State('diameter', 'value'),
    State('hub_height', 'value'),
    State('rated_power', 'value'),
)
def update_output(n_clicks, latitude, longitude, anio, diameter, hub_height, rated_power):
    if n_clicks is None:
        return [], [], [16.5700, -94.7242], [16.5700, -94.7242], [], [], [], [], [], [], [], [], []

    try:
        lat = float(latitude)
        lon = float(longitude)
        year = int(anio)
    except (TypeError, ValueError):
        return [], [], [16.5700, -94.7242], [16.5700, -94.7242], [], [], [], [], [], [], [], [], []

    nueva_posicion = [lat, lon]

    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=WS10M,WD10M,WS50M,WD50M&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={year}0101&end={year}1231&format=csv&header=false"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return [], [], nueva_posicion, nueva_posicion, [], [], [], [], [], [], [], [], []

    try:
        df_nasa_local = pd.read_csv(StringIO(response.text), comment='#')
    except Exception:
        return [], [], nueva_posicion, nueva_posicion, [], [], [], [], [], [], [], [], []

    if 'WS10M' not in df_nasa_local.columns or 'WS50M' not in df_nasa_local.columns:
        return [], [], nueva_posicion, nueva_posicion, [], [], [], [], [], [], [], [], []

    df_calculos = df_nasa_local.copy()


    global df_nasa, tabla_clases # Para descargar los dataframes
    df_nasa = df_nasa_local
    tabla_clases = generaracion_tabla(df_calculos, hub_height, diameter)

    # Annual Production of Energy
    area = math.pi*(diameter/2)**2
    num_rows = df_calculos['WS10M'].count()

    v1_safe = df_calculos['WS10M'].replace(0, 1e-6)
    v2_safe = df_calculos['WS50M'].replace(0, 1e-6)

    df_calculos['alpha'] = [(math.log(v2_safe[x]) - math.log(v1_safe[x]))/(math.log(h2) - math.log(h1)) for x in range(num_rows)]
    df_calculos['v_hub'] = [v2_safe.iloc[x]*(hub_height/h2)**df_calculos['alpha'][x] for x in range(num_rows)]
    df_calculos['P_wind'] = [((0.5)*rho*((df_calculos['v_hub'][x])**3)*area/1e6) for x in range(num_rows)]
    df_calculos['P_rotor'] = [(Cp*df_calculos['P_wind'][x]) for x in range(num_rows)]

    return (
        df_nasa_local.to_dict('records'),
        [{"name": col, "id": col} for col in df_nasa_local.columns],
        nueva_posicion,
        nueva_posicion,
        tabla_clases.to_dict('records'),
        [{"name": col, "id": col} for col in tabla_clases.columns],
        latitude, longitude, hub_height, rated_power, df_nasa.to_dict('records'), tabla_clases.to_dict('records'), df_calculos.to_dict('records')
    )

# --------------------------------------------------------------------------------- Calculating AEP, Cf and WPD
@callback( # ---------------------------------------------------------------------- Print the results
        Output('aep_eml', 'children'),
        Output('aep_epfm', 'children'),
        Output('aep_rayl', 'children'),
        Output('cap_eml', 'children'),
        Output('cap_epfm', 'children'),
        Output('cap_rayl', 'children'),
        Output('wpd', 'children'),
        Input('memoria-rated_power', 'data'),
        Input('memoria-tabla_clases', 'data'),
        Input('memoria-df', 'data')
    )

def annual_energy_production(Rated_power, tabla_clases, df):
    tabla_clases = pd.DataFrame(tabla_clases)
    df = pd.DataFrame(df)

    # ---------------------------------------------------
    # ANNUAL ENERGY PRODUCTION (AEP)
    # ---------------------------------------------------
    df_comparativo = pd.DataFrame({'Método':['Datos crudos', 'EML', 'EPFM', 'Rayleigh']})
    # ----------- Datos crudos
    energia_real_crudos = list()

    for idx, velocidad in enumerate(df['v_hub']):
        if velocidad > 3:
            energia_real_crudos.append(df.loc[idx, 'P_rotor'])

    for x in range(len(energia_real_crudos)):
        if energia_real_crudos[x]>Rated_power:
            energia_real_crudos[x] = Rated_power

    df_comparativo.loc[0, 'AEP [MW]'] = sum(energia_real_crudos)
    df_comparativo.loc[0, 'Cf [%]'] = sum(energia_real_crudos)/(Rated_power*8760)*100

    # ----------- EML
    tabla_clases['tope_EML'] = [tabla_clases['Weib_EML'][x]*(Rated_power*365*24) for x in range(n_clases)]
    energia_real_EML = list()

    for x in range(3, n_clases, 1):
        current_p_eml = tabla_clases['P_EML'][x]
        current_tope_eml = tabla_clases['tope_EML'][x]
        if current_p_eml > current_tope_eml:
            energia_real_EML.append(current_tope_eml)
        else:
            energia_real_EML.append(current_p_eml)

    df_comparativo.loc[1, 'AEP [MW]'] = sum(energia_real_EML)
    df_comparativo.loc[1, 'Cf [%]'] = sum(energia_real_EML)/(Rated_power*8760)*100

    # ----------- EPFM
    tabla_clases['tope_EPFM'] = [tabla_clases['Weib_EPFM'][x]*(Rated_power*365*24) for x in range(n_clases)]
    energia_real_EPFM = list()

    for x in range(3, n_clases, 1):
        current_p_EPFM = tabla_clases['P_EPFM'][x]
        current_tope_EPFM = tabla_clases['tope_EPFM'][x]
        if current_p_EPFM > current_tope_EPFM:
            energia_real_EPFM.append(current_tope_EPFM)
        else:
            energia_real_EPFM.append(current_p_EPFM)

    df_comparativo.loc[2, 'AEP [MW]'] = sum(energia_real_EPFM)
    df_comparativo.loc[2, 'Cf [%]'] = sum(energia_real_EPFM)/(Rated_power*8760)*100

    # ----------- Rayleigh
    tabla_clases['tope_Rayl'] = [tabla_clases['Rayl'][x]*(Rated_power*365*24) for x in range(n_clases)]
    energia_real_Rayl = list()

    for x in range(3, n_clases, 1):
        current_p_Rayl = tabla_clases['P_Rayl'][x]
        current_tope_Rayl = tabla_clases['tope_Rayl'][x]
        if current_p_Rayl > current_tope_Rayl:
            energia_real_Rayl.append(current_tope_Rayl)
        else:
            energia_real_Rayl.append(current_p_Rayl)

    aep_eml = f"{sum(energia_real_EML)/1e3:.3f} GWh"
    aep_epfm = f"{sum(energia_real_EPFM)/1e3:.3f} GWh"
    aep_rayl = f"{sum(energia_real_Rayl)/1e3:.3f} GWh"

    # ENERGY CAPACITY
    cap_eml = f"{(sum(energia_real_EML)/(Rated_power*8760)*100):.3f}%"
    cap_epfm = f"{(sum(energia_real_EPFM)/(Rated_power*8760)*100):.2f}%"
    cap_rayl = f"{(sum(energia_real_Rayl)/(Rated_power*8760)*100):.2f}%"

    # ENERGY CAPACITY CON DATOS CRUDOS
    num_rows = df['WS10M'].count()

    potencia_turbine = list(df['P_rotor'])
    for x in range(num_rows):
        if potencia_turbine[x]>Rated_power:
            potencia_turbine[x] = Rated_power

    wpd_list = [((0.5)*rho*((df['v_hub'][x])**3)) for x in range(num_rows)]
    wpd = np.mean(wpd_list)
    wpd = f"{wpd:.3f} W/m^2"

    return aep_eml, aep_epfm, aep_rayl, cap_eml, cap_epfm, cap_rayl, wpd

# --------------------------------------------------------------------------
# DOWNLOAD NASA DATAFRAME
# --------------------------------------------------------------------------
@callback(
    Output('descarga_nasa_csv', 'data'),
    Input('btn_nasa', 'n_clicks'),
    prevent_initial_call=True,
)
def generar_csv_nasa(n_clicks):
    return dcc.send_data_frame(df_nasa.to_csv, "DATA_NASA.csv", index=False)

# --------------------------------------------------------------------------
# DOWNLOAD CALCULOUS DATAFRAME
# --------------------------------------------------------------------------
@callback(
    Output('descarga_calculos_csv', 'data'),
    Input('btn_calculos', 'n_clicks'),
    prevent_initial_call=True,
)
def generar_csv_calculos(n_clicks):
    return dcc.send_data_frame(tabla_clases.to_csv, "DATA_CALCULOUS.csv", index=False)

# --------------------------------------------------------------------------
# GRAPH: COMPARISON BETWEEN DISTRIBUTIONS AEP
# --------------------------------------------------------------------------
@callback(
    Output("graph1", "figure"),
    Input("checklist1", "value"),
    Input("memoria-tabla_clases", "data")
)

def update_graph(selected_cols, tabla_clases):
    if not selected_cols or not tabla_clases:
        raise PreventUpdate

    tabla_clases = pd.DataFrame(tabla_clases)
    colors = ['blue', 'orange', 'green', 'red']

    labels = {
        'P_EML':'Weibull EML',
        'P_EPFM': 'Weibull EPFM',
        'P_Rayl':'Rayleigh',
        'P_Observed': 'Power Observed'
    }

    fig = go.Figure()

    x_values = np.arange(1, n_clases + 1)

    for i, col in enumerate(selected_cols):
        fig.add_trace(
            go.Scatter(x=x_values, y=tabla_clases[col], mode='lines', name=labels[col], line=dict(color=colors[i], width=2),
            fill='tozeroy',fillcolor=f'rgba{(*__import__("matplotlib").colors.to_rgb(colors[i]), 0.3)}'
            )
        )

    fig.update_layout(
        title='AEP comparison between distributions',
        xaxis_title='Speed [m/s]', yaxis_title='Power [MW]',
        template='plotly_white',
        width=1000, height=600,
        legend=dict(
            x=1.02,
            y=1,
            bordercolor='black',
            borderwidth=1
        )
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=np.arange(0, 22, 1)
    )

    return fig

# --------------------------------------------------------------------------
# GRAPH: COMPARISON BETWEEN DISTRIBUTIONS PDF AND CDF
# --------------------------------------------------------------------------
@callback(
    Output("graph2", "figure"),
    Input("checklist2", "value"),
    Input("memoria-tabla_clases", "data")
)

def update_graph(selected_PDF, tabla_clases):
    if not selected_PDF or not tabla_clases:
        raise PreventUpdate
    
    tabla_clases = pd.DataFrame(tabla_clases)

    columns1 = selected_PDF
    columns2 = list()
    columns_CDF = {
        'P_EML':'EML_CDF',
        'P_EPFM':'EPFM_CDF',
        'P_Rayl':'Rayl_CDF'}
    for col in columns1:
        columns2.append(columns_CDF[col])

    label1 = ['EML PDF', 'EPFM PDF', 'Rayleigh PDF']
    label2 = ['EML CDF', 'EPFM CDF', 'Rayleigh CDF']
    colors = ['blue', 'orange', 'green']

    x_values = list(range(1, n_clases + 1))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i in range(len(columns1)):
        fig.add_trace(
            go.Scatter(x=x_values, y=tabla_clases[columns1[i]],
                mode='lines', name=columns1[i], line=dict(color=colors[i], width=3)
            ), secondary_y=False
        )

    for i in range(len(columns2)):
        fig.add_trace(
            go.Scatter(x=x_values, y=tabla_clases[columns2[i]],
                mode='lines+markers', name=columns2[i], line=dict(color=colors[i], dash='dash'), marker=dict(size=6)
            ), secondary_y=True
        )

    fig.update_layout(
        title='Comparison between distributions PDF & CDF',
        width=1000,
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1)
    )
    fig.update_xaxes(
        title_text='Speed (m/s)'
    )
    fig.update_yaxes(
        title_text='Probability Density Function (PDF)', secondary_y=False)
    fig.update_yaxes(
        title_text='Cumulative Distribution Function (CDF)', range=[0, 1.05], secondary_y=True)

    return fig

# --------------------------------------------------------------------------
# GRAPH: WIND SPEEDS AT DIFFERENT HEIGHTS
# --------------------------------------------------------------------------
@callback(
    Output("graph3", "figure"),
    Input("memoria-df_nasa", "data"),
    Input("memoria-hub_height", "data")
)

def update_graph(df, Hub_Height):
    df = pd.DataFrame(df)

    v1 = df['WS10M'].replace(0, 1e-6)
    v2 = df['WS50M'].replace(0, 1e-6)

    num_rows = df['WS10M'].count()
    df['alpha'] = [(math.log(v2[x]) - math.log(v1[x]))/(math.log(h2) - math.log(h1)) for x in range(num_rows)]
    df['v_hub'] = [v2.iloc[x]*(Hub_Height/h2)**df['alpha'][x] for x in range(num_rows)]

    fig = go.Figure()

    x = np.linspace(
        min(df[['v_hub', 'WS10M', 'WS50M']].min()),
        max(df[['v_hub', 'WS10M', 'WS50M']].max()),
        500
    )

    series = [
        (df['v_hub'].dropna(), 'blue', f'{Hub_Height}m'),
        (df['WS10M'].dropna(), 'red', '10m'),
        (df['WS50M'].dropna(), 'green', '50m')
    ]

    for data, color, label in series:

        kde = gaussian_kde(data)
        y = kde(x)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                fill='tozeroy'
            )
        )

    fig.update_layout(
        title='Wind speed distribution at different heights',
        xaxis_title='Speed (m/s)',
        yaxis_title='Density',
        width=1000,
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=np.arange(0, 21, 1)
    )
    return fig

# --------------------------------------------------------------------------
# GRAPH: WIND ROSE AT 50M HEIGHT
# --------------------------------------------------------------------------
@callback(
    Output("graph4", "figure"),
    Input("memoria-df_nasa", "data")
)

def display_graph(df):
    df = pd.DataFrame(df)
    df_windrose50m = pd.DataFrame()
    df_windrose50m['direction'] = df['WD50M']
    df_windrose50m['frequency'] = df['WS50M']

    df_windrose50m = px.data.wind()
    fig = px.bar_polar(
        df_windrose50m,
        r="frequency",
        theta="direction",
        color="strength",
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Plasma_r
    )

    fig.update_layout(title='Wind Rose at 50m height')

    return fig

# --------------------------------------------------------------------------
# GRAPH: POWER CURVE
# --------------------------------------------------------------------------
@callback(
    Output("graph5", "figure"),
    Input("memoria-tabla_clases", "data"),
    Input("memoria-rated_power", "data")
)

def generacion_grafica(tabla_clases, Rated_power):
    tabla_clases = pd.DataFrame(tabla_clases)

    x_values = [(n+1) for n in range(n_clases)]
    energia_rotor = list(tabla_clases['P_rotor'])
    energia_wind = list(tabla_clases['P_wind'])
    energias = [energia_rotor, energia_wind]

    for energia in energias:
        for x in range(n_clases):
            if energia[x] > Rated_power: 
                energia[x] = Rated_power

    cont = int()
    for valor in energia_rotor:
        cont += 1
        if valor == Rated_power:
            break

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=x_values, y=energia_rotor, 
            mode='lines', name='Rotor', line=dict(width=3))
    )

    fig.add_trace(
        go.Scatter(x=x_values, y=energia_wind, mode='lines', name='Wind',line=dict(width=3))
    )

    fig.add_vline(x=3, line_width=2, line_dash="dot", line_color="red")
    fig.add_vline(x=cont, line_width=2, line_dash="dot", line_color="red")

    fig.update_layout(
        title='Power Curve',
        xaxis_title='Speed [m/s]',
        yaxis_title='Power [MW]',
        template='plotly_white',
        width=1000, 
        height=600,
        legend=dict(
            x=0.02,
            y=0.98,
            bordercolor='black',
            borderwidth=1
        )
    )

    return fig

# --------------------------------------------------------------------------
# CALCULOUS FOR 3 LOCATIONS
# --------------------------------------------------------------------------
@callback(
    Output('tabla_3loc', 'data'),
    Output('tabla_3loc', 'columns'),
    Output('contenedor-mapa', 'children'),
    Output('best_location', 'children'),
    Output('best_aep', 'children'),
    Output('best_lcoe', 'children'),
    Input('my-button_3loc', 'n_clicks'),
    # LOCATION 1
    State('latitude1', 'value'),
    State('longitude1', 'value'),
    State('diameter1', 'value'),
    State('hub_height1', 'value'),
    State('rated_power1', 'value'),
    # LOCATION 2
    State('latitude2', 'value'),
    State('longitude2', 'value'),
    State('diameter2', 'value'),
    State('hub_height2', 'value'),
    State('rated_power2', 'value'),
    # LOCATION 3
    State('latitude3', 'value'),
    State('longitude3', 'value'),
    State('diameter3', 'value'),
    State('hub_height3', 'value'),
    State('rated_power3', 'value'),

    State('anio_3loc', 'value'),
    State('lifespan_3loc', 'value'),
    State('discount_3loc', 'value'),
)

def calculos_3locations(n_clicks,
    latitude1, longitude1, Diameter1, Hub_Height1, Rated_power1,
    latitude2, longitude2, Diameter2, Hub_Height2, Rated_power2,
    latitude3, longitude3, Diameter3, Hub_Height3, Rated_power3, 
    anio, project_lifespan, discount_rate):
# ----------------------------------------------------------------------------
# NASA DATABASE
# ----------------------------------------------------------------------------

    if n_clicks is None:
        return [], []


    def get_data(lon, lat, year):
        url = (
            "https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=WS10M,WD10M,WS50M,WD50M&community=RE"
            f"&longitude={lon}&latitude={lat}"
            f"&start={year}0101&end={year}1231&format=csv&header=false"
        )
        return url

    try:
        response = requests.get(get_data(longitude1, latitude1, anio), timeout=20)
        response.raise_for_status()
        df_loc1 = pd.read_csv(StringIO(response.text), comment='#')

        response = requests.get(get_data(longitude2, latitude2, anio), timeout=20)
        response.raise_for_status()
        df_loc2 = pd.read_csv(StringIO(response.text), comment='#')

        response = requests.get(get_data(longitude3, latitude3, anio), timeout=20)
        response.raise_for_status()
        df_loc3 = pd.read_csv(StringIO(response.text), comment='#')

    except Exception:
        return [], []

    if 'WS10M' not in df_loc1.columns or 'WS50M' not in df_loc1.columns:
        return [], []

    # ----------------------------------------------------------------------------
    # PREPARING FOR THE WAR
    # ----------------------------------------------------------------------------
    lista_df_3loc = [df_loc1, df_loc2, df_loc3]
    list_df_names = ['df_loc1', 'df_loc2', 'df_loc3']
    num_rows = df_loc1['WS10M'].count()

    alturas = {
        'df_loc1': Hub_Height1,
        'df_loc2': Hub_Height2,
        'df_loc3': Hub_Height3,
    }

    diameters = [Diameter1, Diameter2, Diameter3]

    df_3loc = pd.DataFrame()
    df_3loc['Location'] = ['Location A', 'Location B', 'Location C']
    df_3loc['Hub_Height [m]'] =[Hub_Height1, Hub_Height2, Hub_Height3]
    df_3loc['Diameter [m]'] = [Diameter1, Diameter2, Diameter3]
    df_3loc['Rated_power [MW]'] = [Rated_power1, Rated_power2, Rated_power3]

    # ----------------------------------------------------------------------------
    # WIND SPEED EXTRAPOLATION HELLMANN
    # ----------------------------------------------------------------------------
    for idx, df_current in enumerate(lista_df_3loc):
        df_name_key = list_df_names[idx]

        v1_series_raw = df_current['WS10M']
        v2_series_raw = df_current['WS50M']

        v1_series_safe = v1_series_raw.replace(0, 1e-6)
        v2_series_safe = v2_series_raw.replace(0, 1e-6)

        df_current['alpha'] = [(math.log(v2_series_safe.iloc[x]) - math.log(v1_series_safe.iloc[x])) / (math.log(h2) - math.log(h1)) for x in range(num_rows)]
        df_current['v_hub'] = [v2_series_safe.iloc[x] * (alturas[df_name_key] / h2)**df_current['alpha'][x] for x in range(num_rows)]

        df_3loc.loc[idx, 'v_hub_mean'] = df_current['v_hub'].mean()
        df_3loc.loc[idx, 'v_hub_desv'] = df_current['v_hub'].std()

    # ----------------------------------------------------------------------------
    # POWER CALCULOUS
    # ----------------------------------------------------------------------------
        df_current['P_rotor_[MW]'] = [(Cp*(0.5)*rho*((df_current['v_hub'][x])**3)*math.pi * (diameters[idx] / 2)**2/1e6) for x in range(num_rows)]

    # ----------------------------------------------------------------------------
    # CLASS TABLE CREATION 
    # ----------------------------------------------------------------------------
    n_clases = 21
    limites = [0 + i * 1 for i in range(n_clases + 1)]

    tabla_clases_3loc = pd.DataFrame({
        'Intervalo_de_velocidades': [
            f'[{limites[i]:.2f} - {limites[i + 1]:.2f})' if i < n_clases - 1 else
            f'[{limites[i]:.2f} - {limites[i + 1]:.2f}]'  # Último intervalo incluye el valor máximo
            for i in range(n_clases)
        ]
    })

    # ----------------------------------------------------------------------------
    # WEIBULL
    # ----------------------------------------------------------------------------
    def weibull(v,c,k):
        out = (k / c) * (v / c)**(k - 1) * np.exp(-(v / c)**k)
        return (out)

    for idx, df_current in enumerate(lista_df_3loc):
        df_3loc.loc[idx, 'Epf'] = np.mean(df_current['v_hub']**3) / (df_current['v_hub'].mean())**3
        df_3loc.loc[idx, 'k'] = 1 + 3.69/df_3loc.loc[idx, 'Epf']**2
        df_3loc.loc[idx, 'c'] = df_3loc.loc[idx, 'v_hub_mean']/math.gamma(1+1/df_3loc.loc[idx, 'k'])

    tabla_clases_3loc['EPFM_loc1'] = [weibull((x+1),df_3loc.loc[0, 'c'],df_3loc.loc[0, 'k']) for x in range(n_clases)]
    tabla_clases_3loc['EPFM_loc2'] = [weibull((x+1),df_3loc.loc[1, 'c'],df_3loc.loc[1, 'k']) for x in range(n_clases)]
    tabla_clases_3loc['EPFM_loc3'] = [weibull((x+1),df_3loc.loc[2, 'c'],df_3loc.loc[2, 'k']) for x in range(n_clases)]

    # ----------------------------------------------------------------------------
    # CALCULATING ROTOR'S POWER
    # ----------------------------------------------------------------------------

    areas = {
        'df_loc1': math.pi*(Diameter1/2)**2,
        'df_loc2': math.pi*(Diameter2/2)**2,
        'df_loc3': math.pi*(Diameter3/2)**2
    }

    tabla_clases_3loc['P_rotor_[MW]_loc1'] = [(Cp*(0.5)*rho*((x+1)**3)*areas['df_loc1']/1e6) for x in range(n_clases)]
    tabla_clases_3loc['P_rotor_[MW]_loc2'] = [(Cp*(0.5)*rho*((x+1)**3)*areas['df_loc2']/1e6) for x in range(n_clases)]
    tabla_clases_3loc['P_rotor_[MW]_loc3'] = [(Cp*(0.5)*rho*((x+1)**3)*areas['df_loc3']/1e6) for x in range(n_clases)]

    tabla_clases_3loc['P_EPFM_loc1'] = [(tabla_clases_3loc['P_rotor_[MW]_loc1'][x]*tabla_clases_3loc['EPFM_loc1'][x]*24*365) for x in range(n_clases)]
    tabla_clases_3loc['P_EPFM_loc2'] = [(tabla_clases_3loc['P_rotor_[MW]_loc2'][x]*tabla_clases_3loc['EPFM_loc2'][x]*24*365) for x in range(n_clases)]
    tabla_clases_3loc['P_EPFM_loc3'] = [(tabla_clases_3loc['P_rotor_[MW]_loc3'][x]*tabla_clases_3loc['EPFM_loc3'][x]*24*365) for x in range(n_clases)]

    # ----------------------------------------------------------------------------
    # ANNUAL ENERGY PRODUCTION
    # ----------------------------------------------------------------------------
    # ----- Location 1
    tabla_clases_3loc['tope_EPFM1'] = [tabla_clases_3loc['EPFM_loc1'][x]*(df_3loc.loc[0, 'Rated_power [MW]']*365*24) for x in range(n_clases)]
    energia_real_EPFM1 = list()

    for x in range(3, n_clases, 1):
        current_p_EPFM1 = tabla_clases_3loc['P_EPFM_loc1'][x]
        current_tope_EPFM1 = tabla_clases_3loc['tope_EPFM1'][x]
        if current_p_EPFM1 > current_tope_EPFM1:
            energia_real_EPFM1.append(current_tope_EPFM1)
        else:
            energia_real_EPFM1.append(current_p_EPFM1)

    # ------ Location 2
    tabla_clases_3loc['tope_EPFM2'] = [tabla_clases_3loc['EPFM_loc2'][x]*(df_3loc.loc[1, 'Rated_power [MW]']*365*24) for x in range(n_clases)]
    energia_real_EPFM2 = list()

    for x in range(3, n_clases, 1):
        current_p_EPFM2 = tabla_clases_3loc['P_EPFM_loc2'][x]
        current_tope_EPFM2 = tabla_clases_3loc['tope_EPFM2'][x]
        if current_p_EPFM2 > current_tope_EPFM2:
            energia_real_EPFM2.append(current_tope_EPFM2)
        else:
            energia_real_EPFM2.append(current_p_EPFM2)
    
    # ------ Location 3
    tabla_clases_3loc['tope_EPFM3'] = [tabla_clases_3loc['EPFM_loc3'][x]*(df_3loc.loc[2, 'Rated_power [MW]']*365*24) for x in range(n_clases)]
    energia_real_EPFM3 = list()

    for x in range(3, n_clases, 1):
        current_p_EPFM3 = tabla_clases_3loc['P_EPFM_loc3'][x]
        current_tope_EPFM3 = tabla_clases_3loc['tope_EPFM3'][x]
        if current_p_EPFM3 > current_tope_EPFM3:
            energia_real_EPFM3.append(current_tope_EPFM3)
        else:
            energia_real_EPFM3.append(current_p_EPFM3)
    
    df_3loc.loc[0, 'AEP [MW]'] = sum(energia_real_EPFM1)
    df_3loc.loc[1, 'AEP [MW]'] = sum(energia_real_EPFM2)
    df_3loc.loc[2, 'AEP [MW]'] = sum(energia_real_EPFM3)

    # ----------------------------------------------------------------------------
    # CAPACITY FACTOR
    # ----------------------------------------------------------------------------
    df_3loc.loc[0, 'Cf [%]'] = sum(energia_real_EPFM1)/(df_3loc.loc[0, 'Rated_power [MW]']*8760)*100
    df_3loc.loc[1, 'Cf [%]'] = sum(energia_real_EPFM2)/(df_3loc.loc[1, 'Rated_power [MW]']*8760)*100
    df_3loc.loc[2, 'Cf [%]'] = sum(energia_real_EPFM3)/(df_3loc.loc[2, 'Rated_power [MW]']*8760)*100

    # ----------------------------------------------------------------------------
    # LCOE
    # ----------------------------------------------------------------------------
    energias = [energia_real_EPFM1, energia_real_EPFM2, energia_real_EPFM3]

    for idx, energia in enumerate(energias):
        daily_energy_mwh = (sum(energia)/365)
        annual_energy_mwh = daily_energy_mwh * 365

        Pmax = df_3loc.loc[idx, 'Rated_power [MW]']
        capex_per_mw = 2.1 * 10**6 # $/MW (e.g., $1500/kW)
        annual_om_cost_per_mw = 15 * 10**3 # $/MW/year (e.g., $15/kW/year)

        # Calculate total CAPEX for the 0.35 MW system
        total_capex = capex_per_mw * Pmax # Total initial investment

        # Calculate annual O&M cost for the 0.35 MW system
        annual_om_cost = annual_om_cost_per_mw * Pmax

        # Calculate Present Value of Costs (PVC)
        PVC = total_capex
        for t in range(1, int(project_lifespan) + 1):
            PVC += annual_om_cost / ((1 + float(discount_rate))**t)

        # Calculate Present Value of Energy (PVE)
        PVE = 0
        for t in range(1, int(project_lifespan) + 1):
            PVE += annual_energy_mwh / ((1 + float(discount_rate))**t)

        # Calculate LCOE
        if PVE > 0:
            LCOE = PVC / PVE
        else:
            LCOE = 0

        df_3loc.loc[idx, 'LCOE [$/MWh]'] = LCOE

    df_3loc.drop(columns=['v_hub_mean', 'v_hub_desv', 'Epf', 'k', 'c'], inplace=True)

    df_calculos_3loc = df_3loc.copy()
    df_salida = df_calculos_3loc.to_dict('records')
    columns_table = [{"name": col, "id": col} for col in df_calculos_3loc.columns]

    contenedor_mapa = crear_mapa_dash(latitude1, longitude1, latitude2, longitude2, latitude3, longitude3)

    posicion = df_3loc['LCOE [$/MWh]'].idxmin() 

    best_lcoe = f"{df_3loc.loc[posicion, 'LCOE [$/MWh]']:.3f} $/MWh"
    best_location = f"{df_3loc.loc[posicion, 'Location']}"
    best_aep = f"{df_3loc.loc[posicion, 'AEP [MW]']:.3f} MW"
    
    return df_salida, columns_table, contenedor_mapa, best_location, best_aep, best_lcoe

if __name__ == '__main__':
    app.run(debug=True)