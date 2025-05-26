from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
import numpy as np
import joblib

# Colores personalizados
AZUL_ICFES = "#036BB4"
VERDE_ICFES = "#A0D068"

# Cargar modelo y scaler
modelo = load_model("mejor_modelo_f2.h5")
scaler = joblib.load("scaler.pkl")

# Cargar datos
df = pd.read_csv("data2016_P1_limpio.csv")
df_top10 = pd.read_csv("promedios_por_colegio.csv")

# App
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "ICFES Dashboard"

# Barra superior reutilizable
def barra_superior():
    return html.Div([
        html.Div(style={'width': '1440px', 'height': '99px', 'background': 'white', 'border': '1px solid black'}),
        html.Img(
            src='https://www.ucaldas.edu.co/portal/wp-content/uploads/2020/05/WhatsApp-Image-2020-05-22-at-7.16.14-PM.jpeg',
            style={'width': '143px', 'height': '57px', 'position': 'absolute', 'left': '57px', 'top': '21px'}
        ),
        dcc.Link(html.Div("Top 10 Peores Desempeños", style={
            'width': '118px', 'height': '33px', 'position': 'absolute', 'left': '920px', 'top': '35px',
            'textAlign': 'center', 'color': 'black', 'fontSize': '15px', 'fontFamily': 'Nunito', 'fontWeight': '800'
        }), href='/top10'),
        dcc.Link(html.Div("Predictor de Peor Área", style={
            'width': '118px', 'height': '33px', 'position': 'absolute', 'left': '1093px', 'top': '35px',
            'textAlign': 'center', 'color': 'black', 'fontSize': '15px', 'fontFamily': 'Nunito', 'fontWeight': '800'
        }), href='/predictor'),
        dcc.Link(html.Div("Visualizador de Resultaods", style={
            'width': '118px', 'height': '33px', 'position': 'absolute', 'left': '1265px', 'top': '35px',
            'textAlign': 'center', 'color': 'black', 'fontSize': '15px', 'fontFamily': 'Nunito', 'fontWeight': '800'
        }), href='/visualizador')
    ], style={'position': 'relative', 'zIndex': '1000'})

# Página de inicio
inicio_layout = html.Div(style={
    'width': '1440px', 'height': '1024px', 'position': 'relative', 'background': 'white', 'overflow': 'hidden'
}, children=[
    barra_superior(),
    html.Div("¡Bienvenido a los Resultados del ICFES 2016!", style={
        'width': '438px', 'height': '55px', 'left': '79px', 'top': '180px', 'position': 'absolute',
        'textAlign': 'center', 'color': 'black', 'fontSize': '24px', 'fontWeight': '600'
    }),
    html.Div("Selecciona una opción para continuar", style={
        'width': '435px', 'height': '45px', 'left': '91px', 'top': '270px', 'position': 'absolute',
        'textAlign': 'center', 'color': 'black', 'fontSize': '15px', 'fontWeight': '600'
    }),
    dcc.Link(html.Div(style={
        'width': '244px', 'height': '72px', 'left': '178px', 'top': '330px', 'position': 'absolute',
        'background': AZUL_ICFES, 'borderRadius': '26px'
    }, children=[html.Div("Top 10 Peores Desempeños", style={
        'width': '213px', 'height': '58px', 'margin': 'auto', 'marginTop': '7px', 'color': 'white', 'fontSize': '15px',
        'fontWeight': '600', 'textAlign': 'center', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
    ]), href='/top10'),
    dcc.Link(html.Div(style={
        'width': '244px', 'height': '72px', 'left': '178px', 'top': '440px', 'position': 'absolute',
        'background': AZUL_ICFES, 'borderRadius': '26px'
    }, children=[html.Div("Predictor de Peor Área", style={
        'width': '213px', 'height': '58px', 'margin': 'auto', 'marginTop': '7px', 'color': 'white', 'fontSize': '15px',
        'fontWeight': '600', 'textAlign': 'center', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
    ]), href='/predictor'),
    dcc.Link(html.Div(style={
        'width': '244px', 'height': '72px', 'left': '178px', 'top': '550px', 'position': 'absolute',
        'background': AZUL_ICFES, 'borderRadius': '26px'
    }, children=[html.Div("Visualizador de Resultados", style={
        'width': '213px', 'height': '58px', 'margin': 'auto', 'marginTop': '7px', 'color': 'white', 'fontSize': '15px',
        'fontWeight': '600', 'textAlign': 'center', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
    ]), href='/visualizador'),
    html.Img(
        src='https://www.icfes.gov.co/wp-content/uploads/2025/03/comunicao_3032025.jpeg',
        style={'width': '816px', 'height': '524px', 'left': '571px', 'top': '180px', 'position': 'absolute'}
    )
])

# Top 10
layout_top10 = html.Div([
    barra_superior(),
    html.H2("Top 10 Peores Desempeños", style={"textAlign": "center", "color": AZUL_ICFES}),
    html.Label("Seleccione un departamento", style={"marginLeft": "20px"}),
    dcc.Dropdown(id='departamento_dropdown', options=[
        {'label': d, 'value': d} for d in df_top10['cole_depto_ubicacion'].dropna().unique()
    ], style={"width": "60%", "margin": "auto"}),
    dcc.Graph(id='top10_grafico')
])

@app.callback(
    Output('top10_grafico', 'figure'),
    Input('departamento_dropdown', 'value')
)
def actualizar_top10(dep):
    if not dep:
        return px.bar(title="Seleccione un departamento")
    top = df_top10[df_top10["cole_depto_ubicacion"] == dep].sort_values("Promedio_Puntaje_Global").head(10)
    return px.bar(top, x="Promedio_Puntaje_Global", y="cole_nombre_establecimiento",
                  orientation='h', color_discrete_sequence=[AZUL_ICFES],
                  labels={'Promedio_Puntaje_Global': 'Puntaje Promedio', 'cole_nombre_establecimiento': 'Colegio'})


# Predictor

def estilo_label():
    return {
        'width': '417px', 'height': '38px', 'color': AZUL_ICFES,
        'fontSize': '21px', 'fontFamily': 'Nunito', 'fontWeight': '800',
        'display': 'flex', 'alignItems': 'center'
    }

def estilo_dropdown():
    return {
        'width': '218px', 'height': '38px', 'border': '1px solid black',
        'background': 'white', 'gap': '10px', 'marginRight': '5%'
    }

layout_predictor = html.Div([
    barra_superior(),

    html.Div("Predictor de Peor Área", style={
        'width': '560px', 'height': '33px', 'margin': 'auto', 'marginTop': '50px',
        'textAlign': 'center', 'color': AZUL_ICFES, 'fontSize': '36px',
        'fontFamily': 'Nunito', 'fontWeight': '800',
    }),

    html.Div([

        html.Div("Género", style=estilo_label()),
        dcc.Dropdown(id='genero', options=[
            {'label': 'Femenino', 'value': 0},
            {'label': 'Masculino', 'value': 1}
        ], style=estilo_dropdown()),

        html.Div("Educación Madre", style=estilo_label()),
        dcc.Dropdown(id='educacion_madre', options=[
            {'label': 'Primaria incompleta', 'value': 0},
            {'label': 'Primaria completa', 'value': 1},
            {'label': 'Secundaria completa', 'value': 2},
            {'label': 'Técnica/tecnológica incompleta', 'value': 3},
            {'label': 'Técnica/tecnológica completa', 'value': 4},
            {'label': 'Profesional completa', 'value': 5},
            {'label': 'No sabe', 'value': 6}
        ], style=estilo_dropdown()),

        html.Div("Estrato Vivienda", style=estilo_label()),
        dcc.Dropdown(id='estrato', options=[{'label': str(i), 'value': i} for i in range(1, 7)],
                     style=estilo_dropdown()),

        html.Div("Cuartos Hogar", style=estilo_label()),
        dcc.Input(id='cuartos', type='number', value=1, style=estilo_dropdown()),

        html.Div("Automóvil (sí/no)", style=estilo_label()),
        dcc.Dropdown(id='auto', options=[
            {'label': 'Sí', 'value': 1},
            {'label': 'No', 'value': 0}
        ], style=estilo_dropdown()),

        html.Div("Internet (sí/no)", style=estilo_label()),
        dcc.Dropdown(id='internet', options=[
            {'label': 'Sí', 'value': 1},
            {'label': 'No', 'value': 0}
        ], style=estilo_dropdown()),

    ], style={
        'width': '80%', 'margin': 'auto', 'display': 'grid',
        'gridTemplateColumns': '1fr 1fr', 'gap': '10px', 'marginTop': '40px'
    }),

    html.Div([
        html.Button("Calcular Peor Área", id="btn-predictor", n_clicks=0, style={
            'width': '250px', 'height': '90px', 'backgroundColor': AZUL_ICFES, 'borderRadius': '26px',
            'color': 'white', 'fontSize': '30px', 'fontFamily': 'Inter', 'fontWeight': '400', 'border': 'none'
        }),
        html.Div(id='output_pred', style={
            'marginTop': '50px', 'color': AZUL_ICFES, 'fontSize': '36px',
            'fontFamily': 'Nunito', 'fontWeight': '800'
        })
    ], style={
        'width': '80%', 'marginTop': '50px', 'display': 'flex',
        'flexDirection': 'column', 'alignItems': 'flex-end', 'marginRight': '5%'
    })
])


# --- Callback corregido ---

@app.callback(
    Output('output_pred', 'children'),
    Input('btn-predictor', 'n_clicks'),
    State('genero', 'value'),
    State('educacion_madre', 'value'),
    State('estrato', 'value'),
    State('cuartos', 'value'),
    State('auto', 'value'),
    State('internet', 'value')
)
def predecir(n, g, edum, e, c, a, internet):
    if n > 0 and all(x is not None for x in [g, edum, e, c, a, internet]):
        entrada = scaler.transform([[g, edum, e, c, a, internet]])
        pred = modelo.predict(entrada)
        areas = ['Matemáticas', 'Lectura crítica', 'Inglés', 'Ciencias naturales', 'Sociales']
        return f"La peor Área es :\n{areas[np.argmax(pred)]}"
    return "Por favor complete todos los campos."





# Visualizador
layout_visualizador = html.Div([
    barra_superior(),

    html.H2("Visualización de Resultados", style={"textAlign": "center", "color": AZUL_ICFES}),

    html.Div([
        dcc.Graph(
            figure=px.bar(
                df_top10.groupby("cole_depto_ubicacion", as_index=False)["Promedio_Puntaje_Global"]
                .mean()
                .sort_values("Promedio_Puntaje_Global", ascending=False),
                x="cole_depto_ubicacion",
                y="Promedio_Puntaje_Global",
                title="Promedio de Puntaje Global por Departamento",
                labels={"cole_depto_ubicacion": "Departamento", "Promedio_Puntaje_Global": "Puntaje Promedio"},
                color_discrete_sequence=[AZUL_ICFES]
            )
        )
    ], style={'padding': '40px'}),

    html.H3("Características en común", style={"textAlign": "left", "color": AZUL_ICFES, "marginTop": "40px"}),

    html.Div([
        dcc.Dropdown(
            id='filtro_variable',
            options=[
                {'label': 'Bilingüe', 'value': 'cole_bilingue'},
                {'label': 'Calendario', 'value': 'cole_calendario'},
                {'label': 'Carácter', 'value': 'cole_caracter'},
                {'label': 'Género', 'value': 'cole_genero'},
                {'label': 'Jornada', 'value': 'cole_jornada'},
                {'label': 'Naturaleza', 'value': 'cole_naturaleza'},
                {'label': 'Ubicación (Rural/Urbana)', 'value': 'cole_area_ubicacion'},
                {'label': 'Departamento', 'value': 'cole_depto_ubicacion'}
            ],
            placeholder="Seleccione una característica",
            style={'width': '50%', 'margin': 'auto'}
        )
    ], style={'marginTop': '30px'}),

    html.Div([
        dcc.Graph(id='grafico_caracteristicas')
    ], style={'marginTop': '40px'})
])

@app.callback(
    Output('grafico_caracteristicas', 'figure'),
    Input('filtro_variable', 'value')
)
def actualizar_grafico(variable):
    if not variable:
        return px.bar(title="Seleccione una característica para visualizar")

    df_filtrado = df_top10[df_top10["Promedio_Puntaje_Global"] <= df_top10["Promedio_Puntaje_Global"].quantile(0.25)]

    conteo = df_filtrado[variable].value_counts().reset_index()
    conteo.columns = [variable, 'Cantidad']

    return px.bar(conteo, x=variable, y='Cantidad',
                  title=f"Distribución de colegios con bajo desempeño por: {variable.replace('cole_', '').replace('_', ' ').capitalize()}",
                  color_discrete_sequence=[AZUL_ICFES])

# Navegación
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def mostrar_vista(path):
    if path == '/top10':
        return layout_top10
    elif path == '/predictor':
        return layout_predictor
    elif path == '/visualizador':
        return layout_visualizador
    return inicio_layout

if __name__ == '__main__':
    app.run(debug=True)





