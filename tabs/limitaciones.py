"""
tabs/limitaciones.py
--------------------
Pestaña – Limitaciones del análisis exploratorio.
"""

import dash_bootstrap_components as dbc
from dash import html

COLORES = [
    "#1a5276",
    "#1f618d",
    "#2471a3",
    "#2e86c1",
    "#2980b9",
    "#5dade2",
]

LIMITACIONES = [
    (
        "Desbalance extremo de clases",
        "La clase fraude es solo el 0.17% del total (473 de 283.726 transacciones). Esto afecta el EDA, pues las correlaciones globales las domina la clase mayoritaria, los estadísticos descriptivos generales no reflejan el comportamiento real de las transacciones fraudulentas y las pruebas estadísticas pueden detectar diferencias significativas que en la práctica tienen un efecto muy pequeño.",
    ),
    (
        "Dependencia de la transformación PCA",
        "Las variables V1–V28 son componentes principales que anonimizan las variables originales del dataset y limita la interpretabilidad del análisis, porque no se puede conocer qué características reales del comportamiento del usuario corresponden a cada componente, lo que dificulta sacar conclusiones de negocio directamente a partir de los hallazgos del EDA.",
    ),
    (
        "Datos de un único período",
        "El dataset contiene transacciones de dos días de septiembre de 2013, pero los patrones de fraude pueden variar estacionalmente y con el tiempo, por lo que los hallazgos del análisis exploratorio podrían no generalizarse a períodos diferentes o a contextos geográficos distintos al europeo. A menos, que se requiera por un contexto del que se desconoce.",
    ),
    (
        "Umbral de decisión fijo",
        "El modelo usa el umbral de decisión por defecto de 0.5 para realizar las clasificaciones. Esto puede afectar el rendimiento en un entorno real, ya que dependiendo del nivel de riesgo que se quiera manejar, podría ser más importante reducir los falsos positivos o los falsos negativos. No obstante, dado que en este caso no sabemos los intereses del banco se mantuvo en 0.5.",
    ),
    (
        "Sesgo de supervivencia",
        "El dataset solo contiene transacciones que fueron procesadas y registradas. Las transacciones que son bloqueadas preventivamente por los sistemas de seguridad no se incluyen, lo que podría sesgar los patrones hacia fraudes que lograron evadir controles previos. Luego, se podría estar subestimando el fraude real y aprender patrones de fraudes 'exitosos' y no de todos los intentos.",
    ),
    (
        "Inferencia limitada por tamaño muestral en fraude",
        "Con solo 473 transacciones fraudulentas, los estadísticos calculados para esta clase (medianas, correlaciones, distribuciones) tienen mucha variabilidad muestral. Puesto que, pequeñas diferencias entre observaciones individuales pueden generar correlaciones o patrones 'fuertes' que no necesariamente reflejan la realidad, como se ve en el análisis temporal por hora.",
    ),
]


def _lim_card(title: str, body: str, accent: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card([
            html.Div(
                title,
                className="card-header-custom",
                style={"borderTop": f"3px solid {accent}"},
            ),
            dbc.CardBody(
                html.P(body, className="section-body", style={"marginBottom": 0})
            ),
        ], className="hallazgo-card"),
        md=4,
        className="mb-3"
    )


def layout():
    return html.Div([

        dbc.Row(dbc.Col(html.Div([
            html.H5("Limitaciones", className="section-title", style={"marginTop": 0}),
            html.P(
                "Todo análisis exploratorio está sujeto a restricciones que se derivan de los datos, "
                "la metodología y el contexto. A continuación se detallan "
                "las principales limitaciones identificadas en este proyecto.",
                className="section-body",
            ),
        ]), width=12), className="mb-4"),

        dbc.Row([
            _lim_card(title, body, COLORES[i])
            for i, (title, body) in enumerate(LIMITACIONES[:3])
        ], className="mb-2"),

        dbc.Row([
            _lim_card(title, body, COLORES[i])
            for i, (title, body) in enumerate(LIMITACIONES[3:], start=3)
        ]),

    ], className="tab-content-wrapper tab-fade-in")

