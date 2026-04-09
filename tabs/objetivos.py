"""
tabs/objetivos.py
-----------------
Pestaña 4 – Objetivos del proyecto.
"""

import dash_bootstrap_components as dbc
from dash import html


def layout():
    return html.Div([

        # Título
        dbc.Row(dbc.Col(html.Div([
            html.H5("Objetivos del proyecto", className="section-title",
                    style={"marginTop": 0}),
        ]), width=12), className="mb-4"),

        # Objetivo general
        dbc.Row(
            dbc.Col(
                dbc.Card([
                    html.Div(
                        "Objetivo general",
                        className="card-header-custom",
                        style={"display": "flex", "alignItems": "center"}
                    ),
                    dbc.CardBody(
                        html.P(
                            "Diseñar y desarrollar un dashboard interactivo que permita visualizar y analizar la interacción entre las variables de las transacciones con tarjeta de crédito, con la implementación de un modelo de predicción que facilite la identificación de patrones asociados al fraude y contribuya a la detección temprana de transacciones fraudulentas.",
                            className="section-body",
                            style={"marginBottom": 0},
                        )
                    ),
                ], className="hallazgo-card"),
                width=12
            ),
            className="mb-4"
        ),

        # Objetivos específicos
        dbc.Row(dbc.Col(html.Div([
            html.H5("Objetivos específicos", className="section-title"),
        ]), width=12), className="mb-3"),

        dbc.Row([
            _obj_card("1", "Descripción del dataset",
                      "Describir el dataset, su estructura general, los tipos de variables que se encuentran y realizar estadísticas descriptivas de estas."),
            _obj_card("2", "Evaluación del desbalance",
                      "Analizar el desbalance de la variable respuesta (Class) y evaluar las posibles implicaciones en la detección de fraude."),
            _obj_card("3", "Análisis distribucional",
                      "Estudiar la distribución de las variables, por medio de análisis univariado usando estadísticas y visualizaciones interactivas."),
            _obj_card("4", "Comparación entre clases",
                      "Comparar las características de las transacciones fraudulentas y "
                      "no fraudulentas mediante análisis descriptivos."),
        ], className="mb-3"),

        dbc.Row([
            _obj_card("5", "Identificación de patrones",
                      "Identificar posibles patrones asociados al fraude a partir del "
                      "análisis bivariado y multivariado de las variables."),
            _obj_card("6", "Análisis de correlación",
                      "Notar posibles relaciones entre variables mediante el coeficiente "
                      "de correlación de Spearman y el cálculo del VIF."),
            _obj_card("7", "Modelo predictivo",
                      "Entrenar y evaluar un modelo de regresión logística mediante GridSearchCV y validación estratificada."),
            _obj_card("8", "Sistema de predicción",
                      "Crear un formulario que le permita al usuario ingresar características de una transacción y poder predecir si es fraudulenta."),
        ]),

    ], className="tab-content-wrapper tab-fade-in")


def _obj_card(num: str, title: str, body: str):
    return dbc.Col(
        dbc.Card([
            html.Div([
                html.Span(num, style={
    "background": "#fff", "color": "#2e86c1",
    "borderRadius": "50%", "width": "26px", "height": "26px",
    "display": "inline-flex", "alignItems": "center",
    "justifyContent": "center", "fontSize": "0.8rem",
    "fontWeight": "700", "marginRight": "10px", "flexShrink": "0",
}),
                html.Span(title),
            ], className="card-header-custom",
               style={"display": "flex", "alignItems": "center"}),
            dbc.CardBody(
                html.P(body, className="section-body", style={"marginBottom": 0})
            ),
        ], className="hallazgo-card"),
        md=3, className="mb-3"
    )
