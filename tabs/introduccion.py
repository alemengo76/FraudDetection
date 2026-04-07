"""
tabs/introduccion.py
--------------------
Pestaña 1 – Introducción al problema de fraude financiero.
"""

import dash_bootstrap_components as dbc
from dash import html, Input, Output, State


def layout():
    return html.Div([
  
        
        dbc.Row(dbc.Col(html.Div([
            html.H4("Fraude en Transacciones con Tarjeta de Crédito",
                    className="section-title", style={"marginTop": 0}),
            html.P(
                "El fraude en transacciones con tarjeta de crédito de los principales peligros para las personas y entidades financieras a nivel mundial, entonces, poder detectarlo a tiempo es importante para reducir las pérdidas económicas, proteger a los clientes y mantener la confianza en los sistemas de pago electrónico. ",
                className="section-body"
            ),
        ]), width=12), className="mb-4"),

        #Cards de contexto
        dbc.Row([
            dbc.Col(_card_info(
                "Origen del problema",
                "El fraude con tarjeta de crédito apareció por primera vez en la década de los 80, cuando hubo un aumento significativo de los pagos electrónicos (Buonaguidi, 2017). Debido a esto, las instituciones financieras desarrollaron sistemas manuales para detectar actividades sospechosas. Luego empezaron a utilizar métodos estadísticos, y con el tiempo, se fueron incorporando técnicas de aprendizaje automático (Bolton & Hand, 2002; Tarazona Nieto et al., 2022).",
                 "card-1"
            ), md=4),
            dbc.Col(_card_info(
                "Desafío principal",
                "Una de las principales características que dificultan la detección del las transacciones fraudulentas es que es poco frecuente en comparación con las transacciones legítimas, por lo que hay un problema de desbalance de clases, lo que hace que la clase minoritaria esté subrepresentada y se afecte el análisis exploratorio y los modelos.",
                "card-2"
            ), md=4),
            dbc.Col(_card_info(
                "Dataset utilizado",
                "El análisis se basa en el dataset Credit Card Fraud Detection (Kaggle / ULB Machine Learning Group), el cual tiene las transacciones reales realizadas por titulares de tarjetas de crédito europeos durante dos días de septiembre de 2013. Las variables originales fueron anonimizadas por PCA, y se conservaron Time, Amount y la variable objetivo Class.",
                "card-3"
            ), md=4),
        ], className="mb-4"),

        # Impacto global
        dbc.Row(
    dbc.Col(
        dbc.Card([
            html.Div(
                "Impacto en el sistema financiero",
                className="card-header-custom",
                style={
                    "display": "flex",
                    "alignItems": "center"
                }
            ),

            dbc.CardBody([
                html.P(
                    "El fraude financiero no se limita a una pérdida económica, sus efectos van más allá:",
                    className="section-body"
                ),

                dbc.Row([
                    _impact_item("Pérdidas directas",
                        "Cargos no autorizados que afectan al banco y al cliente."),
                    _impact_item("Costos operacionales",
                        "Investigación, reversión de transacciones y atención al cliente."),
                    _impact_item("Falsos positivos",
                        "Bloquear tarjetas legítimas genera molestias y pérdida de confianza."),
                    _impact_item("Daño reputacional",
                        "Incidentes recurrentes afectan la credibilidad de los bancos."),
                ]),
            ]),
        ]),
        width=12
    ),
    className="mb-4"
),

        #Objetivo del dashboard 
        dbc.Row(dbc.Col(html.Div([
            html.H5("Sobre este dashboard", className="section-title"),
            html.P(
                "Esta aplicación integra el análisis exploratorio, las pruebas estadísticas y en el futuro, el modelo predictivo. Lo que permite analizar patrones de las variables, y en un futuro evaluar el modelo y hacer predicciones.",
                className="section-body"
            ),
        ]), width=12)),

    ], className="tab-content-wrapper")
    



# Funciones para usar arriba
def _card_info(title: str, body: str, card_id: str):
    return dbc.Card([
        html.Div(
            [
                html.Span(title, style={"flex": "1"}),
                html.Span("▼", id=f"{card_id}-arrow", style={"fontSize": "0.75rem"}),
            ],
            className="card-header-custom",
            id=f"{card_id}-toggle",
            style={"cursor": "pointer", "display": "flex", "alignItems": "center"},
        ),
        dbc.Collapse(
            dbc.CardBody(
                html.P(body, className="section-body", style={"marginBottom": 0})
            ),
            id=f"{card_id}-collapse",
            is_open=False
        ),
    ])

def _impact_item(title: str, desc: str):
    return dbc.Col(html.Div([
        html.Div(title, style={"fontWeight": 700, "fontSize": "0.85rem",
                                "color": "#1a2540", "marginBottom": "4px"}),
        html.Div(desc, className="section-body", style={"marginBottom": 0}),
    ], style={"borderLeft": "3px solid #2e86c1", "paddingLeft": "10px",
              "marginBottom": "12px"}), md=3)


def register_callbacks(app):

    for card_id in ["card-1", "card-2", "card-3"]:
        def make_toggle(cid):
            @app.callback(
                Output(f"{cid}-collapse", "is_open"),
                Output(f"{cid}-arrow", "style"),
                Input(f"{cid}-toggle", "n_clicks"),
                State(f"{cid}-collapse", "is_open"),
                prevent_initial_call=True
            )
            def toggle(n, is_open, _cid=cid):
                abierto = not is_open
                arrow_style = {
                    "fontSize": "0.75rem",
                    "transform": "rotate(180deg)" if abierto else "rotate(0deg)",
                    "transition": "transform 0.2s ease",
                }
                return abierto, arrow_style
        make_toggle(card_id)
