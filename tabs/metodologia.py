"""
tabs/metodologia.py
-------------------
Pestaña 6 – Metodología del proyecto.
"""

import dash_bootstrap_components as dbc
from dash import html, Input, Output, State


def layout():
    return html.Div([

        dbc.Row(dbc.Col(html.Div([
            html.H5("Metodología", className="section-title", style={"marginTop": 0}),
            html.P(
                "El proyecto se desarrolla bajo un enfoque cuantitativo de alcance descriptivo-correlacional. Tiene las siguientes fases:",
                className="section-body",
            ),
        ]), width=12), className="mb-4"),

        # Fases metodológicas
        dbc.Row([
            _fase("01", "Fuente de datos",
                  "Dataset",
                  "Se utilizó el dataset Credit Card Fraud Detection (Kaggle / ULB Machine Learning Group), que tiene 284.807 transacciones realizadas por titulares de tarjetas de crédito europeos en dos días de septiembre de 2013."),
            _fase("02", "Limpieza y preparación",
                  "Preprocesamiento",
                  "Se identificaron y eliminaron 1.081 registros duplicados, quedando 283.726 transacciones únicas. También, se verificó la presencia de valores faltantes (NaN), sin encontrar ninguno. La variable Class fue transformada a string para el EDA y a float64 para el modelado."),
        ], className="mb-3"),

        dbc.Row([
            _fase("03", "Análisis Exploratorio",
                  "EDA",
                  "Se realizó un análisis univariado y bivariado con estadísticas y visualizaciones interactivas. Se incluyen la prueba no paramétrica U de Mann-Whitney, el coeficiente RBC como tamaño de efecto, y un análisis multivariado con matrices de correlación de Spearman y cálculo del VIF."),
            _fase("04", "Dashboard interactivo",
                  "Implementación en Dash",
                  "Los resultados del análisis se integraron en una aplicación Dash modular (Detección de Fraude Financiero) con arquitectura de tabs. En el futuro se planea implementar un modelo de regresión logística para generar un formulario intaractivo."),
        ], className="mb-3"),
            

        # Diagrama de flujo 
        dbc.Row(dbc.Col(dbc.Card([
            html.Div("Flujo del pipeline", className="card-header-custom"),
            dbc.CardBody(html.Div([
                _step("Carga del CSV", "creditcard.csv → pandas"),
                _arrow(),
                _step("Limpieza", "Eliminación de duplicados"),
                _arrow(),
                _step("EDA", "Univariado → Bivariado → Multivariado"),
                _arrow(),
                _step("Entrenamiento", "GridSearchCV + StratifiedKFold"),
                _arrow(),
                _step("Evaluación", "Precision / Recall / F1 / ROC-AUC"),
                _arrow(),
                _step("Predicción", "Formulario interactivo"),
            ], style={"display": "flex", "alignItems": "center",
                      "flexWrap": "wrap", "gap": "4px"})),
        ]), width=12)),

    ], className="tab-content-wrapper")


#funciones para que se vea bonito

def _fase(num: str, subtitle: str, title: str, body: str):
    card_id = f"fase-{num}"
    return dbc.Col(dbc.Card([
        html.Div([
            html.Span(num, style={
                "fontSize": "1.6rem", "fontWeight": "800",
                "color": "#2e86c1", "lineHeight": "1",
            }),
            html.Div([
                html.Div(title, style={"fontWeight": "700", "fontSize": "0.9rem",
                                       "color": "#1a2540"}),
                html.Div(subtitle, style={"fontSize": "0.75rem", "color": "#718096"}),
            ], style={"marginLeft": "12px", "flex": "1"}),
            html.Div([
                html.Span("▼", id=f"{card_id}-arrow",
                          style={"fontSize": "0.7rem", "color": "#2e86c1"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={"display": "flex", "alignItems": "center",
                  "padding": "14px 18px", "borderBottom": "1px solid #e2e8f0",
                  "cursor": "pointer"},
           id=f"{card_id}-toggle"),
        dbc.Collapse(
            dbc.CardBody(html.P(body, className="section-body",
                                style={"marginBottom": 0})),
            id=f"{card_id}-collapse",
            is_open=False,
        ),
    ]), md=6, className="mb-3")


def register_callbacks(app):
    for i in ["01", "02", "03", "04"]:
        _make_callback(app, f"fase-{i}")


def _make_callback(app, card_id: str):
    @app.callback(
        Output(f"{card_id}-collapse", "is_open"),
        Output(f"{card_id}-arrow", "style"),
        Input(f"{card_id}-toggle", "n_clicks"),
        State(f"{card_id}-collapse", "is_open"),
        prevent_initial_call=True
    )
    def toggle(n, is_open):
        abierto = not is_open
        arrow_style = {
            "fontSize": "0.7rem", "color": "#2e86c1",
            "transform": "rotate(180deg)" if abierto else "rotate(0deg)",
            "transition": "transform 0.2s ease",
        }
        return abierto, arrow_style


def _step(label: str, sub: str):
    return html.Div([
        html.Div(label, style={"fontWeight": "700", "fontSize": "0.8rem",
                                "color": "#fff", "lineHeight": "1"}),
        html.Div(sub, style={"fontSize": "0.7rem", "color": "#c5d8e8",
                              "marginTop": "2px"}),
    ], style={
        "background": "#1a2540", "padding": "10px 14px",
        "borderRadius": "6px", "minWidth": "120px", "textAlign": "center",
    })


def _arrow():
    return html.Span("→", style={"color": "#2e86c1", "fontWeight": "700",
                                   "fontSize": "1.2rem", "padding": "0 4px"})
