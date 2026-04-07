"""
tabs/prediccion.py
------------------
Pestaña 8 – Formulario de predicción en tiempo real.
Carga model.pkl y devuelve predicción + probabilidad.
"""

import os
import joblib
import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# Variables de entrada del formulario (Time, Amount, V1–V28)
V_COLS   = [f"V{i}" for i in range(1, 29)]
ALL_COLS = ["Time", "Amount"] + V_COLS

DEFAULTS = {
    "Time":   94811.0,
    "Amount": 88.47,
    **{f"V{i}": 0.0 for i in range(1, 29)},
}


def layout() -> html.Div:
    return html.Div([

        dbc.Row(dbc.Col(html.Div([
            html.H5("Predicción de fraude en tiempo real",
                    className="section-title", style={"marginTop": 0}),
            html.P(
                "Ingresa los valores de una transacción para obtener la predicción "
                "del modelo. Los campos V1–V28 corresponden a las componentes PCA del "
                "dataset original. Se utilizará el umbral de decisión de 0.13 para "
                "maximizar la detección de fraude.",
                className="section-body",
            ),
        ]), width=12), className="mb-4"),

        # Formulario
        dbc.Row([
            # Time + Amount
            dbc.Col(dbc.Card([
                html.Div("Variables de transacción", className="card-header-custom"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(_input_field("Time", "time-input",
                                             "Segundos desde la primera transacción",
                                             DEFAULTS["Time"]), md=6),
                        dbc.Col(_input_field("Amount", "amount-input",
                                             "Monto de la transacción (€)",
                                             DEFAULTS["Amount"]), md=6),
                    ]),
                ]),
            ]), md=12, className="mb-3"),
        ]),

        # V1–V14
        dbc.Row([dbc.Col(dbc.Card([
            html.Div("Componentes PCA (V1–V14)", className="card-header-custom"),
            dbc.CardBody(dbc.Row([
                dbc.Col(_input_field(f"V{i}", f"v{i}-input",
                                     f"Componente V{i}", 0.0), md=2)
                for i in range(1, 15)
            ])),
        ]), md=12)], className="mb-3"),

        # V15–V28
        dbc.Row([dbc.Col(dbc.Card([
            html.Div("Componentes PCA (V15–V28)", className="card-header-custom"),
            dbc.CardBody(dbc.Row([
                dbc.Col(_input_field(f"V{i}", f"v{i}-input",
                                     f"Componente V{i}", 0.0), md=2)
                for i in range(15, 29)
            ])),
        ]), md=12)], className="mb-4"),

        # Botón + resultado
        dbc.Row([
            dbc.Col(html.Button(
                "Predecir",
                id="predict-btn",
                className="predict-btn",
                n_clicks=0,
            ), width="auto"),
        ], className="mb-3"),

        dbc.Row(dbc.Col(html.Div(id="predict-result"), width=12)),

    ], className="tab-content-wrapper")




def _input_field(label: str, input_id: str, placeholder: str,
                 default_val: float) -> html.Div:
    return html.Div([
        html.Div(label, className="input-label"),
        dcc.Input(
            id=input_id,
            type="number",
            placeholder=placeholder,
            value=default_val,
            debounce=True,
            style={
                "width": "100%", "padding": "6px 10px",
                "border": "1px solid #cbd5e0", "borderRadius": "4px",
                "fontSize": "0.82rem", "fontFamily": "Segoe UI, Arial",
                "marginBottom": "10px",
            },
        ),
    ])




def register_callbacks(app) -> None:
    """
    Registra el callback de predicción.
    Se llama desde app.py después de inicializar la app.
    """

    @app.callback(
        Output("predict-result", "children"),
        Input("predict-btn", "n_clicks"),
        State("time-input",   "value"),
        State("amount-input", "value"),
        *[State(f"v{i}-input", "value") for i in range(1, 29)],
        prevent_initial_call=True,
    )
    def predict(n_clicks, time_val, amount_val, *v_vals):
        if not os.path.exists(MODEL_PATH):
            return dbc.Alert(
                "El modelo no está disponible. Ejecuta model/train_model.py primero.",
                color="warning",
            )

        # Construcción del vector de entrada
        values = [time_val, amount_val] + list(v_vals)
        if any(v is None for v in values):
            return dbc.Alert("Por favor completa todos los campos.", color="info")

        X = np.array(values, dtype="float64").reshape(1, -1)

        model = joblib.load(MODEL_PATH)
        prob  = model.predict_proba(X)[0, 1]
        pred  = int(prob >= 0.13)

        if pred == 1:
            css   = "result-box result-fraude"
            msg   = f"ALERTA: Transacción clasificada como FRAUDE"
            detail = f"Probabilidad estimada: {prob:.4f} (umbral: 0.13)"
        else:
            css   = "result-box result-no-fraude"
            msg   = "Transacción clasificada como NO FRAUDULENTA"
            detail = f"Probabilidad de fraude: {prob:.4f} (umbral: 0.13)"

        return html.Div([
            html.Div(msg, className=css),
            html.Div(detail, style={
                "textAlign": "center", "fontSize": "0.82rem",
                "color": "#718096", "marginTop": "8px",
            }),
        ])
