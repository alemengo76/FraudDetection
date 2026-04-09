"""
tabs/dataset.py
---------------
Pestaña – Carga y estructura del dataset.
"""

import dash_bootstrap_components as dbc
from dash import html, dash_table, dcc, Input, Output
import pandas as pd


from data.data_loader import get_df


def layout():
    df = get_df()
    n_filas = len(df)
    n_cols = len(df.columns)
    n_nulos = int(df.isnull().sum().sum())
    n_duplicados = 284807 - n_filas  # original - limpios
    time_min = int(df["Time"].min())
    time_max = int(df["Time"].max())

    # Tipos de variables
    n_num = int((df.dtypes != "object").sum())
    n_cat = int((df.dtypes == "object").sum())

    return html.Div(
        [
            # Título
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H5(
                                "Carga del dataset",
                                className="section-title",
                                style={"marginTop": 0},
                            ),
                            html.P(
                                "Estructura general del dataset Credit Card Fraud Detection "
                                "tras la limpieza y preparación de los datos.",
                                className="section-body",
                            ),
                        ]
                    ),
                    width=12,
                ),
                className="mb-4",
            ),
            # KPIs fila 1
            dbc.Row(
                [
                    _kpi(f"{n_filas:,}", "Filas (transacciones)", "kpi-accent"),
                    _kpi(f"{n_cols}", "Columnas", ""),
                    _kpi(f"{n_nulos}", "Valores nulos", ""),
                    _kpi(f"{n_duplicados:,}", "Duplicados eliminados", "kpi-danger"),
                ],
                className="mb-3",
            ),
            # KPIs fila 2
            dbc.Row(
                [
                    _kpi(f"{n_num}", "Variables numéricas", ""),
                    _kpi(f"{n_cat}", "Variables categóricas", ""),
                    _kpi(f"{time_min:,}s", "Time mínimo", ""),
                    _kpi(f"{time_max:,}s", "Time máximo", "kpi-accent"),
                ],
                className="mb-4",
            ),
            #datos atípicos
            #datos atípicos
dbc.Row(
    dbc.Col(
        dbc.Card(
            [
                html.Div(
                    "Análisis de datos atípicos",
                    className="card-header-custom",
                ),
                dbc.CardBody(
                    [
                        html.Ul(
                            [
                                html.Li(
                                    "Las variables V1-V28 corresponden a componentes principales que se aplicaron por razones de confidencialidad. "
                                    "Teniendo en cuenta que PCA produce variables que ya están normalizadas y centradas, el análisis de outliers "
                                    "clásico no se puede aplicar, pues los valores 'extremos' no son errores, son parte de una transformación "
                                    "matemática, tampoco se pueden interpretar, pues no se saben qué mezcla de variables originales producen los valores.",
                                    className="section-body",
                                    style={"marginBottom": "10px"},
                                ),
                                html.Li(
                                    f"En cuanto a Time, el rango del dataset va de {time_min:,}s a "
                                    f"{time_max:,}s, lo que corresponde exactamente a los dos días del estudio. "
                                    "Es un timestamp real que registra los segundos "
                                    "transcurridos desde la primera transacción, no una "
                                    "hora del día.",
                                    className="section-body",
                                    style={"marginBottom": "10px"},
                                ),
                                html.Li(
                                    "Finalmente, en cuanto a las transacciones, en el contexto de detección de fraude, los valores extremos "
                                    "no se eliminan, ya que pueden corresponder precisamente a transacciones "
                                    "fraudulentas, que es lo que nos interesa detectar.",
                                    className="section-body",
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        width=12,
    ),
    className="mb-4",
),
            # Vista previa
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            html.Div(
                                "Vista previa del dataset",
                                className="card-header-custom",
                            ),
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Número de filas a mostrar:",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "color": "#1a2540",
                                                    "fontWeight": "600",
                                                    "marginBottom": "8px",
                                                },
                                            ),
                                            dcc.Slider(
                                                id="slider-filas",
                                                min=5,
                                                max=50,
                                                step=5,
                                                value=10,
                                                marks={
                                                    i: str(i) for i in range(5, 51, 5)
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "20px"},
                                    ),
                                    html.Div(id="tabla-preview"),
                                ]
                            ),
                        ]
                    ),
                    width=12,
                ),
                className="mb-4",
            ),
            html.P(
    "Para la descripción detallada de cada variable consulta la tabla de operacionalización en el marco teórico.",
    className="section-body",
    style={"fontStyle": "italic", "color": "#64748b", "marginBottom": "16px"},
),
        ],
        className="tab-content-wrapper tab-fade-in",
    )


def register_callbacks(app):

    @app.callback(
        Output("tabla-preview", "children"),
        Input("slider-filas", "value"),
    )
    def actualizar_tabla(n):
        df = get_df()
        muestra = df.head(n).round(4)
        return dash_table.DataTable(
            data=muestra.to_dict("records"),
            columns=[{"name": c, "id": c} for c in muestra.columns],
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#1a2540",
                "color": "white",
                "fontWeight": "600",
                "fontSize": "0.82rem",
                "border": "none",
            },
            style_cell={
                "fontSize": "0.8rem",
                "padding": "8px 12px",
                "textAlign": "left",
                "border": "1px solid #e2e8f0",
                "color": "#2c3e50",
                "fontFamily": "Segoe UI, Arial, sans-serif",
                "minWidth": "80px",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
                {
                    "if": {"state": "selected"},
                    "backgroundColor": "#d6eaf8",
                    "border": "1px solid #2e86c1",
                },
            ],
        )


def _kpi(value: str, label: str, accent: str) -> dbc.Col:
    return dbc.Col(
        html.Div(
            [
                html.Div(value, className=f"kpi-value {accent}"),
                html.Div(label, className="kpi-label"),
            ],
            className="kpi-card",
        ),
        md=3,
        className="mb-3",
    )
