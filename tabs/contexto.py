"""
tabs/contexto.py
----------------
Pestaña 2 – Contexto empresarial, costos y desbalance de clases.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

from data.data_loader import get_df


# grafico de pastel
def _fig_dona():
    df = get_df()
    counts = df["Class"].value_counts().sort_index()
    labels = ["No fraude (0)", "Fraude (1)"]
    values = [counts.get("0", 0), counts.get("1", 0)]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker_colors=["#2e86c1", "#c0392b"],
            textinfo="percent+label",
            textfont_size=12,
        )
    )
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=False,
        paper_bgcolor="white",
        annotations=[
            dict(
                text=f"{values[1]:,}<br>fraudes",
                x=0.5,
                y=0.5,
                font_size=14,
                showarrow=False,
                font_color="#c0392b",
            )
        ],
    )
    return fig


def layout():
    df = get_df()
    n = len(df)
    n_f = (df["Class"] == "1").sum()
    n_nf = (df["Class"] == "0").sum()
    pct = n_f / n * 100

    return html.Div(
        [
            # KPIs
            dbc.Row(
                [
                    _kpi(f"{n:,}", "Total de transacciones", "kpi-accent"),
                    _kpi(f"{n_nf:,}", "No fraudulentas", ""),
                    _kpi(f"{n_f:,}", "Fraudulentas", "kpi-danger"),
                    _kpi(f"{pct:.2f}%", "Tasa de fraude", "kpi-danger"),
                ],
                className="mb-4",
            ),

            # Dona + columna derecha
            dbc.Row(
                [
                    # Dona
                    dbc.Col(
                        dbc.Card([
                            html.Div("Distribución de clases", className="card-header-custom"),
                            dbc.CardBody(
                                dcc.Graph(figure=_fig_dona(), config={"displayModeBar": False})
                            ),
                        ]),
                        md=5,
                    ),

                    # Columna derecha: desbalance + costos
                    dbc.Col([

                        # Desbalance
                        dbc.Card([
    html.Div(
        "Desbalance de clases",
        className="card-header-custom",
        style={
            "display": "flex",
            "alignItems": "center"
        }
    ),

    dbc.CardBody([
        html.P(
            "El dataset presenta un desbalance muy fuerte en la variable respuesta Class. Se tienen 283.726 transacciones totales, el 99.83% corresponde a transacciones normales y solo el 0.17% a transacciones fraudulentas.",
            className="section-body",
        ),
        html.P(
            "Este desequilibrio tiene implicaciones directas sobre los modelos predictivos:",
            className="section-body",
        ),
        html.Ul([
            html.Li(
                "Los modelos usualmente tienden a clasificar todo como no fraude y aun así obtener alta precisión (accuracy).",
                style={"fontSize": "0.87rem", "color": "#4a5568", "marginBottom": "6px"}
            ),
            html.Li(
                "Con esto en mente, se debe tener en cuenta al recall como una métrica muy importante, pues un falso negativo implica no detectar una transacción fraudulenta.",
                style={"fontSize": "0.87rem", "color": "#4a5568", "marginBottom": "6px"}
            ),
            html.Li(
                "Se debe emplear validación cruzada estratificada (StratifiedKFold) para mantener la proporción de clases en cada fold.",
                style={"fontSize": "0.87rem", "color": "#4a5568"}
            ),
        ]),
    ]),
], className="mb-3"),

                        # Falso negativo
                        dbc.Card([
                            html.Div([
    html.Span("Costo del fraude no detectado (Falso Negativo)", style={"flex": "1"}),
    html.Span("▼", id="falso-negativo-arrow", style={"fontSize": "0.75rem"}),
], className="card-header-custom", id="falso-negativo-toggle",
   style={"cursor": "pointer", "display": "flex", "alignItems": "center"}),
                            dbc.Collapse(
                                dbc.CardBody(html.P(
                                    "Un falso negativo ocurre cuando el modelo clasifica una transacción fraudulenta como legítima. Las consecuencias incluyen pérdidas económicas tanto para la empresa como para el cliente, daño en la reputación de los bancos y posibles sanciones regulatorias. En contextos financieros, este error es considerado el más costoso.",
                                    className="section-body", style={"marginBottom": 0},
                                )),
                                id="falso-negativo-collapse",
                                is_open=False,
                            ),
                        ], className="mb-3"),

                        # Falso positivo
                        dbc.Card([
                            html.Div([
    html.Span("Costo de las alarmas falsas (Falso Positivo)", style={"flex": "1"}),
    html.Span("▼", id="falso-positivo-arrow", style={"fontSize": "0.75rem"}),
], className="card-header-custom", id="falso-positivo-toggle",
   style={"cursor": "pointer", "display": "flex", "alignItems": "center"}),
                            dbc.Collapse(
                                dbc.CardBody(html.P(
                                    "Un falso positivo bloquea una transacción legítima, aunque se podría pensar que no es tan grave como un falso negativo, en realidad genera molestias en la experiencia del cliente, llamadas al soporte y posible pérdida de confianza en el servicio. Por lo que se debe encontrar un balance entre precision y recall, por medio de la optimización de F1-score según la tolerancia al riesgo de las empresas.",
                                    className="section-body", style={"marginBottom": 0},
                                )),
                                id="falso-positivo-collapse",
                                is_open=False,
                            ),
                        ]),

                    ], md=7),
                ],
                className="mb-4",
            ),
        ],
        className="tab-content-wrapper",
    )


# funcionesss


def _kpi(value: str, label: str, accent: str):
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

def register_callbacks(app):

    for toggle_id in ["falso-negativo", "falso-positivo"]:
        def make_toggle(tid):
            @app.callback(
                Output(f"{tid}-collapse", "is_open"),
                Output(f"{tid}-arrow", "style"),
                Input(f"{tid}-toggle", "n_clicks"),
                State(f"{tid}-collapse", "is_open"),
                prevent_initial_call=True
            )
            def toggle(n, is_open, _tid=tid):
                abierto = not is_open
                arrow_style = {
                    "fontSize": "0.75rem",
                    "transform": "rotate(180deg)" if abierto else "rotate(0deg)",
                    "transition": "transform 0.2s ease",
                }
                return abierto, arrow_style
        make_toggle(toggle_id)