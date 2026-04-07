"""
tabs/marco_teorico.py
---------------------
Pestaña 5 – Marco teórico y tabla de operacionalización de variables.
"""

import dash_bootstrap_components as dbc
from dash import html, dash_table, Input, Output, State


VARIABLES = [
    {
        "Variable": "Time",
        "Tipo": "Cuantitativa continua",
        "Nivel": "De razón",
        "Descripción": "Segundos transcurridos desde la primera transacción del dataset.",
        "Rol": "Predictora",
    },
    {
        "Variable": "V1–V28",
        "Tipo": "Cuantitativa continua",
        "Nivel": "De intervalo",
        "Descripción": "Componentes principales (PCA) que anonomizan las variables originales.",
        "Rol": "Predictoras",
    },
    {
        "Variable": "Amount",
        "Tipo": "Cuantitativa continua",
        "Nivel": "De razón",
        "Descripción": "Monto de la transacción en euros.",
        "Rol": "Predictora",
    },
    {
        "Variable": "Class",
        "Tipo": "Cualitativa nominal",
        "Nivel": "Dicotómica",
        "Descripción": "Indicador de fraude: 0 = legítima, 1 = fraudulenta.",
        "Rol": "Variable objetivo",
    },
]


def layout():
    return html.Div(
        [
            # Fraude financiero
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                html.Div([
                                    html.Span("Fraude financiero", style={"flex": "1"}),
                                    html.Span("▼", id="fraude-arrow", style={"fontSize": "0.75rem"}),
                                ], className="card-header-custom", id="fraude-toggle",
                                   style={"cursor": "pointer", "display": "flex", "alignItems": "center"}),
                                dbc.Collapse(
                                    dbc.CardBody(
                                        html.P(
                                            "El fraude con tarjeta de crédito es el uso ilegal y no autorizado de la tarjeta de crédito o de los datos de la tarjeta de otra persona para realizar transacciones o compras fraudulentas (Microblink, s.f.). Su detección temprana es esencial para reducir pérdidas, proteger usuarios y mantener confianza en los sistemas de pago digitales (Bolton & Hand, 2002).",
                                            className="section-body",
                                            style={"marginBottom": 0},
                                        )
                                    ),
                                    id="fraude-collapse",
                                    is_open=False,
                                ),
                            ]
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                html.Div([
                                    html.Span("Clasificación binaria", style={"flex": "1"}),
                                    html.Span("▼", id="clasificacion-arrow", style={"fontSize": "0.75rem"}),
                                ], className="card-header-custom", id="clasificacion-toggle",
                                   style={"cursor": "pointer", "display": "flex", "alignItems": "center"}),
                                dbc.Collapse(
                                    dbc.CardBody(
                                        html.P(
                                            "La detección de fraude se puede observar como un problema de clasificación "
                                            "binaria supervisada, en el que la variable objetivo toma el valor 1 "
                                            "(fraude) o 0 (no fraude). No obstante, como se mencionó anteriormente, la complejidad está en el "
                                            "desbalance de clases que exige estrategias específicas de modelado, que se profundizarán en el futuro",
                                            className="section-body",
                                            style={"marginBottom": 0},
                                        )
                                    ),
                                    id="clasificacion-collapse",
                                    is_open=False,
                                ),
                            ]
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                html.Div([
                                    html.Span("Regresión logística", style={"flex": "1"}),
                                    html.Span("▼", id="regresion-arrow", style={"fontSize": "0.75rem"}),
                                ], className="card-header-custom", id="regresion-toggle",
                                   style={"cursor": "pointer", "display": "flex", "alignItems": "center"}),
                                dbc.Collapse(
                                    dbc.CardBody(
                                        html.P(
                                            "La regresión logística es un modelo estadístico que predice la probabilidad de ocurrencia de un evento binario en función de un conjunto de variables predictoras, mediante la función sigmoide (Hosmer et al., 2013). "
                                            "Es interpretable, eficiente computacionalmente y da "
                                            "probabilidades calibradas, por lo que es base sólida para problemas de detección de fraude.",
                                            className="section-body",
                                            style={"marginBottom": 0},
                                        )
                                    ),
                                    id="regresion-collapse",
                                    is_open=False,
                                ),
                            ]
                        ),
                        md=4,
                    ),
                ],
                className="mb-4",
            ),

            #Pruebas estadísticas
            dbc.Row([

                dbc.Col(dbc.Card([
                    html.Div([
                        html.Span("Correlación de Spearman y VIF", style={"flex": "1"}),
                        html.Span("▼", id="spearman-vif-arrow", style={"fontSize": "0.75rem"}),
                    ], className="card-header-custom", id="spearman-vif-toggle",
                       style={"cursor": "pointer", "display": "flex", "alignItems": "center"}),
                    dbc.Collapse(
                        dbc.CardBody(html.P(
                            "La correlación de Spearman evalúa de forma no paramétrica la relación monótona entre dos variables sin asumir normalidad. El Factor de Inflación de la Varianza (VIF) mide la multicolinealidad entre variables explicativas: un VIF ≥ 5 indica multicolinealidad moderada y un VIF ≥ 10 indica multicolinealidad severa, y esto puede afectar la estabilidad de los coeficientes del modelo.",
                            className="section-body", style={"marginBottom": 0},
                        )),
                        id="spearman-vif-collapse", is_open=False,
                    ),
                ]), md=4),

                dbc.Col(dbc.Card([
                    html.Div([
                        html.Span("Prueba U de Mann-Whitney", style={"flex": "1"}),
                        html.Span("▼", id="mann-whitney-arrow", style={"fontSize": "0.75rem"}),
                    ], className="card-header-custom", id="mann-whitney-toggle",
                       style={"cursor": "pointer", "display": "flex", "alignItems": "center"}),
                    dbc.Collapse(
                        dbc.CardBody(html.P(
                            "Dado que las variables del dataset presentan distribuciones asimétricas, alta presencia "
                            "de valores atípicos y hay un marcado desbalance de clases, se realizó la prueba "
                            "no paramétrica de U de Mann-Whitney, que permite comparar las medianas de las distribuciones entre las transacciones no fraudulentas y fraudulentas sin asumir una distribución normal.",
                            className="section-body", style={"marginBottom": 0},
                        )),
                        id="mann-whitney-collapse", is_open=False,
                    ),
                ]), md=4),

                dbc.Col(dbc.Card([
                    html.Div([
                        html.Span("Poder discriminante", style={"flex": "1"}),
                        html.Span("▼", id="rbc-arrow", style={"fontSize": "0.75rem"}),
                    ], className="card-header-custom", id="rbc-toggle",
                       style={"cursor": "pointer", "display": "flex", "alignItems": "center"}),
                    dbc.Collapse(
                        dbc.CardBody([
                            html.P(
                                "Debido a que el gran tamaño muestral puede hacer que diferencias pequeñas resulten estadísticamente significativas, para medir la capacidad discriminativa de cada variable, se calculó "
                                "el coeficiente RBC (Rank Biserial Correlation) como medida de tamaño de "
                                "efecto, y un índice de solapamiento basado en los rangos intercuartílicos "
                                "(IQR) de ambas clases. Luego, tenemos: ",
                                className="section-body", style={"marginBottom": "12px"},
                            ),
                            html.Ul([
                                html.Li([
                                    html.Strong("Alto poder discriminante: "),
                                    "cuando los intervalos intercuartílicos no se solapan y el coeficiente RBC es |RBC| > 0.5.",
                                ], className="section-body", style={"marginBottom": "6px"}),
                                html.Li([
                                    html.Strong("Moderado poder discriminante: "),
                                    "cuando el tamaño del efecto es moderado (0.3 ≤ |RBC| ≤ 0.5), aunque los cuartiles puedan superponerse parcialmente.",
                                ], className="section-body", style={"marginBottom": "6px"}),
                                html.Li([
                                    html.Strong("Bajo poder discriminante: "),
                                    "cuando los cuartiles se solapan ampliamente y el tamaño del efecto es muy bajo (|RBC| < 0.3).",
                                ], className="section-body"),
                            ], style={"marginBottom": 0, "paddingLeft": "18px"}),
                        ]),
                        id="rbc-collapse", is_open=False,
                    ),
                ]), md=4),

            ], className="mb-4"),

            #Tabla de operacionalización
            dbc.Row(
                dbc.Col(
                    html.Div([
                        html.H5(
                            "Tabla de operacionalización de variables",
                            className="section-title",
                        ),
                    ]),
                    width=12,
                )
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card([
                        html.Div("Variables del dataset", className="card-header-custom"),
                        dbc.CardBody(
                            dash_table.DataTable(
                                data=VARIABLES,
                                columns=[
                                    {"name": "Variable", "id": "Variable"},
                                    {"name": "Tipo", "id": "Tipo"},
                                    {"name": "Nivel", "id": "Nivel"},
                                    {"name": "Descripción", "id": "Descripción"},
                                    {"name": "Rol", "id": "Rol"},
                                ],
                                style_table={"overflowX": "auto"},
                                style_header={
                                    "backgroundColor": "#1a2540",
                                    "color": "white",
                                    "fontWeight": "600",
                                    "fontSize": "0.82rem",
                                    "border": "none",
                                },
                                style_cell={
                                    "fontSize": "0.82rem",
                                    "padding": "10px 14px",
                                    "textAlign": "left",
                                    "border": "1px solid #e2e8f0",
                                    "color": "#2c3e50",
                                    "fontFamily": "Segoe UI, Arial, sans-serif",
                                    "whiteSpace": "normal",
                                    "height": "auto",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"row_index": "odd"},
                                        "backgroundColor": "#f8fafc",
                                    },
                                    {
                                        "if": {"state": "selected"},
                                        "backgroundColor": "#d6eaf8",
                                        "border": "1px solid #2e86c1",
                                    },
                                ],
                            )
                        ),
                    ]),
                    width=12,
                )
            ),
        ],
        className="tab-content-wrapper",
    )


def register_callbacks(app):

    for card_id in ["fraude", "clasificacion", "regresion", "spearman-vif", "mann-whitney", "rbc"]:
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