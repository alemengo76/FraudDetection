"""
tabs/conclusiones.py
--------------------
Pestaña – Síntesis, Conclusiones, Pasos Futuros y Referencias.
"""

import dash_bootstrap_components as dbc
from dash import html, Output, Input, State

#paleta de colores
C_BLUE   = "#1f618d"
C_RED    = "#2471a3"
C_GREEN  = "#2e86c1"
C_GOLD   = "#2980b9"
C_PURPLE = "#5dade2"
C_DARK   = "#1a2540"


#funciones auxiliares

def _kpi_card(value: str, label: str, color: str):
    return dbc.Col(
        html.Div([
            html.Div(value, style={
                "fontSize": "2rem",
                "fontWeight": "800",
                "color": color,
                "lineHeight": "1",
                "letterSpacing": "-0.5px",
            }),
            html.Div(label, style={
                "fontSize": "0.82rem",
                "color": "#64748b",
                "fontWeight": "600",
                "textTransform": "uppercase",
                "letterSpacing": "0.05em",
                "marginTop": "6px",
            }),
        ], className="kpi-card",
            style={
            "background": "#fff",
            "border": f"2px solid {color}",
            "borderRadius": "12px",
            "padding": "20px 18px",
            "textAlign": "center",
            "boxShadow": f"0 4px 20px {color}22",
            "height": "100%",
        }),
        md=3, className="mb-3",
    )


def _hallazgo_card(numero: str, titulo: str, cuerpo: str, color: str):
    return dbc.Col(
        html.Div([
            #Barra de acento
            html.Div(style={
                "width": "36px",
                "height": "4px",
                "backgroundColor": color,
                "borderRadius": "2px",
                "marginBottom": "10px",
            }),
            html.Div(titulo, style={
                "fontSize": "0.92rem",
                "fontWeight": "700",
                "color": C_DARK,
                "marginBottom": "8px",
                "lineHeight": "1.3",
            }),
            html.P(cuerpo, style={
                "fontSize": "0.85rem",
                "color": "#475569",
                "lineHeight": "1.6",
                "marginBottom": 0,
            }),
        ], className="hallazgo-card",
            style={
            "background": "#fff",
            "borderRadius": "12px",
            "padding": "20px 16px 16px 16px",
            "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
            "borderTop": f"4px solid {color}",
            "position": "relative",
            "overflow": "hidden",
            "height": "100%",
            
        }),
        md=4, className="mb-3",
    )


def _paso_card(numero: str, titulo: str, cuerpo: str, color: str, id_base: str):
    return dbc.Col(
        html.Div([
            html.Div([
                html.Div(numero, style={
                    "width": "36px",
                    "height": "36px",
                    "borderRadius": "50%",
                    "backgroundColor": color,
                    "color": "#fff",
                    "fontWeight": "800",
                    "fontSize": "0.85rem",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "flexShrink": "0",
                }),
                html.Div(titulo, style={
                    "fontWeight": "700",
                    "fontSize": "0.92rem",
                    "color": C_DARK,
                    "marginLeft": "12px",
                    "flex": "1",
                }),
                html.Span("▼", id=f"{id_base}-arrow", style={  # 👈 id agregado
                    "fontSize": "0.7rem",
                    "color": color,
                    "marginLeft": "8px",
                }),
            ], id=f"{id_base}-toggle",
               style={"display": "flex", "alignItems": "center",
                      "cursor": "pointer"}),
            dbc.Collapse(
                html.P(cuerpo, style={
                    "fontSize": "0.85rem",
                    "color": "#475569",
                    "lineHeight": "1.6",
                    "marginBottom": 0,
                    "paddingLeft": "48px",
                    "marginTop": "10px",
                }),
                id=f"{id_base}-collapse",
                is_open=False,
            ),
        ], style={
            "background": "#fff",
            "borderRadius": "12px",
            "padding": "18px 16px",
            "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
            "borderLeft": f"4px solid {color}",
            "height": "100%",
        }),
        md=6, className="mb-3",
    )


def _ref(texto: str):
    return html.P(texto, style={
        "fontSize": "0.85rem",
        "color": "#475569",
        "lineHeight": "1.6",
        "marginBottom": "8px",
        "paddingLeft": "16px",
        "borderLeft": f"3px solid #e2e8f0",
    })


#Layout

def layout():
    return html.Div([

        #Título
        dbc.Row(dbc.Col(html.Div([
            html.H5("Síntesis y Conclusiones", className="section-title",
                    style={"marginTop": 0}),
            html.P(
                "Principales hallazgos del análisis exploratorio del dataset de fraude en "
                "transacciones con tarjeta de crédito.",
                className="section-body",
            ),
        ]), width=12), className="mb-4"),

        #KPIs
        dbc.Row([
            _kpi_card("283.726", "Transacciones analizadas", C_BLUE),
            _kpi_card("31",      "Variables analizadas",     C_PURPLE),
            _kpi_card("0.17%",   "Prevalencia de fraude",    C_RED),
            _kpi_card("V14",     "Mayor poder discriminante", C_GREEN),
        ], className="mb-4"),

        #Hallazgos
        dbc.Row(dbc.Col(html.Div([
            html.H5("Hallazgos clave", className="section-title"),
            html.P(
                "Los principales patrones identificados en el análisis exploratorio.",
                className="section-body",
            ),
        ]), width=12), className="mb-3"),

        dbc.Row([
            _hallazgo_card(
                "01",
                "Desbalance de clases extremo",
                "Solo 473 transacciones (0.17%) son fraudulentas frente a 283.253 no "
                "fraudulentas. Este desbalance es el principal desafío y "
                "exige estrategias como SMOTE o ajuste de pesos de clase.",
                C_RED,
            ),
            _hallazgo_card(
                "02",
                "Variables PCA con alto poder discriminativo",
                "V2, V3, V4, V7, V9, V10, V11, V12, V14, V16 y V17 presentan rangos "
                "intercuartílicos separados entre clases y RBC > 0.5, siendo las más "
                "útiles para detectar fraude. V14 lidera con RBC = −0.894.",
                C_BLUE,
            ),
            _hallazgo_card(
                "03",
                "Time y Amount: significativas pero poco discriminativas",
                "Aunque presentan diferencias estadísticamente significativas (p < 0.05), "
                "tienen RBC bajo (−0.169 y −0.112) que indican poco poder discriminativo. "
                "Como están, no aportan mucho al modelo.",
                C_GOLD,
            ),
            _hallazgo_card(
                "04",
                "Patrones internos en transacciones fraudulentas",
                "Las correlaciones entre V1–V18 son moderadas a fuertes en el grupo "
                "fraude (hasta r ≈ 0.96), mientras que en no fraude son cercanas a 0, "
                "revelando estructura interna consistente en las transacciones fraudulentas.",
                C_PURPLE,
            ),
            _hallazgo_card(
                "05",
                "Baja multicolinealidad en variables PCA",
                "Las componentes V1–V28 presentan VIF bajos, lo que confirma la ortogonalidad "
                "del PCA. Solo Amount muestra multicolinealidad severa (VIF = 12.30), por "
                "lo que podría generar redundancia al incluirse junto al resto.",
                C_GREEN,
            ),
            _hallazgo_card(
                "06",
                "Fraude en transacciones de monto bajo",
                "La mediana del monto en fraude (9.82 €) es menor que en no fraude "
                "(22 €), esto sugiere que el fraude típico no ocurre en transacciones de "
                "alto valor. También, la distribución de Amount es muy asimétrica en ambas clases.",
                C_RED,
            ),
        ], className="mb-4"),

        dbc.Row(dbc.Col(html.H5("Conclusión", className="section-title"), width=12),
        className="mb-2"),

dbc.Row(dbc.Col(dbc.Card([
    html.Div("Conclusión general", className="card-header-custom"),
    dbc.CardBody([
        html.P(
            "El EDA del dataset de fraude en transacciones con tarjeta de crédito permitió identificar patrones que diferencian las transacciones fraudulentas de las no fraudulentas. A lo largo de este se aplicaron técnicas descriptivas univariadas y bivariadas, pruebas no paramétricas (U de Mann-Whitney), medidas de tamaño de efecto (RBC), análisis de correlación de Spearman y cálculo de VIF, que nos permiten entender las variables.",
            className="section-body",
        ),
        html.P(
            "Las variables con mayor capacidad discriminativa son las componentes PCA V14, V12, V4, V11 y V10, todas con RBC > 0.8 en valor absoluto y rangos intercuartílicos sin solapamiento entre clases. En contraste, a pesar de que Time y Amount son estadísticamente significativas, tienen un tamaño de efecto pequeño y mucho solapamiento, es decir no discriminan de forma efectiva solas. Algo importante es que las transacciones fraudulentas muestran patrones de correlación interna más fuertes entre V1–V18 que las no fraudulentas, esto muestra una estructura consistente en la clase minoritaria.",
            className="section-body",
        ),
        html.P(
            "El fuerte desbalance de clases (99.83% vs 0.17%) es el principal desafío para el futuro modelo. También, la multicolinealidad de Amount (VIF = 12.30) dice que hay que tener cuidado al incluirla junto con el resto de variables en el modelo. En general, el EDA muestra que las componentes PCA con alto poder discriminativo van a ser las variables más valiosas para el modelado, mientras que Time y Amount necesitan transformaciones o combinación con otras variables para mejorar su aporte al modelo.",
            className="section-body", style={"marginBottom": 0},
        ),
    ]),
],className="hallazgo-card"), width=12), className="mb-4"),

    
        dbc.Row(dbc.Col(html.Div([
    html.H5("Pasos futuros", className="section-title"),
    html.P(
        "A partir de los hallazgos del EDA, se plantean los siguientes pasos para "
        "la etapa de modelado predictivo.",
        className="section-body",
    ),
]), width=12), className="mb-3"),

dbc.Row([
    _paso_card("1", "Modelado predictivo base",
               "Implementar un modelo de regresión logística como modelo base de clasificación binaria, para aprovechar las componentes PCA con alto poder discriminativo que se identificaron.",
               C_BLUE, "paso1"),
    _paso_card("2", "Técnicas de balanceo de clases",
               "Aplicar SMOTE o undersampling por el desbalance (0.17% fraude) y así, mejorar la capacidad del modelo para aprender patrones de la clase fraude.",
               C_RED, "paso2"),
    _paso_card("3", "Métricas ajustadas al desbalance",
               "Se debe evaluar el modelo con métricas como F1-score, AUC-ROC y Recall, pues si se usa accuracy, se pueden tener scores altos a pesar del bajo desempeño en la detección del fraude.",
               C_GREEN, "paso3"),
    _paso_card("4", "Comparación de modelos",
               "Usar modelos más complejos como Random Forest o XGBoost después de la regresión logística base, para evaluar relaciones no lineales entre las variables predictoras y la objetivo (Class).",
               C_PURPLE, "paso4"),
], className="mb-4"),


        dbc.Row(dbc.Col(html.H5("Referencias", className="section-title"), width=12),
                className="mb-2"),

        dbc.Row(dbc.Col(dbc.Card([
            html.Div("Referencias", className="card-header-custom"),
            dbc.CardBody([
                _ref("Microblink. (s.f.). Fraude con tarjetas de crédito. "
                     "https://microblink.com/es/resources/glossary/credit-card-fraud/"),
                _ref("Buonaguidi, B. (2017). Cómo se producen los fraudes con tarjeta de "
                     "crédito. Euromonitor International."),
                _ref("Bolton, R. J., & Hand, D. J. (2002). Statistical Fraud Detection: "
                     "A Review. Statistical Science, 17(3), 235–255."),
                _ref("Tarazona Nieto et al. (2022). Detección de fraude financiero mediante "
                     "técnicas de machine learning. Revisión sistemática de literatura."),
                _ref("Dal Pozzolo, A. et al. (2015). Calibrating probability with "
                     "undersampling for unbalanced classification. IEEE SSCI."),
                _ref("Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). "
                     "Applied Logistic Regression (3rd ed.). Wiley."),
            ]),
        ], className="hallazgo-card"), width=12)),

    ], className="tab-content-wrapper tab-fade-in")
    
    
def register_callbacks(app):

    pasos = [
        ("paso1", C_BLUE),
        ("paso2", C_RED),
        ("paso3", C_GREEN),
        ("paso4", C_PURPLE),
    ]

    for id_base, color in pasos:
        def make_toggle(idb, col):
            @app.callback(
                Output(f"{idb}-collapse", "is_open"),
                Output(f"{idb}-arrow", "style"),
                Input(f"{idb}-toggle", "n_clicks"),
                State(f"{idb}-collapse", "is_open"),
                prevent_initial_call=True,
            )
            def toggle_paso(n, is_open, _idb=idb, _col=col):
                abierto = not is_open
                arrow_style = {
                    "fontSize": "0.7rem",
                    "color": _col,
                    "marginLeft": "8px",
                    "transform": "rotate(180deg)" if abierto else "rotate(0deg)",
                    "transition": "transform 0.2s ease",
                }
                return abierto, arrow_style
        make_toggle(id_base, color)
