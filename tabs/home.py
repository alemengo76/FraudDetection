"""
tabs/home.py
------------
Páginas Home y Equipo.
"""

import dash_bootstrap_components as dbc
from dash import html, Input, Output, callback, ctx, no_update


def _btn_style():
    return {
        "flex": "1 1 0",
        "minWidth": "0",
        "padding": "16px 10px",
        "borderRadius": "12px",
        "background": "#fff",
        "border": "1px solid #e2e8f0",
        "cursor": "pointer",
        "textDecoration": "none",
        "display": "block",
    }

def _icon(icon_class: str):
    return html.I(className=icon_class, style={
        "fontSize": "20px",
        "color": "#2e86c1",
        "marginBottom": "8px",
        "display": "block",
    })


def _btn_inner(icon: str, label: str, sublabel: str):
    return html.Div([
        _icon(icon),
        html.Div(label, style={
            "fontWeight": "600",
            "fontSize": "0.80rem",
            "color": "#1a2540",
            "marginBottom": "2px",
        }),
        html.Div(sublabel, style={
            "fontSize": "0.68rem",
            "color": "#718096",
            "whiteSpace": "nowrap",
        }),
    ], style={"textAlign": "center"})


def _action_btn_link(icon, label, sublabel, href):
    return html.A(
        href=href,
        target="_blank",
        children=_btn_inner(icon, label, sublabel),
        style=_btn_style(),
        className="action-btn", 
    )


def _action_btn_internal(icon, label, sublabel, btn_id):
    return html.Div(
        id=btn_id,
        n_clicks=0,
        children=_btn_inner(icon, label, sublabel),
        style=_btn_style(),
        className="action-btn", 
    )


def layout_home():
    return html.Div([

        dbc.Row([
            dbc.Col(
                html.Div(
                    html.Img(
                        src="/assets/imagen_home.jpeg",
                        style={
                            "width": "100%",
                            "height": "420px",
                            "objectFit": "cover",
                            "borderRadius": "16px",
                            "boxShadow": "0 8px 32px rgba(26,37,64,0.22)",
                            "display": "block",
                        },
                    ),
                    style={"padding": "8px 0"},
                ),
                md=5,
            ),

            dbc.Col(
                html.Div([

                    html.Div("DETECCIÓN DE FRAUDE FINANCIERO", style={
                        "fontSize": "1.45rem",
                        "fontWeight": "800",
                        "color": "#1a2540",
                        "lineHeight": "1.2",
                        "letterSpacing": "0.3px",
                        "marginBottom": "6px",
                    }),

                    html.Div(style={
                        "width": "48px", "height": "3px",
                        "background": "linear-gradient(90deg, #2e86c1, #5dade2)",
                        "borderRadius": "2px", "marginBottom": "20px",
                    }),

                    html.P(
                        "El fraude con tarjetas de crédito es una gran amenaza para el sistema financiero en todo el mundo. Entonces, es importante detectarlo a tiempo para proteger a los clientes y evitar pérdidas que pueden llegar a ser millonarias.",
                        className="section-body", style={"marginBottom": "14px"},
                    ),

                    html.P(
                        "En este proyecto se usa el dataset Credit Card Fraud Detection del ULB Machine Learning Group. Dicho dataset tiene 283.726 transacciones de titulares europeos que se exploraron por medio de un análisis exploratorio de datos, en el que se identificaron patrones y variables que separan transacciones fraudulentas de no fraudulentas.",
                        className="section-body", style={"marginBottom": "14px"},
                    ),

                    html.P(
                        "El análisis incluye pruebas no paramétricas, tamaño de efecto, correlaciones, VIF y, próximamente, un modelo de regresión logística para predicción interactiva.",
                        className="section-body", style={"marginBottom": "32px"},
                    ),

                    html.Div([
                        _action_btn_internal("bi bi-grid",   "Aplicación", "Ver dashboard",     "home-btn-dashboard-real"),
                        _action_btn_link("bi bi-github",     "Código",     "Ver en GitHub",     "https://github.com/alemengo76"),
                        _action_btn_link("bi bi-database",   "Dataset",    "Ver en Kaggle",     "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"),
                        _action_btn_internal("bi bi-people", "Equipo",     "Conocer el equipo", "home-btn-about-real"),
                    ], style={
                        "display": "flex",
                        "gap": "10px",
                        "width": "100%",
                    }),

                ], style={"padding": "16px 8px"}),
                md=7,
            ),
        ], align="center"),

    ], className="tab-content-wrapper")


@callback(
    Output("home-btn-dashboard", "n_clicks"),
    Output("home-btn-about",     "n_clicks"),
    Input("home-btn-dashboard-real", "n_clicks"),
    Input("home-btn-about-real",     "n_clicks"),
    prevent_initial_call=True,
)
def relay_home_buttons(nd, na):
    triggered = ctx.triggered_id
    if triggered == "home-btn-dashboard-real":
        return (nd or 0), no_update
    if triggered == "home-btn-about-real":
        return no_update, (na or 0)
    return no_update, no_update


def layout_about():
    return html.Div([

        html.H5("Equipo", className="section-title",
                style={"marginTop": 0, "marginBottom": "40px"}),

        html.Div([
            _member_card(
                name="Alejandra Meneses Gómez",
                role="Estudiante",
                programs=["Ciencia de Datos", "Matemáticas"],
                university="Universidad del Norte",
                github="https://github.com/alemengo76",
                linkedin="https://www.linkedin.com/in/alejandra-meneses-gómez-aaa97b3b7/",
            ),
            _member_card(
                name="Mariangel Yepes Negrete",
                role="Estudiante",
                programs=["Ciencia de Datos"],
                university="Universidad del Norte",
                github="https://github.com/Mary-Yepes",
                linkedin=None,
            ),
        ], style={
            "display": "flex",
            "gap": "32px",
            "justifyContent": "center",
            "flexWrap": "wrap",
        }),

    ], className="tab-content-wrapper")


def _member_card(name, role, programs, university, github, linkedin):

    # Acento lateral izquierdo (igual que .section-title del dashboard)
    # Encabezado con nombre y rol
    header = html.Div([
        html.Div(name, style={
            "fontWeight": "800",
            "fontSize": "1.05rem",
            "color": "#1a2540",
            "letterSpacing": "0.1px",
            "marginBottom": "4px",
        }),
        html.Div(role, style={
            "fontWeight": "500",
            "fontSize": "0.82rem",
            "color": "#2e86c1",
            "textTransform": "uppercase",
            "letterSpacing": "0.8px",
        }),
    ], style={
        "borderLeft": "4px solid #2e86c1",
        "paddingLeft": "14px",
        "marginBottom": "20px",
        "textAlign": "left",
    })

    # Fila: programas académicos
    programs_row = html.Div([
        html.Div("Programa", style={
            "fontSize": "0.70rem",
            "fontWeight": "700",
            "color": "#a0aec0",
            "textTransform": "uppercase",
            "letterSpacing": "0.6px",
            "marginBottom": "6px",
        }),
        html.Div(
            " · ".join(programs),
            style={
                "fontSize": "0.85rem",
                "color": "#2d3748",
                "fontWeight": "500",
            }
        ),
    ], style={"marginBottom": "14px", "textAlign": "left"})

    # universidad
    university_row = html.Div([
        html.Div("Institución", style={
            "fontSize": "0.70rem",
            "fontWeight": "700",
            "color": "#a0aec0",
            "textTransform": "uppercase",
            "letterSpacing": "0.6px",
            "marginBottom": "6px",
        }),
        html.Div([
            university,
        ], style={
            "fontSize": "0.85rem",
            "color": "#2d3748",
            "fontWeight": "500",
            "display": "flex",
            "alignItems": "center",
        }),
    ], style={"marginBottom": "24px", "textAlign": "left"})

    # Separador
    divider = html.Hr(style={
        "border": "none",
        "borderTop": "1px solid #e2e8f0",
        "margin": "0 0 18px 0",
    })

    # Links
    link_items = []
    if github:
        link_items.append(html.A(
            [html.I(className="bi bi-github", style={"marginRight": "6px"}), "GitHub"],
            href=github,
            target="_blank",
            className="action-btn",
            style={
                "display": "inline-flex",
                "alignItems": "center",
                "padding": "6px 18px",
                "borderRadius": "6px",
                "border": "1px solid #cbd5e0",
                "fontSize": "0.76rem",
                "fontWeight": "600",
                "color": "#1a2540",
                "textDecoration": "none",
                "background": "#f8fafc",
            },
        ))
    if linkedin:
        link_items.append(html.A(
            [html.I(className="bi bi-linkedin",
                    style={"marginRight": "6px", "color": "#0a66c2"}), "LinkedIn"],
            href=linkedin,
            target="_blank",
            className="action-btn",
            style={
                "display": "inline-flex",
                "alignItems": "center",
                "padding": "6px 18px",
                "borderRadius": "6px",
                "border": "1px solid #cbd5e0",
                "fontSize": "0.76rem",
                "fontWeight": "600",
                "color": "#1a2540",
                "textDecoration": "none",
                "background": "#f8fafc",
            },
        ))

    links = html.Div(link_items, style={
        "display": "flex",
        "gap": "8px",
        "justifyContent": "flex-start",
    })

    return html.Div([
        header,
        programs_row,
        university_row,
        divider,
        links,
    ], style={
        "padding": "28px",
        "background": "#ffffff",
        "borderRadius": "12px",
        "boxShadow": "0 4px 16px rgba(26,37,64,0.09)",
        "borderTop": "3px solid #2e86c1",
        "width": "320px",
        "flexShrink": "0",
    })
    
