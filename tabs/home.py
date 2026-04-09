"""
tabs/home.py
------------
Páginas Home y Equipo.
"""

import dash_bootstrap_components as dbc
from dash import html, Input, Output, callback, ctx, no_update




def _pill_btn_link(icon, label, href):
    return html.A(
        href=href, target="_blank",
        children=[html.I(className=icon, style={"marginRight": "7px"}), label],
        style={
            "display": "inline-flex", "alignItems": "center",
            "padding": "9px 20px", "borderRadius": "20px",
            "border": "1px solid rgba(74,111,165,0.45)",
            "background": "rgba(74,111,165,0.15)",
            "color": "#7fb3d3", "fontSize": "0.82rem",
            "fontWeight": "600", "textDecoration": "none",
            "transition": "all 0.2s ease", "cursor": "pointer",
            "letterSpacing": "0.2px",
        },
        className="home-pill-btn",
    )


def _pill_btn_internal(icon, label, btn_id):
    return html.Div(
        id=btn_id, n_clicks=0,
        children=[html.I(className=icon, style={"marginRight": "7px"}), label],
        style={
            "display": "inline-flex", "alignItems": "center",
            "padding": "9px 20px", "borderRadius": "20px",
            "border": "1px solid rgba(74,111,165,0.45)",
            "background": "rgba(74,111,165,0.15)",
            "color": "#7fb3d3", "fontSize": "0.82rem",
            "fontWeight": "600", "cursor": "pointer",
            "transition": "all 0.2s ease", "letterSpacing": "0.2px",
        },
        className="home-pill-btn",
    )


def _primary_btn(icon, label, btn_id):
    return html.Div(
        id=btn_id, n_clicks=0,
        children=[html.I(className=icon, style={"marginRight": "8px"}), label],
        style={
            "display": "inline-flex", "alignItems": "center",
            "padding": "12px 28px", "borderRadius": "8px",
            "background": "#2e86c1",
            "boxShadow": "0 4px 18px rgba(46,134,193,0.45)",
            "color": "#fff", "fontSize": "0.88rem",
            "fontWeight": "700", "cursor": "pointer",
            "transition": "all 0.22s ease", "letterSpacing": "0.3px",
            "border": "none",
        },
        className="home-primary-btn",
    )


def _float_card(icon_class, fc_color, value, label):
    icon_colors = {
        "fc-blue":   ("rgba(46,134,193,0.2)",  "#4a9fe0"),
        "fc-red":    ("rgba(192,57,43,0.2)",    "#e74c3c"),
        "fc-amber":  ("rgba(243,156,18,0.2)",   "#f39c12"),
        "fc-green":  ("rgba(39,174,96,0.2)",    "#27ae60"),
        "fc-purple": ("rgba(93,173,226,0.2)",   "#5dade2"),
    }
    bg, color = icon_colors.get(fc_color, ("rgba(46,134,193,0.2)", "#4a9fe0"))

    return html.Div([
        html.Div(
            html.I(className=icon_class),
            style={
                "width": "40px", "height": "40px", "borderRadius": "10px",
                "background": bg, "color": color,
                "display": "flex", "alignItems": "center",
                "justifyContent": "center", "fontSize": "1.1rem", "flexShrink": "0",
            }
        ),
        html.Div([
            html.Div(value, style={
                "fontSize": "1.05rem", "fontWeight": "700",
                "color": "#fff", "lineHeight": "1",
            }),
            html.Div(label, style={
                "fontSize": "0.70rem", "color": "#8a9bbf", "marginTop": "3px",
            }),
        ]),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "12px",
        "background": "rgba(255,255,255,0.06)",
        "backdropFilter": "blur(12px)", "WebkitBackdropFilter": "blur(12px)",
        "border": "1px solid rgba(255,255,255,0.10)",
        "borderRadius": "14px", "padding": "13px 16px",
        "minWidth": "170px",
        "boxShadow": "0 8px 32px rgba(0,0,0,0.28)",
    })


def _mini_chart():
    return html.Div([
        html.Div("Distribución de clases", style={
            "fontSize": "0.68rem", "color": "#8a9bbf",
            "fontWeight": "600", "marginBottom": "10px",
            "textTransform": "uppercase", "letterSpacing": "0.5px",
        }),
        html.Div([
            html.Div([
                html.Div(style={
                    "width": "28px", "height": "52px", "borderRadius": "4px 4px 0 0",
                    "background": "linear-gradient(180deg,#4a9fe0,#2e6fa5)",
                }),
                html.Div("99.8%", style={"fontSize": "0.64rem", "fontWeight": "700",
                                         "color": "#c8d4e8", "marginTop": "4px"}),
                html.Div("Legítimas", style={"fontSize": "0.58rem", "color": "#5a7abf"}),
            ], style={"display": "flex", "flexDirection": "column",
                      "alignItems": "center", "gap": "3px"}),
            html.Div([
                html.Div(style={
                    "width": "28px", "height": "6px", "borderRadius": "4px 4px 0 0",
                    "background": "linear-gradient(180deg,#e74c3c,#c0392b)",
                }),
                html.Div("0.17%", style={"fontSize": "0.64rem", "fontWeight": "700",
                                         "color": "#c8d4e8", "marginTop": "4px"}),
                html.Div("Fraudes", style={"fontSize": "0.58rem", "color": "#5a7abf"}),
            ], style={"display": "flex", "flexDirection": "column",
                      "alignItems": "center", "gap": "3px"}),
        ], style={"display": "flex", "alignItems": "flex-end",
                  "gap": "14px", "height": "80px"}),
    ], style={
        "background": "rgba(255,255,255,0.06)",
        "backdropFilter": "blur(12px)", "WebkitBackdropFilter": "blur(12px)",
        "border": "1px solid rgba(255,255,255,0.10)",
        "borderRadius": "14px", "padding": "14px 18px",
        "boxShadow": "0 8px 32px rgba(0,0,0,0.28)",
    })


def _orbital_ring():
    return html.Div(
        html.Div(style={
            "position": "absolute",
            "top": "50%", "left": "50%",
            "transform": "translate(-50%, -52%) rotate(-10deg)",
            "width": "80%", "height": "72%",
            "borderRadius": "50%",
            "border": "1px dashed rgba(74,111,165,0.20)",
            "pointerEvents": "none",
        }),
        style={
            "position": "absolute", "inset": "0",
            "pointerEvents": "none", "zIndex": "0",
        },
    )


def _card_wrapper(child, pos, anim_class):
    style = {"position": "absolute", "zIndex": "2"}
    style.update(pos)
    return html.Div(child, className=anim_class, style=style)



def layout_home():
    return html.Div([

        html.Div([

            html.Div(style={
                "position": "absolute", "inset": "0",
                "pointerEvents": "none", "zIndex": "0",
                "backgroundImage": (
                    "radial-gradient(circle at 78% 17%, rgba(74,111,165,0.18) 0%, transparent 40%),"
                    "radial-gradient(circle at 93% 71%, rgba(46,134,193,0.10) 0%, transparent 35%),"
                    "radial-gradient(circle at 14% 85%, rgba(30,52,96,0.35) 0%, transparent 40%),"
                    "radial-gradient(circle at 4%  14%, rgba(74,111,165,0.09) 0%, transparent 30%)"
                ),
            }),

            html.Div([

                html.Div([
                    html.I(className="bi bi-shield-lock-fill",
                           style={"marginRight": "7px"}),
                    "Análisis de Fraude · ULB Dataset",
                ], style={
                    "display": "inline-flex", "alignItems": "center",
                    "background": "rgba(74,111,165,0.2)",
                    "border": "1px solid rgba(74,111,165,0.4)",
                    "color": "#7fb3d3", "fontSize": "0.75rem",
                    "fontWeight": "600", "letterSpacing": "1px",
                    "textTransform": "uppercase",
                    "padding": "5px 14px", "borderRadius": "20px",
                    "marginBottom": "28px",
                }),

                html.H1([
                    "Detección de", html.Br(),
                    html.Span("Fraude", style={
                        "color": "#4a9fe0",
                        "borderBottom": "3px solid rgba(74,159,224,0.4)",
                        "paddingBottom": "2px",
                    }),
                    " con", html.Br(),
                    "Tarjeta de Crédito",
                ], style={
                    "fontSize": "clamp(2.2rem, 4vw, 3.2rem)",
                    "fontWeight": "800", "color": "#ffffff",
                    "lineHeight": "1.12", "margin": "0 0 20px 0",
                    "letterSpacing": "-0.5px",
                }),

                html.P(
                    "284K transacciones reales · EDA completo · Regresión Logística",
                    style={
                        "fontSize": "0.95rem", "color": "#8a9bbf",
                        "margin": "0 0 36px 0", "lineHeight": "1.6",
                        "letterSpacing": "0.2px",
                    }
                ),

                html.Div([
                    _primary_btn("bi bi-grid-fill", "Ver Dashboard", "home-btn-dashboard-real"),
                    html.Div([
                        _pill_btn_link("bi bi-github",   "GitHub",
                                       "https://github.com/alemengo76/FraudDetection"),
                        _pill_btn_link("bi bi-database", "Kaggle",
                                       "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"),
                        _pill_btn_internal("bi bi-people-fill", "Equipo", "home-btn-about-real"),
                    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
                ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                          "alignItems": "center"}),

            ], className="home-hero-left", style={
                "position": "relative", "zIndex": "1",
                "flex": "1", "maxWidth": "520px",
            }),

            html.Div([

                _orbital_ring(),

                _card_wrapper(
                    _float_card("bi bi-arrow-left-right", "fc-blue",
                                "283.726", "Transacciones totales"),
                    {"top": "3%", "left": "50%", "transform": "translateX(-50%)"},
                    "fc1",
                ),
                _card_wrapper(
                    _float_card("bi bi-exclamation-triangle-fill", "fc-red",
                                "473", "Fraudes detectados"),
                    {"top": "22%", "right": "3%"},
                    "fc2",
                ),
                _card_wrapper(
                    _mini_chart(),
                    {"bottom": "22%", "right": "3%"},
                    "fc3",
                ),
                _card_wrapper(
                    _float_card("bi bi-cpu-fill", "fc-purple",
                                "V14", "Mayor discriminante"),
                    {"bottom": "3%", "left": "50%", "transform": "translateX(-50%)"},
                    "fc4",
                ),
                _card_wrapper(
                    _float_card("bi bi-diagram-3-fill", "fc-green",
                                "28 PCA", "Variables anónimas"),
                    {"bottom": "22%", "left": "3%"},
                    "fc5",
                ),
                _card_wrapper(
                    _float_card("bi bi-percent", "fc-amber",
                                "0.17%", "Tasa de fraude"),
                    {"top": "22%", "left": "3%"},
                    "fc1",
                ),

            ], className="home-hero-right", style={
                "position": "relative", "zIndex": "1", "flex": "1",
                "height": "calc(100vh - 100px)",
                "minHeight": "500px", "minWidth": "420px",
            }),

        ], style={
            "position": "relative", "display": "flex", "alignItems": "center",
            "height": "calc(100vh - 100px)",
            "background": "linear-gradient(135deg, #080e1c 0%, #0f1e3d 40%, #0d2a4a 70%, #102238 100%)",
            "overflow": "hidden", "padding": "0 64px", "gap": "40px",
        }),

    ])




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




def _member_card_dark(name, role, programs, university, github, linkedin):
    """Tarjeta dark glassmorphism con animaciones de entrada y hover en pills."""

    accent_bar = html.Div(style={
        "width": "4px", "flexShrink": "0",
        "background": "linear-gradient(180deg, #4a9fe0, #4a6fa5)",
        "borderRadius": "16px 0 0 16px",
    })

    avatar = html.Div(
        html.I(className="bi bi-person-circle",
               style={"fontSize": "2.6rem", "color": "#4a9fe0", "opacity": "0.85"}),
        style={"marginBottom": "14px"},
    )

    nombre = html.Div(name, style={
        "fontSize": "1rem", "fontWeight": "700",
        "color": "#ffffff", "margin": "0 0 5px 0",
    })

    badge = html.Span(role, style={
        "fontSize": "0.65rem", "fontWeight": "600",
        "color": "#5a7abf", "letterSpacing": "1px",
        "textTransform": "uppercase", "display": "block",
        "marginBottom": "18px",
    })

    def _info(label, value):
        return html.Div([
            html.P(label, style={
                "fontSize": "0.65rem", "fontWeight": "700", "color": "#5a7abf",
                "letterSpacing": "1px", "textTransform": "uppercase", "margin": "0",
            }),
            html.P(value, style={
                "fontSize": "0.85rem", "color": "#c8d4e8", "margin": "3px 0 10px 0",
            }),
        ])

    info_section = html.Div([
        _info("Programa", " · ".join(programs)),
        _info("Institución", university),
    ], style={"marginBottom": "18px"})

    # Pills con clase CSS para hover animado
    link_pills = []
    if github:
        link_pills.append(html.A(
            [html.I(className="bi bi-github", style={"marginRight": "6px"}), "GitHub"],
            href=github, target="_blank",
            className="equipo-pill",
        ))
    if linkedin:
        link_pills.append(html.A(
            [html.I(className="bi bi-linkedin", style={"marginRight": "6px"}), "LinkedIn"],
            href=linkedin, target="_blank",
            className="equipo-pill",
        ))

    links_row = html.Div(link_pills, style={
        "display": "flex", "gap": "10px", "flexWrap": "wrap",
        "paddingTop": "16px",
        "borderTop": "1px solid rgba(255,255,255,0.08)",
    })

    body = html.Div([
        avatar, nombre, badge, info_section, links_row,
    ], style={"padding": "28px 24px", "flex": "1"})

    # La clase equipo-card-new tiene el hover definido en CSS
    return html.Div([accent_bar, body], className="equipo-card-new")


def layout_about():
    return html.Div([

        # Fondo radial decorativo
        html.Div(style={
            "position": "absolute", "inset": "0",
            "pointerEvents": "none", "zIndex": "0",
            "backgroundImage": (
                "radial-gradient(circle at 20% 30%, rgba(74,111,165,0.18) 0%, transparent 45%),"
                "radial-gradient(circle at 80% 70%, rgba(46,134,193,0.12) 0%, transparent 40%),"
                "radial-gradient(circle at 50% 10%, rgba(30,52,96,0.25) 0%, transparent 35%)"
            ),
        }),

        html.Div([

            # Eyebrow + título con animación fadeIn
            html.Div([
                html.Div([
                    html.I(className="bi bi-people-fill", style={"marginRight": "7px"}),
                    "Nuestro Equipo",
                ], style={
                    "display": "inline-flex", "alignItems": "center",
                    "background": "rgba(74,111,165,0.2)",
                    "border": "1px solid rgba(74,111,165,0.4)",
                    "color": "#7fb3d3", "fontSize": "0.75rem",
                    "fontWeight": "600", "letterSpacing": "1px",
                    "textTransform": "uppercase",
                    "padding": "5px 14px", "borderRadius": "20px",
                    "marginBottom": "18px",
                }),

                html.H2([
                    "Detrás del ",
                    html.Span("Proyecto", style={
                        "color": "#4a9fe0",
                        "borderBottom": "3px solid rgba(74,159,224,0.4)",
                        "paddingBottom": "2px",
                    }),
                ], style={
                    "fontSize": "clamp(1.8rem, 3vw, 2.6rem)",
                    "fontWeight": "800", "color": "#ffffff",
                    "lineHeight": "1.15", "margin": "0 0 10px 0",
                    "letterSpacing": "-0.3px",
                }),

                html.P(
                    "Universidad del Norte · Ciencia de Datos · 2026",
                    style={
                        "fontSize": "0.88rem", "color": "#8a9bbf",
                        "margin": "0", "letterSpacing": "0.3px",
                    }
                ),

            # Clase CSS con animación slideInLeft
            ], className="equipo-header-anim", style={"textAlign": "center", "marginBottom": "48px"}),

            # Cards con animación slideInUp escalonada
            html.Div([
                html.Div(
                    _member_card_dark(
                        name="Alejandra Meneses Gómez",
                        role="Estudiante",
                        programs=["Ciencia de Datos", "Matemáticas"],
                        university="Universidad del Norte",
                        github="https://github.com/alemengo76",
                        linkedin="https://www.linkedin.com/in/alejandra-meneses-gómez-aaa97b3b7/",
                    ),
                    className="equipo-card-anim-1",  # delay 0.1s
                ),
                html.Div(
                    _member_card_dark(
                        name="Mariangel Yepes Negrete",
                        role="Estudiante",
                        programs=["Ciencia de Datos"],
                        university="Universidad del Norte",
                        github="https://github.com/Mary-Yepes",
                        linkedin=None,
                    ),
                    className="equipo-card-anim-2",  # delay 0.25s
                ),
            ], style={
                "display": "flex", "gap": "28px",
                "justifyContent": "center", "flexWrap": "wrap",
            }),

        ], style={"position": "relative", "zIndex": "1", "width": "100%"}),

    ],  className="tab-fade-in",
        style={
        "position": "relative",
        "minHeight": "calc(100vh - 100px)",
        "background": "linear-gradient(135deg, #080e1c 0%, #0f1e3d 40%, #0d2a4a 70%, #102238 100%)",
        "display": "flex", "flexDirection": "column",
        "alignItems": "center", "justifyContent": "center",
        "padding": "48px 64px", "overflow": "hidden",
    })