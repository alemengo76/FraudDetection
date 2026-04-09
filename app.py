"""
app.py
------
Punto de entrada del dashboard de detección de fraude financiero.
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import time

from tabs import (
    introduccion, contexto, problema, objetivos,
    marco_teorico, metodologia, dataset, eda,
    resultados, prediccion, limitaciones, conclusiones,
)
import tabs.home as home

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"],
    suppress_callback_exceptions=True,
    title="Detección de Fraude Financiero",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

introduccion.register_callbacks(app)
prediccion.register_callbacks(app)
contexto.register_callbacks(app)
marco_teorico.register_callbacks(app)
metodologia.register_callbacks(app)
dataset.register_callbacks(app)
eda.register_callbacks(app)
conclusiones.register_callbacks(app)

TABS_DASHBOARD = [
    ("introduccion",  "Introducción"),
    ("contexto",      "Contexto"),
    ("problema",      "Problema"),
    ("objetivos",     "Objetivos"),
    ("marco-teorico", "Marco teórico"),
    ("metodologia",   "Metodología"),
    ("dataset",       "Dataset"),
    ("eda",           "EDA"),
    ("limitaciones",  "Limitaciones"),
    ("conclusiones",  "Conclusiones"),
]

app.layout = html.Div([

    html.Div([
        html.Div([
            html.Div("Detección de Fraude Financiero", className="navbar-brand-text"),
            html.Div("Credit Card Fraud Detection · ULB Dataset · Regresión Logística",
                     className="navbar-subtitle"),
        ]),
        html.Div([
            html.Span("Inicio", id="nav-home", className="nav-pill nav-pill-active", n_clicks=0),
            html.Span("Dashboard", id="nav-dashboard", className="nav-pill", n_clicks=0),
            html.Span("Equipo", id="nav-about", className="nav-pill", n_clicks=0),
        ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
    ], className="navbar-custom",
       style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),

    html.Div(
        dcc.Tabs(
            id="main-tabs",
            value="introduccion",
            children=[
                dcc.Tab(label=label, value=tab_id,
                        className="custom-tab",
                        selected_className="custom-tab--selected")
                for tab_id, label in TABS_DASHBOARD
            ],
            className="custom-tabs",
            colors={"border": "transparent", "primary": "#2e86c1", "background": "#f4f6f9"},
        ),
        id="subtabs-bar",
        style={"display": "none", "background": "#ffffff",
               "borderBottom": "1px solid #e2e8f0", "paddingLeft": "20px"},
    ),

    dcc.Store(id="current-section", data="home"),


    html.Div(
        id="home-btn-dashboard",
        n_clicks=0,
        style={"display": "none"},
    ),
    html.Div(
        id="home-btn-about",
        n_clicks=0,
        style={"display": "none"},
    ),

    html.Div(home.layout_home(), id="page-content"),

    html.Div(
        "Proyecto de análisis de fraude financiero · Python · Dash · Scikit-learn",
        className="footer",
    ),
])


@app.callback(
    Output("page-content", "children"),
    Output("subtabs-bar", "style"),
    Output("nav-home", "className"),
    Output("nav-dashboard", "className"),
    Output("nav-about", "className"),
    Output("current-section", "data"),
    Input("nav-home", "n_clicks"),
    Input("nav-dashboard", "n_clicks"),
    Input("nav-about", "n_clicks"),
    Input("home-btn-dashboard", "n_clicks"),
    Input("home-btn-about", "n_clicks"),
    prevent_initial_call=True,   
)

def render_section(h, d, a, hb, ha):
    from dash import ctx

    triggered = ctx.triggered_id

    subtabs_hidden = {"display": "none", "background": "#ffffff",
                      "borderBottom": "1px solid #e2e8f0", "paddingLeft": "20px"}
    subtabs_visible = {"display": "block", "background": "#ffffff",
                       "borderBottom": "1px solid #e2e8f0", "paddingLeft": "20px"}

    base = "nav-pill"
    active = "nav-pill nav-pill-active"

    if triggered in ["nav-dashboard", "home-btn-dashboard"]:
        return render_dashboard_tab("introduccion"), subtabs_visible, base, active, base, "dashboard"

    if triggered in ["nav-about", "home-btn-about"]:
        return html.Div(home.layout_about(), key=f"about-{time.time()}"), subtabs_hidden, base, base, active, "about"

    return html.Div(home.layout_home(), key=f"home-{time.time()}"), subtabs_hidden, active, base, base, "home"


@app.callback(
    Output("page-content", "children", allow_duplicate=True),
    Input("main-tabs", "value"),
    Input("current-section", "data"),
    prevent_initial_call=True,
)
def render_dashboard_subtab(tab, section):
    if section != "dashboard":
        from dash import no_update
        return no_update
    return render_dashboard_tab(tab)


def render_dashboard_tab(tab: str):
    routing = {
        "introduccion": introduccion.layout,
        "contexto": contexto.layout,
        "problema": problema.layout,
        "objetivos": objetivos.layout,
        "marco-teorico": marco_teorico.layout,
        "metodologia": metodologia.layout,
        "dataset": dataset.layout,
        "eda": eda.layout,
        "resultados": resultados.layout,
        "prediccion": prediccion.layout,
        "limitaciones": limitaciones.layout,
        "conclusiones": conclusiones.layout,
    }
    fn = routing.get(tab)
    content = fn() if fn else html.Div("Pestaña no encontrada.", style={"padding": "40px"})
    
    # Envolver en un div con key único fuerza a React a recrear el nodo
    return html.Div(content, key=f"{tab}-{time.time()}")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)