"""
tabs/prediccion.py
------------------
Pestaña "Predicción" — réplica del predictor interactivo de mod_modelo.R.

Funcionalidades:
  · Formulario con sliders para V1–V28, Amount y Time
  · Botones "Ejemplo legítima" / "Ejemplo fraude" para carga rápida
  · Predicción en tiempo real al pulsar "Analizar transacción"
  · Gauge SVG (velocímetro de probabilidad)
  · Veredicto color-coded: Legítima / Sospechosa / Fraude
  · Pills de métricas del modelo
  · Historial de transacciones analizadas en la sesión
  · Tarjeta de crédito animada (flip 3D al analizar)
  · Perfil comparativo: esta transacción vs. promedios legítima/fraude

CARGA DEL MODELO:
  Se carga una sola vez con joblib y se cachea con @lru_cache.
  Si el archivo no existe, se usa un modelo dummy que responde aleatoriamente
  con un aviso visual en la interfaz.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELO_PATH = Path(r"C:\Users\Alejandra\Documents\Modelos_Optimizados\optuna_XGBoost_ADASYN.joblib")

# ─── Paleta ───────────────────────────────────────────────────────────────────
C_DARK   = "#1a2540"
C_BLUE   = "#2e86c1"
C_LIGHT  = "#5dade2"
C_GREEN  = "#1e8449"
C_AMBER  = "#d4ac0d"
C_DANGER = "#c0392b"

# ─── Ejemplos del dataset ULB ─────────────────────────────────────────────────
EJEMPLO_LEGITIMA = {
    "amount": 9.0, "time": 10.0,
    "v": [-1.3598,-0.0728, 2.5363, 1.3781,-0.3383, 0.4624, 0.2396, 0.0987,
           0.3638, 0.0908,-0.5516,-0.6178,-0.9913,-0.3111, 1.4681,-0.4704,
           0.2076, 0.0258, 0.4033, 0.2514,-0.0183, 0.2778,-0.1105, 0.0669,
           0.1285,-0.1891, 0.1336,-0.0211],
}
EJEMPLO_FRAUDE = {
    "amount": 149.62, "time": 2.0,
    "v": [-1.3598,-0.0728, 2.5363, 1.3781,-0.3383, 0.4624, 0.2396, 0.0987,
          -2.3122, 1.9519,-1.6096, 3.9979,-0.5223, 2.2195,-3.2997,-0.7298,
          -0.0927,-0.1701,-0.4534,-1.4183, 0.8970,-0.0248, 0.0374, 0.1337,
          -0.1366, 0.0135, 0.0193, 0.0000],
}

# Promedios de referencia (para perfil comparativo)
PERFIL_REF = pd.DataFrame({
    "Variable": ["V1","V3","V4","V7","V10","V12","V14","Amount"],
    "Legitima": [ 0.01, 0.02,-0.01, 0.00, 0.01,  0.00, 0.01,  88.35],
    "Fraude":   [-4.77,-3.04, 4.52, 5.14,-5.33, -6.81, 5.89, 122.21],
})

FEATURE_COLS = ([f"V{i}" for i in range(1, 29)] + ["Amount", "Time"])


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _cargar_modelo():
    """
    Carga el pipeline .joblib una sola vez.
    Devuelve (pipeline, True) si existe, o (None, False) para usar el dummy.

    El pipeline debe tener un método .predict_proba() que reciba un DataFrame
    con columnas V1–V28, Amount, Time.
    """
    if MODELO_PATH.exists():
        try:
            import joblib
            pipeline = joblib.load(MODELO_PATH)
            return pipeline, True
        except Exception as e:
            print(f"[prediccion.py] Error al cargar el modelo: {e}")
    return None, False


def _predecir(values_v: list[float], amount: float, time_val: float) -> float:
    """
    Retorna la probabilidad de fraude (0-1).
    Si el modelo no está disponible, usa regla heurística demo.
    """
    pipeline, ok = _cargar_modelo()
    row_dict = {f"V{i+1}": values_v[i] for i in range(28)}
    row_dict["Amount"] = amount
    row_dict["Time"]   = time_val
    df_input = pd.DataFrame([row_dict])[FEATURE_COLS]

    if ok:
        prob = float(pipeline.predict_proba(df_input)[0, 1])
    else:
        # Heurística demo: V14 y V4 son los más importantes
        v14, v4 = values_v[13], values_v[3]
        score = 0.5 + 0.18 * v14 - 0.15 * v4 + 0.05 * (amount / 500)
        prob  = float(np.clip(1 / (1 + np.exp(-score * 1.5)), 0.01, 0.99))
    return round(prob, 4)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENTES DE INTERFAZ
# ══════════════════════════════════════════════════════════════════════════════

def _slider_block(n: int) -> html.Div:
    """Slider para la variable PCA Vn."""
    slider_id = {"type": "v-slider", "index": n}
    label_id  = {"type": "v-label",  "index": n}
    return html.Div([
        html.Div([
            html.Span(f"V{n}", style={
                "fontSize": "0.78rem", "fontWeight": "700",
                "color": C_DARK, "minWidth": "28px",
            }),
            html.Span(id=label_id, style={
                "fontSize": "0.78rem", "color": C_BLUE,
                "fontWeight": "600", "marginLeft": "auto",
                "fontFamily": "'Courier New', monospace",
            }),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "marginBottom": "2px"}),
        dcc.Slider(
            id=slider_id, min=-5, max=5, step=0.1, value=0,
            marks=None,
            tooltip={"always_visible": False},
            className="mb-1",
        ),
    ], style={"marginBottom": "10px"})


def _pca_tab_group(from_n: int, to_n: int) -> html.Div:
    return html.Div(
        [_slider_block(n) for n in range(from_n, to_n + 1)],
        style={"maxHeight": "320px", "overflowY": "auto",
               "paddingRight": "4px",
               "scrollbarWidth": "thin"},
    )


def _gauge_svg(prob: float | None) -> html.Div:
    """Velocímetro SVG, réplica del de mod_modelo.R."""
    if prob is None:
        return html.Div(
            html.Div("Analiza una transacción para ver el resultado",
                     style={"color": "#9aa0a8", "fontSize": "0.85rem",
                            "textAlign": "center", "padding": "32px 0"}),
        )

    pct = round(prob * 100)
    # Color de la aguja
    if prob < 0.30:
        nc = C_GREEN
    elif prob < 0.65:
        nc = C_AMBER
    else:
        nc = C_DANGER

    # Ángulo: 0% → -90°, 100% → 90°
    needle_deg = -90 + 180 * prob
    pct_label  = f"{pct}%"

    svg = html.Svg([
        # Arco fondo
        html.Defs(html.LinearGradient([
            html.Stop(offset="0%",   style={"stopColor": "#27ae60"}),
            html.Stop(offset="50%",  style={"stopColor": "#f39c12"}),
            html.Stop(offset="100%", style={"stopColor": "#e74c3c"}),
        ], id="gauge-grad", x1="0%", y1="0%", x2="100%", y2="0%")),
        html.Path(d="M 30 140 A 120 120 0 0 1 270 140",
                  fill="none", stroke="#e9ecef", strokeWidth="18",
                  strokeLinecap="round"),
        html.Path(d="M 30 140 A 120 120 0 0 1 270 140",
                  fill="none", stroke="url(#gauge-grad)", strokeWidth="12",
                  strokeLinecap="round", opacity="0.7"),
        # Aguja
        html.Line(x1="150", y1="140", x2="150", y2="48",
                  stroke=nc, strokeWidth="3", strokeLinecap="round",
                  style={
                      "transformOrigin": "150px 140px",
                      "transform": f"rotate({needle_deg:.1f}deg)",
                      "transition": "transform 1.2s cubic-bezier(0.34,1.56,0.64,1)",
                  }),
        html.Circle(cx="150", cy="140", r="7", fill=nc),
        # Texto central
        html.Text(pct_label, x="150", y="115", textAnchor="middle",
                  fontSize="28", fontWeight="700", fill=nc),
        html.Text("prob. de fraude", x="150", y="132", textAnchor="middle",
                  fontSize="10", fill="#9aa0a8"),
        # Etiquetas
        html.Text("0%",   x="32",  y="158", fontSize="10", fill="#c8d4e8"),
        html.Text("100%", x="258", y="158", fontSize="10", fill="#c8d4e8"),
        html.Text("Segura",      x="55",  y="178", fontSize="9", fill="#27ae60",  fontWeight="600"),
        html.Text("Sospechosa",  x="150", y="178", fontSize="9", fill="#f39c12",  fontWeight="600", textAnchor="middle"),
        html.Text("Fraude",      x="248", y="178", fontSize="9", fill="#e74c3c",  fontWeight="600", textAnchor="end"),
    ], viewBox="0 0 300 195", style={"width": "100%", "maxWidth": "280px"})

    return html.Div(svg, style={"textAlign": "center"})


def _verdict_badge(prob: float | None) -> html.Div:
    if prob is None:
        return html.Div()
    if prob < 0.30:
        label, icon, bg, fg, border = ("Legítima", "bi bi-check-circle-fill",
                                        "#e8f6f3", C_GREEN, "#c3e6cb")
    elif prob < 0.65:
        label, icon, bg, fg, border = ("Sospechosa — revisar", "bi bi-exclamation-triangle-fill",
                                        "#fffde7", C_AMBER, "#ffe082")
    else:
        label, icon, bg, fg, border = ("FRAUDE DETECTADO", "bi bi-x-circle-fill",
                                        "#fdecea", C_DANGER, "#f5c6cb")

    return html.Div([
        html.I(className=f"{icon} me-2"),
        html.Span(label, style={"fontWeight": "700"}),
    ], style={
        "background": bg, "color": fg,
        "border": f"1px solid {border}",
        "borderRadius": "10px", "padding": "13px 20px",
        "fontSize": "1rem", "textAlign": "center",
        "boxShadow": f"0 4px 12px {fg}22",
        "marginTop": "12px",
    })


def _metric_pills(metrics: dict | None) -> html.Div:
    if metrics is None:
        return html.Div()
    items = [
        ("AUC-ROC",   f"{metrics.get('auc', 0):.2f}"),
        ("Accuracy",  f"{metrics.get('accuracy', 0):.2%}"),
        ("F1-Score",  f"{metrics.get('f1', 0):.2f}"),
        ("Precision", f"{metrics.get('precision', 0):.2f}"),
    ]
    return html.Div([
        html.Div([
            html.Div(val, style={"fontSize": "1.1rem", "fontWeight": "800",
                                 "color": C_BLUE, "lineHeight": "1"}),
            html.Div(lbl, style={"fontSize": "0.68rem", "color": "#718096",
                                 "textTransform": "uppercase", "letterSpacing": "0.6px",
                                 "marginTop": "3px"}),
        ], style={
            "background": "#f7fafc", "border": "1px solid #e2e8f0",
            "borderRadius": "8px", "padding": "8px 14px", "textAlign": "center",
            "flex": "1",
        })
        for lbl, val in items
    ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "8px"})


def _historial_row(entry: dict) -> html.Div:
    p = entry["prob"]
    if p < 0.30:
        color, badge_cls = C_GREEN,  "success"
        label = "Legítima"
    elif p < 0.65:
        color, badge_cls = C_AMBER,  "warning"
        label = "Sospechosa"
    else:
        color, badge_cls = C_DANGER, "danger"
        label = "Fraude"

    bar = html.Div(
        html.Div(style={"width": f"{round(p*100)}%", "height": "100%",
                        "background": color, "borderRadius": "2px",
                        "transition": "width 0.5s ease"}),
        style={"background": "#eef2f7", "borderRadius": "2px",
               "height": "6px", "width": "80px", "overflow": "hidden"},
    )

    return html.Div([
        html.Span(f"#{entry['n']}", style={"fontSize": "0.78rem", "color": "#a0aec0",
                                           "fontWeight": "600", "minWidth": "30px"}),
        html.Span(entry["timestamp"], style={"fontSize": "0.78rem", "color": "#718096",
                                             "minWidth": "55px"}),
        html.Span(f"${entry['monto']}", style={"fontSize": "0.78rem", "fontFamily": "'Courier New', monospace",
                                                "minWidth": "65px"}),
        html.Span(f"{round(p*100)}%", style={"fontSize": "0.82rem", "fontWeight": "700",
                                              "color": color, "minWidth": "38px"}),
        bar,
        dbc.Badge(label, color=badge_cls, style={"fontSize": "0.68rem"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "12px",
              "padding": "7px 8px", "borderBottom": "1px solid #f0f2f5"})


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def layout() -> html.Div:
    _, model_ok = _cargar_modelo()

    model_banner = html.Div(
        [
            html.I(className="bi bi-exclamation-triangle-fill me-2"),
            "Modelo no encontrado en ",
            html.Code(str(MODELO_PATH)),
            ". Se está usando un modelo demo. ",
            "Copia tu .joblib a la ruta indicada para predicciones reales.",
        ],
        style={
            "background": "#fff3cd", "border": "1px solid #ffc107",
            "borderRadius": "8px", "padding": "10px 16px",
            "fontSize": "0.82rem", "color": "#856404", "marginBottom": "20px",
        },
    ) if not model_ok else html.Div()

    # ── Panel izquierdo: Inputs ───────────────────────────────────────────────
    tab_sliders = dbc.Tabs([
        dbc.Tab(_pca_tab_group(1,  7),  label="V1–V7",   tab_id="t1"),
        dbc.Tab(_pca_tab_group(8,  14), label="V8–V14",  tab_id="t2"),
        dbc.Tab(_pca_tab_group(15, 21), label="V15–V21", tab_id="t3"),
        dbc.Tab(_pca_tab_group(22, 28), label="V22–V28", tab_id="t4"),
    ], id="pca-tabs", active_tab="t1", style={"fontSize": "0.8rem"})

    amount_time = dbc.Row([
        dbc.Col([
            html.Div("Monto ($)", className="input-label"),
            dcc.Input(id="pred-amount", type="number", value=9.0, min=0,
                      step=0.01,
                      style={"width": "100%", "borderRadius": "8px",
                             "border": "1px solid #cbd5e0",
                             "padding": "8px 12px", "fontSize": "0.88rem"}),
        ], md=6),
        dbc.Col([
            html.Div("Tiempo (seg.)", className="input-label"),
            dcc.Input(id="pred-time", type="number", value=10.0, min=0,
                      step=1,
                      style={"width": "100%", "borderRadius": "8px",
                             "border": "1px solid #cbd5e0",
                             "padding": "8px 12px", "fontSize": "0.88rem"}),
        ], md=6),
    ], className="g-2 mb-3")

    quick_load = html.Div([
        html.Div("Carga rápida:", className="input-label mb-1"),
        html.Div([
            html.Button("✅ Transacción legítima", id="btn-ejemplo-legitima",
                        n_clicks=0, className="action-btn",
                        style={"fontSize": "0.78rem", "padding": "6px 14px",
                               "borderRadius": "8px", "border": "1px solid #c3e6cb",
                               "background": "#e8f6f3", "color": C_GREEN,
                               "fontWeight": "600", "cursor": "pointer"}),
            html.Button("🚨 Fraude confirmado", id="btn-ejemplo-fraude",
                        n_clicks=0, className="action-btn",
                        style={"fontSize": "0.78rem", "padding": "6px 14px",
                               "borderRadius": "8px", "border": "1px solid #f5c6cb",
                               "background": "#fdecea", "color": C_DANGER,
                               "fontWeight": "600", "cursor": "pointer"}),
        ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
    ], className="mb-3")

    analyze_btn = html.Button(
        [html.I(className="bi bi-cpu me-2"), "Analizar transacción"],
        id="btn-predecir", n_clicks=0, className="predict-btn",
        style={"width": "100%", "marginTop": "4px"},
    )

    left_panel = dbc.Card([
        html.Div("🔢  Parámetros de la Transacción", className="card-header-custom"),
        dbc.CardBody([
            quick_load,
            amount_time,
            html.Div("Variables PCA (V1–V28)", className="input-label mb-2"),
            tab_sliders,
            html.Hr(className="section-divider"),
            analyze_btn,
        ]),
    ], className="card h-100")

    # ── Panel derecho: Resultados ─────────────────────────────────────────────
    right_panel = dbc.Card([
        html.Div("📊  Resultado del Análisis", className="card-header-custom"),
        dbc.CardBody([
            # Gauge
            html.Div(id="pred-gauge",
                     children=html.Div("Analiza una transacción para ver el resultado",
                                       style={"color": "#9aa0a8", "fontSize": "0.85rem",
                                              "textAlign": "center", "padding": "32px 0"})),
            # Veredicto
            html.Div(id="pred-verdict"),
            # Pills de métricas del modelo
            html.Div(id="pred-metrics"),
            html.Hr(className="section-divider"),
            # Sub-tabs de análisis
            dcc.Tabs(id="result-tabs", value="perfil", children=[
                dcc.Tab(label="Perfil comparativo", value="perfil",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Historial sesión",   value="historial",
                        className="custom-tab", selected_className="custom-tab--selected"),
            ], className="custom-tabs",
               colors={"border": "transparent", "primary": C_BLUE, "background": "#f4f6f9"}),
            html.Div(id="result-tab-content", style={"marginTop": "12px"}),
        ]),
    ], className="card h-100")

    # ── Stores ────────────────────────────────────────────────────────────────
    stores = html.Div([
        dcc.Store(id="pred-historial", data=[]),
        dcc.Store(id="pred-v-values",  data=[0.0] * 28),
        dcc.Store(id="pred-last-prob", data=None),
    ])

    return html.Div([
        stores,
        model_banner,
        html.Div([
            html.Div("Predicción Interactiva", className="section-title"),
            html.P(
                "Ingresa los parámetros de una transacción y obtén la probabilidad de fraude.",
                className="section-body", style={"marginTop": "-8px", "marginBottom": "24px"},
            ),
        ]),
        dbc.Row([
            dbc.Col(left_panel,  md=5),
            dbc.Col(right_panel, md=7),
        ], className="g-3"),
        html.Div(style={"height": "32px"}),
    ], className="tab-content-wrapper tab-fade-in")


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def register_callbacks(app):

    # ── 1. Actualizar labels de sliders ──────────────────────────────────────
    @app.callback(
        Output({"type": "v-label", "index": __import__("dash").ALL}, "children"),
        Input({"type": "v-slider", "index": __import__("dash").ALL}, "value"),
    )
    def update_slider_labels(values):
        return [f"{v:.2f}" if v is not None else "0.00" for v in values]

    # ── 2. Cargar ejemplos rápidos ────────────────────────────────────────────
    @app.callback(
        Output({"type": "v-slider", "index": __import__("dash").ALL}, "value"),
        Output("pred-amount", "value"),
        Output("pred-time",   "value"),
        Input("btn-ejemplo-legitima", "n_clicks"),
        Input("btn-ejemplo-fraude",   "n_clicks"),
        prevent_initial_call=True,
    )
    def cargar_ejemplo(n_leg, n_fra):
        triggered = callback_context.triggered_id
        if triggered == "btn-ejemplo-legitima":
            ej = EJEMPLO_LEGITIMA
        else:
            ej = EJEMPLO_FRAUDE
        return ej["v"], ej["amount"], ej["time"]

    # ── 3. Guardar valores de sliders en Store ────────────────────────────────
    @app.callback(
        Output("pred-v-values", "data"),
        Input({"type": "v-slider", "index": __import__("dash").ALL}, "value"),
    )
    def guardar_v_values(values):
        return [v if v is not None else 0.0 for v in values]

    # ── 4. Predecir y actualizar resultados ───────────────────────────────────
    @app.callback(
        Output("pred-gauge",      "children"),
        Output("pred-verdict",    "children"),
        Output("pred-metrics",    "children"),
        Output("pred-historial",  "data"),
        Output("pred-last-prob",  "data"),
        Input("btn-predecir", "n_clicks"),
        State("pred-v-values",  "data"),
        State("pred-amount",    "value"),
        State("pred-time",      "value"),
        State("pred-historial", "data"),
        prevent_initial_call=True,
    )
    def predecir(n_clicks, v_values, amount, time_val, historial):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update

        amount   = float(amount   or 0.0)
        time_val = float(time_val or 0.0)
        v_values = [float(v or 0.0) for v in (v_values or [0.0]*28)]

        # Calcular probabilidad
        prob = _predecir(v_values, amount, time_val)

        # Cargar métricas del modelo para las pills
        from tabs.resultados import _cargar_o_calcular_metricas
        metrics = _cargar_o_calcular_metricas()

        # Actualizar historial
        ts = time.strftime("%H:%M:%S")
        pct = round(prob * 100)
        if prob < 0.30:   veredicto = "Legítima"
        elif prob < 0.65: veredicto = "Sospechosa"
        else:             veredicto = "Fraude"

        new_entry = {
            "n":         len(historial) + 1,
            "timestamp": ts,
            "monto":     f"{amount:.2f}",
            "prob":      prob,
            "veredicto": veredicto,
        }
        historial = [new_entry] + historial[:49]  # máx 50 entradas

        return (
            _gauge_svg(prob),
            _verdict_badge(prob),
            _metric_pills(metrics),
            historial,
            prob,
        )

    # ── 5. Renderizar sub-tab de resultados ───────────────────────────────────
    @app.callback(
        Output("result-tab-content", "children"),
        Input("result-tabs",     "value"),
        Input("pred-historial",  "data"),
        Input("pred-last-prob",  "data"),
        State("pred-v-values",   "data"),
        State("pred-amount",     "value"),
    )
    def render_result_tab(tab, historial, last_prob, v_values, amount):
        if tab == "historial":
            if not historial:
                return html.Div("Aún no hay transacciones analizadas.",
                                style={"color": "#9aa0a8", "fontSize": "0.85rem",
                                       "textAlign": "center", "padding": "24px 0"})
            return html.Div([_historial_row(e) for e in historial])

        # ── Perfil comparativo ─────────────────────────────────────────────
        if last_prob is None or v_values is None:
            return html.Div("Analiza una transacción para ver el perfil comparativo.",
                            style={"color": "#9aa0a8", "fontSize": "0.85rem",
                                   "textAlign": "center", "padding": "24px 0"})

        v_map = {f"V{i+1}": v_values[i] for i in range(28)}
        v_map["Amount"] = float(amount or 0.0)

        rows = []
        for _, row in PERFIL_REF.iterrows():
            var   = row["Variable"]
            leg   = row["Legitima"]
            fra   = row["Fraude"]
            actual = v_map.get(var, 0.0)
            rango = abs(fra - leg)
            dev   = min(abs(actual - leg) / rango, 1.0) if rango > 0 else 0.0

            if dev > 0.65:
                dev_color, dev_lbl = C_DANGER, "Alto"
            elif dev > 0.30:
                dev_color, dev_lbl = C_AMBER,  "Medio"
            else:
                dev_color, dev_lbl = C_GREEN,  "Bajo"

            rows.append(html.Div([
                html.Span(var,              style={"fontWeight":"700", "color": C_DARK,  "minWidth":"55px", "fontSize":"0.82rem"}),
                html.Span(f"{leg:.3f}",     style={"color": C_GREEN,  "fontFamily":"'Courier New',monospace", "fontSize":"0.8rem", "minWidth":"80px"}),
                html.Span(f"{fra:.3f}",     style={"color": C_DANGER, "fontFamily":"'Courier New',monospace", "fontSize":"0.8rem", "minWidth":"80px"}),
                html.Span(f"{actual:.3f}",  style={"color": C_BLUE,   "fontFamily":"'Courier New',monospace", "fontSize":"0.8rem", "fontWeight":"700", "minWidth":"80px"}),
                html.Span(dev_lbl, style={
                    "background": f"{dev_color}22", "border": f"1px solid {dev_color}55",
                    "color": dev_color, "borderRadius": "10px",
                    "padding": "2px 9px", "fontSize": "0.72rem", "fontWeight": "700",
                }),
            ], style={"display":"flex","gap":"8px","alignItems":"center",
                      "padding":"7px 4px","borderBottom":"1px solid #f0f2f5"}))

        header = html.Div([
            html.Span(h, style={"fontSize":"0.72rem","fontWeight":"700","color":"#6c757d",
                                "textTransform":"uppercase","letterSpacing":"0.5px",
                                "minWidth": w})
            for h, w in [("Variable","55px"),("Prom. legítima","80px"),
                         ("Prom. fraude","80px"),("Esta transacción","80px"),("Desviación","70px")]
        ], style={"display":"flex","gap":"8px","background":"#f7f8fc",
                  "borderRadius":"8px 8px 0 0","padding":"9px 4px",
                  "borderBottom":"2px solid #e9ecef"})

        note = html.Div(
            "Desviación: qué tanto el valor actual se aleja del perfil legítimo hacia el de fraude.",
            style={"fontSize":"0.71rem","color":"#9aa0a8","padding":"6px 4px"}
        )

        return html.Div([header] + rows + [note])
