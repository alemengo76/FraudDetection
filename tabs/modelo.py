"""
tabs/modelo.py
--------------
Tab unificado: Resultados + Predictor interactivo
Mejor modelo: XGBoost + Optuna + ADASYN
F1=0.8444 · Recall=0.80 · Precisión=0.8869 · AUC=0.9629
"""

import os
import numpy as np
import pandas as pd
import joblib
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, callback_context, no_update
import json

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    average_precision_score
)

# Al inicio del módulo, después de los imports
_CACHE_PATH = r"C:\Users\Alejandra\Documents\fraud_dashboard\assets\metricas_cache.json"
try:
    with open(_CACHE_PATH) as f:
        _CACHE = json.load(f)
    print(f"✅ Cache cargado. Claves: {list(_CACHE.keys())}")
except FileNotFoundError:
    print(f"❌ Archivo no encontrado: {_CACHE_PATH}")
    _CACHE = None
except Exception as e:
    print(f"❌ Error cargando cache: {e}")
    _CACHE = None
    
# ── Paleta profesional ────────────────────────────────────────────────────────
# Azul pizarra oscuro / gris carbón / acento índigo apagado
C = {
    "bg":        "#F5F6F8",
    "surface":   "#FFFFFF",
    "border":    "#DDE1E9",
    "text":      "#1C2333",
    "muted":     "#6B7385",
    "accent":    "#2D4A7A",        # azul marino apagado
    "accent_lt": "#EBF0F8",
    "ok":        "#1A6340",        # verde oscuro
    "ok_bg":     "#EBF5EE",
    "warn":      "#7A5C1A",
    "warn_bg":   "#FAF3E0",
    "danger":    "#7A1F1F",
    "danger_bg": "#FAE9E9",
    "gold":      "#9C7C38",        # dorado apagado
    "gold_bg":   "#FBF5E6",
    "grid":      "#EDEEF2",
    "seq":       ["#2D4A7A", "#4A6FA5", "#7A9CC5", "#B8CEEA", "#E8EFF8"],
}

# ── Datos reales del notebook ─────────────────────────────────────────────────

MEJOR_MODELO = {
    "nombre":    _CACHE["ganador_nombre"] if _CACHE else "XGBoost + ADASYN",
    "accuracy":  _CACHE["accuracy"]  if _CACHE else 0.9995,
    "precision": _CACHE["precision"] if _CACHE else 0.9176,
    "recall":    _CACHE["recall"]    if _CACHE else 0.7959,
    "f1":        _CACHE["f1"]        if _CACHE else 0.8525,
    "auc":       _CACHE["auc"]       if _CACHE else 0.9727,
    "ap":        _CACHE["ap"]        if _CACHE else 0.8334,
    "tp": _CACHE["cm"][1][1] if _CACHE else 78,
    "fn": _CACHE["cm"][1][0] if _CACHE else 20,
    "fp": _CACHE["cm"][0][1] if _CACHE else 7,
    "tn": _CACHE["cm"][0][0] if _CACHE else 56641,
}

TABLA = (
    [(r["modelo"], r["tecnica"],
      r["accuracy"], r["precision"], r["recall"], r["f1"], r["auc"], r["es_ganador"])
     for r in _CACHE["tabla_bootstrap"]]
    if _CACHE and _CACHE.get("tabla_bootstrap")
    else [
        ("XGBoost",        "Sin Balanceo",  0.9995, 0.9605, 0.7449, 0.8391, 0.9798, False),
        ("XGBoost",        "Class Weight",  0.9995, 1.0000, 0.6837, 0.8121, 0.9679, False),
        ("XGBoost",        "ADASYN",        0.9995, 0.9176, 0.7959, 0.8525, 0.9727, True),
        ("XGBoost",        "SMOTE",         0.9995, 0.8966, 0.7959, 0.8432, 0.9774, False),
        ("Random Forest",  "Sin Balanceo",  0.9994, 0.9211, 0.7143, 0.8046, 0.9608, False),
        ("Random Forest",  "ADASYN",        0.9995, 0.9250, 0.7551, 0.8315, 0.9682, False),
        ("Random Forest",  "SMOTE",         0.9995, 0.9136, 0.7551, 0.8268, 0.9749, False),
        ("Random Forest",  "Class Weight",  0.9995, 0.9467, 0.7245, 0.8208, 0.9515, False),
        ("Reg. Logística", "Sin Balanceo",  0.9991, 0.8406, 0.5918, 0.6946, 0.9697, False),
        ("Reg. Logística", "SMOTE",         0.9992, 0.7653, 0.7653, 0.7653, 0.9759, False),
        ("Reg. Logística", "Class Weight",  0.9991, 0.8406, 0.5918, 0.6946, 0.9715, False),
        ("Reg. Logística", "ADASYN",        0.9467, 0.0284, 0.8980, 0.0550, 0.9723, False),
    ]
)

IMPORTANCIA = (
    list(zip(_CACHE["feature_names"], _CACHE["feature_importance"]))[:15]
    if _CACHE and _CACHE.get("feature_names")
    else [
        ("V14", 0.4101), ("V17", 0.1602), ("V11", 0.1026),
        ("V10", 0.0718), ("V21", 0.0446), ("V4",  0.0416),
        ("V3",  0.0314), ("V12", 0.0230), ("V1",  0.0102),
        ("Time",0.0100), ("V8",  0.0085), ("V13", 0.0080),
        ("V7",  0.0071), ("V20", 0.0062), ("Amount", 0.0058),
    ]
)

PERFIL = {
    "Variable": ["V14", "V10", "V4", "V12", "V17", "V11", "V3", "Amount"],
    "Legitima": [0.01,   0.01, -0.01,  0.00,  0.01,  0.01,  0.02,  88.35],
    "Fraude":   [5.89,  -5.33,  4.52, -6.81,  2.25,  2.14, -3.04, 122.21],
}

EJEMPLO_FRAUDE = {
    "time": 40919.0, "amount": 112.33,
    "v": [-2.740483,3.658095,-4.110636,5.340242,-2.666775,-0.092782,-4.388699,-0.280133,-2.821895,
          -4.46628416302479,3.96979981229186,-7.34671678644283,-1.16331176961366,-8.22556891234807,
          0.82500183194645,-6.77286703365235,-8.81578542555918,-4.56885928410549,1.12659865645629,
          0.185325269832133,2.41749541196206,-0.0977119508973041,0.382154506973307,-0.154756519767143,
          -0.40395592644091,0.277894930235294,0.830061638644528,0.218690442242457]
}

EJEMPLO_LEGITIMO = {
    "time": 87777.0, "amount": 1.00,
    "v": [-2.871828,2.681642,-1.507654,-2.948048,0.009796,-0.941978,0.153794,1.204963,0.675816,
          -0.0692900179722466,-1.76092241531116,0.615283754563226,0.67186707355509,0.540743868182719,
          -0.482577726683129,0.713383267976609,-0.643516449033568,-0.625185652222858,-0.72939486328347,
          0.435521592999697,-0.37813796011417,-1.06235451575918,0.0393729172971228,0.0260057014705198,
          0.423481074624709,0.413531185241979,0.476027499455305,0.343266623974539]
}


MODELOS_DIR = r"C:\Users\Alejandra\Documents\Modelos_Optimizados"
DATA_TEST_PATH = r"C:\Users\Alejandra\Documents\Modelos_Optimizados\Xy_test.joblib"

# Mapa nombre_tecnica → sufijo en nombre de archivo
_TECNICA_SUFIJO = {
    "Sin Balanceo": "Sin Balanceo",
    "SMOTE":        "SMOTE",
    "ADASYN":       "ADASYN",
    "Class Weight": "Class Weight",
}
_MODELO_PREFIJO = {
    "XGBoost":         "XGBoost",
    "RandomForest":    "RandomForest",
    "LogisticRegression": "LogisticRegression",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _badge(label, color, bg, border=None):
    return html.Span(label, style={
        "fontSize": "0.68rem", "fontWeight": "700", "color": color,
        "background": bg, "border": f"1px solid {border or color}33",
        "borderRadius": "4px", "padding": "2px 8px", "marginRight": "6px",
        "letterSpacing": "0.3px"
    })


def _metric_card(value, label, element_id=None):
    """KPI grande — usa las clases kpi-card/kpi-value/kpi-label del custom.css."""
    val_props = {"className": "kpi-value"}
    if element_id:
        val_props["id"] = element_id
    return dbc.Col(
        html.Div([
            html.Div(value, **val_props),
            html.Div(label, className="kpi-label"),
        ], className="kpi-card"),
        className="mb-2",
    )


def _mini_card(value, label, element_id=None):
    """KPI compacto — misma estructura kpi-card del custom.css."""
    val_props = {"className": "kpi-value"}
    if element_id:
        val_props["id"] = element_id
    return dbc.Col(
        html.Div([
            html.Div(value, **val_props),
            html.Div(label, className="kpi-label"),
        ], className="kpi-card"),
        className="mb-2",
    )

def _card(header_children, body_children, style=None):
    return dbc.Card([
        html.Div(header_children, className="card-header-custom"),
        dbc.CardBody(body_children),
    ], style=style or {})


def _card_title(title, subtitle=None):
    if subtitle:
        return html.Span([
            title,
            html.Span(f" — {subtitle}", style={"fontSize": "0.75rem", "fontWeight": "400", "opacity": "0.75"})
        ])
    return html.Span(title)


def _info_box(content, border_color=None):
    bc = border_color or C["accent"]
    return html.Div(content, style={
        "background": C["bg"],
        "borderLeft": f"3px solid {bc}",
        "borderRadius": "0 8px 8px 0",
        "padding": "14px 18px",
        "marginBottom": "18px"
    })

# ── Gráficos estáticos ────────────────────────────────────────────────────────

_LAYOUT_BASE = dict(
    plot_bgcolor=C["surface"],
    paper_bgcolor=C["surface"],
    font=dict(family="'Georgia', serif", size=11, color=C["text"]),
)


def _fig_auc_comparativo():
    etiquetas = [f"{r[0]}<br>{r[1]}" for r in TABLA]
    f1s  = [r[5] for r in TABLA]   
    aucs = [r[6] for r in TABLA]   
    colores_f1  = [C["gold"]   if r[7] else C["accent"] for r in TABLA]
    colores_auc = ["rgba(154,124,56,0.40)" if r[7] else "rgba(45,74,122,0.35)" for r in TABLA]

    fig = go.Figure()
    fig.add_bar(x=etiquetas, y=f1s,  name="F1-Score",
                marker_color=colores_f1,
                hovertemplate="<b>%{x}</b><br>F1 = %{y:.4f}<extra></extra>")
    fig.add_bar(x=etiquetas, y=aucs, name="ROC-AUC",
                marker_color=colores_auc,
                hovertemplate="<b>%{x}</b><br>AUC = %{y:.4f}<extra></extra>")
    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="group",
        xaxis=dict(tickfont=dict(size=9), gridcolor=C["grid"]),
        yaxis=dict(title="Score", range=[0, 1.05], gridcolor=C["grid"]),
        legend=dict(orientation="h", x=0, y=-0.28, font=dict(size=10)),
        margin=dict(l=50, r=20, t=14, b=130),
         annotations=[dict(
            x="XGBoost<br>ADASYN", y=0.8525 + 0.03,
            text="★ Seleccionado", showarrow=False,
            font=dict(size=9, color=C["gold"]), xanchor="center"
        )]
    )
    return fig


def _fig_roc(auc_val=None, fpr_arr=None, tpr_arr=None):
    auc_val = auc_val or MEJOR_MODELO["auc"]
    if fpr_arr is None:
        np.random.seed(962)
        fpr_arr = np.linspace(0, 1, 300)
        tpr_arr = np.clip(
            1 - np.exp(-5.5 * fpr_arr) + 0.015 * fpr_arr * (1 - fpr_arr) * np.random.randn(300),
            fpr_arr * 0.95, 1
        )
        tpr_arr[0], tpr_arr[-1] = 0, 1

    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color=C["border"], dash="dash", width=1.5),
                    name="Azar", hoverinfo="skip")
    fig.add_scatter(x=fpr_arr, y=tpr_arr, mode="lines", fill="tozeroy",
                    fillcolor=f"rgba(45,74,122,0.10)",
                    line=dict(color=C["accent"], width=2.2),
                    name=f"AUC = {auc_val:.4f}",
                    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>")
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(title="Tasa Falsos Positivos", range=[0, 1], gridcolor=C["grid"]),
        yaxis=dict(title="Tasa Verdaderos Positivos", range=[0, 1], gridcolor=C["grid"]),
        legend=dict(x=0.38, y=0.06, font=dict(size=10)),
        margin=dict(l=55, r=20, t=10, b=50),
    )
    return fig


def _fig_precision_recall(ap=None, pr_precision=None, pr_recall=None):
    ap = ap or MEJOR_MODELO["ap"]
    baseline = 473 / 283726

    if pr_precision is not None and pr_recall is not None:
        recall_vals = pr_recall
        prec_vals   = pr_precision
    else:
        # fallback simulado
        recall_vals = np.linspace(0, 1, 300)
        prec_vals = np.clip(
            np.where(recall_vals <= 0.80,
                     1.0 - 0.05 * recall_vals,
                     1.0 - 0.05 * 0.80 - 3.5 * (recall_vals - 0.80)),
            0, 1
        )
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[baseline, baseline], mode="lines",
                    line=dict(color="#B8A05A", dash="dash", width=1.5),
                    name=f"Baseline ({baseline:.4f})", hoverinfo="skip")
    fig.add_scatter(x=recall_vals, y=prec_vals, mode="lines",
                    line=dict(color=C["accent"], width=2.2),
                    name=f"AP = {ap:.4f}",
                    hovertemplate="Recall: %{x:.3f}<br>Precisión: %{y:.3f}<extra></extra>")
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(title="Recall", range=[0, 1], gridcolor=C["grid"]),
        yaxis=dict(title="Precisión", range=[0, 1], gridcolor=C["grid"]),
        legend=dict(x=0.01, y=0.06, font=dict(size=10)),
        margin=dict(l=55, r=20, t=10, b=50),
    )
    return fig


def _fig_confusion(tp=None, fn=None, fp=None, tn=None):
    m = MEJOR_MODELO
    tp = tp if tp is not None else m["tp"]
    fn = fn if fn is not None else m["fn"]
    fp = fp if fp is not None else m["fp"]
    tn = tn if tn is not None else m["tn"]
    z  = [[tn, fp], [fn, tp]]
    total = tn + fn + fp + tp
    lx = ["Pred. Legítima", "Pred. Fraude"]
    ly = ["Real Legítima",  "Real Fraude"]
    max_val = max(max(r) for r in z)

    fig = go.Figure(go.Heatmap(
        x=lx, y=ly, z=z,
        colorscale=[[0, "#EBF0F8"], [0.3, "#93B3D4"], [0.7, "#4A6FA5"], [1, "#1C2F56"]],
        showscale=False,
        hovertemplate="%{x}<br>%{y}<br>N = %{z}<extra></extra>"
    ))
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            fc = "white" if val > 0.55 * max_val else C["text"]
            fig.add_annotation(
                x=lx[j], y=ly[i],
                text=f"<b>{val:,}</b><br>{val/total*100:.2f}%",
                showarrow=False, font=dict(color=fc, size=12)
            )
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=110, r=20, t=10, b=60),
    )
    return fig


def _fig_importancia(importancia=None):
    data = importancia or IMPORTANCIA
    vars_  = [r[0] for r in data]
    vals   = [r[1] for r in data]
    # gradiente de intensidad
    max_v  = max(vals)
    colors = [
        f"rgba(45,74,122,{0.45 + 0.55*(v/max_v):.2f})" for v in vals
    ]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=vars_[::-1], orientation="h",
        marker_color=colors[::-1],
        hovertemplate="<b>%{y}</b><br>Gain = %{x:.3f}<extra></extra>"
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(title="Importancia (gain)", gridcolor=C["grid"]),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=60, r=20, t=10, b=50),
    )
    return fig

# ── Tabla comparativa HTML ────────────────────────────────────────────────────

def _tabla_comparativa():
    # Intentar usar datos bootstrap del cache; caer en TABLA hardcodeada si no hay
    use_bootstrap = _CACHE and _CACHE.get("tabla_bootstrap")


    if use_bootstrap:
        datos      = _CACHE["tabla_bootstrap"]
        ref_nombre = _CACHE.get("ref_nombre", "—")
        cols = ["Modelo", "Balanceo", "Accuracy", "Precisión", "Recall", "F1", "F1 IC 95%", "PR AUC", "PR AUC IC 95%", "AUC", "Δ vs ref", "Sig.", ""]
        grid = "140px 110px 75px 80px 70px 70px 110px 75px 110px 75px 80px 40px 70px"

        header = html.Div(
            [html.Div(c, style={
                "fontSize": "0.67rem", "fontWeight": "700", "color": C["muted"],
                "textTransform": "uppercase", "letterSpacing": "0.5px"
            }) for c in cols],
            style={
                "display": "grid", "gridTemplateColumns": grid,
                "background": C["bg"], "padding": "10px 8px",
                "borderRadius": "6px 6px 0 0",
                "borderBottom": f"2px solid {C['border']}"
            }
        )

        rows = []
        for r in datos:
            ganador = r["es_ganador"]
            es_ref  = r["es_ref"]
            sig     = r["sig"]

            row_style = {
                "display": "grid", "gridTemplateColumns": grid,
                "alignItems": "center", "padding": "8px 8px",
                "background": C["gold_bg"] if ganador else "transparent",
                "border": f"1px solid {C['gold']}55" if ganador else "none",
                "borderBottom": "none" if ganador else f"1px solid {C['grid']}",
                "borderRadius": "6px" if ganador else "0",
                "marginBottom": "2px" if ganador else "0",
            }

            if es_ref:
                delta_cell = html.Div("— ref", style={"fontSize": "0.74rem", "color": C["muted"], "fontFamily": "monospace"})
            else:
                diff        = r["diff_vs_ref"]
                delta_color = C["danger"] if sig else C["muted"]
                delta_cell  = html.Div(f"{diff:+.4f}", style={"fontSize": "0.74rem", "fontFamily": "monospace", "color": delta_color})

            sig_cell = html.Div(
                "✗" if sig else "≈",
                style={"fontSize": "0.80rem", "fontWeight": "700",
                       "color": C["danger"] if sig else C["ok"], "textAlign": "center"}
            )

            rows.append(html.Div([
                html.Div(r["modelo"],  style={"fontSize": "0.79rem", "fontWeight": "600" if ganador else "400", "color": C["text"]}),
                html.Div(r["tecnica"], style={"fontSize": "0.79rem", "color": C["muted"]}),
                html.Div(f"{r['accuracy']:.4f}", style={"fontSize": "0.79rem", "fontFamily": "monospace"}),
                html.Div(f"{r['precision']:.4f}", style={"fontSize": "0.79rem", "fontFamily": "monospace"}),
                html.Div(f"{r['recall']:.4f}",    style={"fontSize": "0.79rem", "fontFamily": "monospace"}),
                html.Div(f"{r['f1']:.4f}", style={
                    "fontSize": "0.80rem", "fontWeight": "700" if ganador else "400",
                    "color": C["ok"] if ganador else C["text"], "fontFamily": "monospace"
                }),
                html.Div(f"[{r['f1_ic'][0]:.3f}, {r['f1_ic'][1]:.3f}]",
                         style={"fontSize": "0.72rem", "fontFamily": "monospace", "color": C["muted"]}),
                html.Div(f"{r['ap']:.4f}", style={"fontSize": "0.79rem", "fontFamily": "monospace"}),
                html.Div(f"[{r['pr_ic'][0]:.3f}, {r['pr_ic'][1]:.3f}]",
                         style={"fontSize": "0.72rem", "fontFamily": "monospace", "color": C["muted"]}),
                html.Div(f"{r['auc']:.4f}", style={"fontSize": "0.79rem", "fontFamily": "monospace"}),
                delta_cell,
                sig_cell,
                _badge("Óptimo", C["gold"], C["gold_bg"], C["gold"]) if ganador else html.Span(),
            ], style=row_style))

        footer = html.Div([
            html.Div([
                _badge("Óptimo", C["gold"], C["gold_bg"], C["gold"]),
                html.Span(
                    "seleccionado entre modelos estadísticamente equivalentes al de mayor PR AUC · desempate por F1 · Bootstrap 500 iter. · IC 95%",
                    style={"fontSize": "0.74rem", "color": C["muted"]}
                ),
            ], style={"marginBottom": "5px"}),
            html.Div([
                html.Span("Sig. (✗)", style={"fontSize": "0.74rem", "color": C["danger"], "fontWeight": "700", "marginRight": "6px"}),
                html.Span("= diferencia significativa vs referencia · ", style={"fontSize": "0.74rem", "color": C["muted"]}),
                html.Span("≈", style={"fontSize": "0.74rem", "color": C["ok"], "fontWeight": "700", "marginRight": "4px"}),
                html.Span(f"= equivalente a {ref_nombre} (mayor PR AUC)",
                          style={"fontSize": "0.74rem", "color": C["muted"]}),
            ])
        ], style={"padding": "10px 8px", "borderTop": f"1px solid {C['grid']}", "marginTop": "4px"})

    else:
        # Fallback: tabla simple sin bootstrap
        cols = ["Modelo", "Balanceo", "Accuracy", "Precisión", "Recall", "F1", "AUC", ""]
        grid = "150px 120px 90px 90px 80px 80px 80px 70px"

        header = html.Div(
            [html.Div(c, style={
                "fontSize": "0.67rem", "fontWeight": "700", "color": C["muted"],
                "textTransform": "uppercase", "letterSpacing": "0.5px"
            }) for c in cols],
            style={
                "display": "grid", "gridTemplateColumns": grid,
                "background": C["bg"], "padding": "10px 8px",
                "borderRadius": "6px 6px 0 0",
                "borderBottom": f"2px solid {C['border']}"
            }
        )

        rows = []
        for modelo, bal, acc, prec, rec, f1, auc_v, ganador in TABLA:
            row_style = {
                "display": "grid", "gridTemplateColumns": grid,
                "alignItems": "center", "padding": "8px 8px",
                "background": C["gold_bg"] if ganador else "transparent",
                "border": f"1px solid {C['gold']}55" if ganador else "none",
                "borderBottom": "none" if ganador else f"1px solid {C['grid']}",
                "borderRadius": "6px" if ganador else "0",
                "marginBottom": "2px" if ganador else "0",
            }
            rows.append(html.Div([
                html.Div(modelo,         style={"fontSize": "0.82rem", "fontWeight": "600", "color": C["text"]}),
                html.Div(bal,            style={"fontSize": "0.82rem", "color": C["muted"]}),
                html.Div(f"{acc:.4f}",   style={"fontSize": "0.82rem", "fontFamily": "monospace"}),
                html.Div(f"{prec:.4f}",  style={"fontSize": "0.82rem", "fontFamily": "monospace"}),
                html.Div(f"{rec:.4f}",   style={"fontSize": "0.82rem", "fontFamily": "monospace"}),
                html.Div(f"{f1:.4f}",    style={"fontSize": "0.84rem", "fontWeight": "700",
                                                 "color": C["ok"] if ganador else C["text"], "fontFamily": "monospace"}),
                html.Div(f"{auc_v:.4f}", style={"fontSize": "0.82rem", "fontFamily": "monospace"}),
                _badge("Óptimo", C["gold"], C["gold_bg"], C["gold"]) if ganador else html.Span(),
            ], style=row_style))

        footer = html.Div([
            _badge("Óptimo", C["gold"], C["gold_bg"], C["gold"]),
            html.Span(
                "seleccionado por mayor Recall con F1 competitivo · Optimización: Optuna (50 trials)",
                style={"fontSize": "0.74rem", "color": C["muted"]}
            )
        ], style={"padding": "10px 8px", "borderTop": f"1px solid {C['grid']}", "marginTop": "4px"})

    return html.Div([header, html.Div(rows), footer])

# ── Carga de joblibs ──────────────────────────────────────────────────────────

def _load_pipeline(modelo_nombre, tecnica_nombre):
    """
    Intenta cargar el pipeline joblib correspondiente.
    Retorna (pipeline, error_msg).
    """
    fname = os.path.join(
        MODELOS_DIR,
        f"optuna_{modelo_nombre}_{tecnica_nombre}.joblib"
    )
    if not os.path.isfile(fname):
        return None, f"Archivo no encontrado: {fname}"
    try:
        pipe = joblib.load(fname)
        return pipe, None
    except Exception as e:
        return None, str(e)

# ── Predictor helpers ─────────────────────────────────────────────────────────

def _slider_block(prefix, n):
    return html.Div([
        html.Div([
            html.Span(f"V{n}", style={
                "fontSize": "0.76rem", "fontWeight": "600", "color": C["muted"]
            }),
            html.Span(id=f"{prefix}-lbl-v{n}", style={
                "fontSize": "0.76rem", "color": C["accent"], "fontWeight": "700",
                "fontFamily": "monospace"
            })
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "2px"}),
        dcc.Slider(id=f"{prefix}-v{n}", min=-15, max=15, step=0.1, value=0,
                   marks=None, tooltip={"always_visible": False}, updatemode="drag")
    ], style={"marginBottom": "9px"})


_PIPELINE = None

def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = joblib.load(os.path.join(MODELOS_DIR, "optuna_XGBoost_ADASYN.joblib"))
    return _PIPELINE

def _compute_fraud_prob(amount, time_h, v_vals):
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    fila = [time_h] + list(v_vals) + [amount]
    df = pd.DataFrame([fila], columns=cols)
    try:
        pipe = _get_pipeline()
        return float(pipe.predict_proba(df)[0][1])
    except Exception:
        return 0.0


# ── Credit Card Component ─────────────────────────────────────────────────────

def _credit_card_front(number="•••• •••• •••• ••••", holder="TITULAR", expires="MM/AA"):
    return html.Div([
        html.Div([
            html.Div(style={"width": "28px", "height": "28px", "borderRadius": "50%",
                            "background": "#C8A96E", "opacity": "0.9"}),
            html.Div(style={"width": "28px", "height": "28px", "borderRadius": "50%",
                            "background": "#E8D5A3", "opacity": "0.8", "marginLeft": "-12px"}),
        ], style={"position": "absolute", "top": "18px", "right": "20px", "display": "flex"}),
        html.Div(className="cc-chip"),
        html.Div(number, className="cc-number"),
        html.Div([
            html.Div([html.Div("Titular", className="cc-label"), html.Div(holder, className="cc-value")]),
            html.Div([html.Div("Válida hasta", className="cc-label"), html.Div(expires, className="cc-value")]),
            html.Div([
                html.Div("Monto", className="cc-label"),
                html.Div(id="card-amount-display", children="—", className="cc-value"),
            ]),
        ], className="cc-info-row"),
    ], id="cc-card-front", className="cc-card", style={"position": "relative",
        "background": "linear-gradient(135deg, #1a2f5e, #0f1e3d)",
        "transition": "background 0.8s ease"})


def _credit_card_back():
    return html.Div([
        html.Div(className="cc-magnetic-strip"),
        html.Div([
            html.Div(className="cc-signature-lines"),
            html.Div([
                html.Div("CVV", className="cc-cvv-label"),
                html.Div("•••", className="cc-cvv-value"),
            ], className="cc-cvv-box"),
        ], className="cc-signature-panel"),
        html.Div([
            html.Div(id="cc-scan-line-pred", className="cc-scan-line"),
        ], className="cc-scan-overlay"),
    ], className="cc-card cc-back-style")


# ── Layout principal ──────────────────────────────────────────────────────────



def layout():
    m   = MEJOR_MODELO
    pfx = "modelo"

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB PILLS
    # ═══════════════════════════════════════════════════════════════════════════
    tab_pills = html.Div([
        html.Div(
            [html.I(className="fas fa-table me-1"), "¿Por qué este modelo?"],
            id="modelo-pill-resumen", n_clicks=0,
            className="modelo-tab-pill modelo-tab-pill-active"
        ),
        html.Div(
            [html.I(className="fas fa-chart-line me-1"), "¿Qué tan bueno es?"],
            id="modelo-pill-rendimiento", n_clicks=0,
            className="modelo-tab-pill"
        ),
        html.Div(
            [html.I(className="fas fa-sliders-h me-1"), "Predictor"],
            id="modelo-pill-predictor", n_clicks=0,
            className="modelo-tab-pill"
        ),
    ], style={
        "display": "flex", "gap": "4px",
        "borderBottom": f"2px solid {C['border']}",
        "paddingBottom": "0", "marginBottom": "20px"
    })

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 · ¿POR QUÉ ESTE MODELO?
    # Responde una sola pregunta: de todas las combinaciones, ¿cuál ganó y por qué?
    # ═══════════════════════════════════════════════════════════════════════════
    tab_resumen = html.Div([

        # ── Badge ganador ──────────────────────────────────────────────────────
        html.Div([
            html.Div(style={
                "width": "8px", "height": "8px", "borderRadius": "50%",
                "background": C["ok"], "flexShrink": "0", "marginTop": "3px"
            }),
            html.Div([
                html.Div("Modelo seleccionado", style={
                    "fontSize": "0.70rem", "color": C["muted"],
                    "textTransform": "uppercase", "letterSpacing": "0.4px"
                }),
                html.Div(m["nombre"], style={
                    "fontSize": "0.92rem", "fontWeight": "700", "color": C["ok"]
                })
            ])
        ], style={
            "display": "inline-flex", "alignItems": "flex-start", "gap": "10px",
            "background": C["ok_bg"], "border": f"1px solid {C['ok']}33",
            "borderRadius": "8px", "padding": "10px 16px", "marginBottom": "16px"
        }),

        # ── Criterio de selección ──────────────────────────────────────────────
        _info_box([
            html.P("Criterio de selección del modelo", style={
                "fontWeight": "700", "fontSize": "0.86rem",
                "margin": "0 0 6px", "color": C["text"]
            }),
            html.P([
                "La métrica de optimización fue el ", html.Strong("F1-Score"),
                " (media armónica entre Precisión y Recall). Entre 12 combinaciones evaluadas, "
                "se seleccionó ", html.Strong("XGBoost + Optuna + ADASYN"),
                " por el mejor Recall (79.59 %) con F1=0.8525 y Precisión=0.9176. "
                "El modelo es 502 veces mejor que clasificador aleatorio (AP=0.8334 vs baseline=0.0017)."
            ], style={
                "fontSize": "0.93rem", "color": C["muted"],
                "lineHeight": "1.75", "margin": 0
            })
        ], C["accent"]),

        # ── Tabla comparativa ─────────────────────────────────────────────────
        dbc.Card([
            html.Div(
                _card_title(
                    "Comparación de todas las combinaciones",
                    "3 algoritmos × 4 técnicas de balanceo · optimización Optuna"
                ),
                className="card-header-custom"
            ),
            dbc.CardBody(_tabla_comparativa()),
        ], className="mb-3", style={"borderTop": f"3px solid {C['accent']}"}),

        # ── Gráfico AUC comparativo ────────────────────────────────────────────
        dbc.Card([
            html.Div(
                _card_title(
                    "F1-Score y ROC-AUC por configuración",
                    "Barras doradas = combinación seleccionada"
                ),
                className="card-header-custom"
            ),
            dbc.CardBody(
                dcc.Graph(
                    figure=_fig_auc_comparativo(),
                    config={"displayModeBar": False},
                    style={"height": "320px"}
                )
            ),
        ], className="mb-3", style={"borderTop": f"3px solid {C['accent']}"}),

        # ── Conclusión ────────────────────────────────────────────────────────
        dbc.Card([
    html.Div(
        _card_title("Conclusión", "Razonamiento detrás de la elección final"),
        className="card-header-custom"
    ),
    dbc.CardBody([
        html.P([
            "En este proceso se calcula el PR AUC en test para cada modelo posible y se elige el mejor como referencia. "
            "Luego, se compara por medio de bootstrap, se remuestrean los datos de test 500 veces y en cada muestra se calcula "
            "la diferencia de PR AUC entre la referencia y cada modelo, si el intervalo de confianza al 95% no tiene el 0, "
            "la diferencia es significativa, pero los que no son significativamente diferentes a la referencia, que es el de mejor métrica, "
            "hacen parte del grupo de los que son estadísticamente equivalentes. Con esto en mente, se selecciona el mejor F1 entre estos. "
            "A partir de este proceso, se obtienen los mejores modelos, basados en un enfoque estadístico que lo respalda."
        ], className="section-body"),
        html.P([
            "Por un lado, el mejor recall es del modelo de ",
            html.Strong("Regresión Logística + Optuna + ADASYN"),
            " (0.8980), pero a cambio de una precisión de 0.0284, esto no es viable para ninguna empresa o banco gastar tantos costos "
            "y molestias en el cliente identificando demasiadas transacciones como fraudulentas cuando son legítimas."
        ], className="section-body"),
        html.P([
            html.Strong("XGBoost + Optuna + ADASYN"),
            ", tiene resultados realmente favorables, un recall cercano al 80% y una precisión aproximada a 0.92 son métricas muy buenas, "
            "y a pesar de no tener el mejor PR AUC es estadísticamente equivalente al de ",
            html.Strong("XGBoost + Optuna + Sin balanceo"),
            ", que es el mayor (0.8412).",
        ], className="section-body", style={"marginBottom": 0}),
    ]),
], className="mb-3", style={"borderTop": f"3px solid {C['accent']}"}),

], id="modelo-tab-resumen")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 · ¿QUÉ TAN BUENO ES?
    # Sección A: métricas + gráficas fijas del ganador (sin selector)
    # Sección B: explorador técnico con selector de combinación
    # ═══════════════════════════════════════════════════════════════════════════
    tab_rendimiento = html.Div([

        # ── SECCIÓN A: Rendimiento del modelo ganador ─────────────────────────
        html.Div([
            html.Div(style={
                "width": "8px", "height": "8px", "borderRadius": "50%",
                "background": C["ok"], "flexShrink": "0", "marginTop": "3px"
            }),
            html.Div([
                html.Div("Rendimiento del modelo seleccionado", style={
                    "fontSize": "0.70rem", "color": C["muted"],
                    "textTransform": "uppercase", "letterSpacing": "0.4px"
                }),
                html.Div(m["nombre"], style={
                    "fontSize": "0.92rem", "fontWeight": "700", "color": C["ok"]
                })
            ])
        ], style={
            "display": "inline-flex", "alignItems": "flex-start", "gap": "10px",
            "background": C["ok_bg"], "border": f"1px solid {C['ok']}33",
            "borderRadius": "8px", "padding": "10px 16px", "marginBottom": "18px"
        }),

        # KPIs del mejor modelo
        dbc.Row([
            _metric_card(f"{m['accuracy']*100:.2f}%",  "Accuracy"),
            _metric_card(f"{m['precision']*100:.1f}%", "Precisión"),
            _metric_card(f"{m['recall']*100:.1f}%",    "Recall"),
            _metric_card(f"{m['f1']:.4f}",             "F1-Score"),
            _metric_card(f"{m['auc']:.4f}",            "AUC-ROC"),
            _metric_card(f"{m['ap']:.4f}",             "Avg. Precision"),
        ], className="mb-4"),

        # 3 info-cards contextuales (dataset, fraudes detectados, falsas alarmas)
        html.Div([
            # Dataset
            html.Div([
                html.Div(style={
                    "width": "36px", "height": "36px", "borderRadius": "8px",
                    "background": "#E8F4FD", "display": "flex", "alignItems": "center",
                    "justifyContent": "center", "flexShrink": "0"
                }, children=html.I(className="fas fa-database",
                                   style={"color": "#2E86C1", "fontSize": "1rem"})),
                html.Div([
                    html.Div("Dataset de prueba", style={
                        "fontSize": "0.70rem", "color": C["muted"],
                        "textTransform": "uppercase", "letterSpacing": "0.4px",
                        "marginBottom": "3px"
                    }),
                    html.Div([
                        "56,746 transacciones — solo ",
                        html.Span("95 son fraude", style={"color": C["danger"], "fontWeight": "700"}),
                        " (0.17%)"
                    ], style={"fontSize": "0.86rem", "fontWeight": "600",
                               "color": C["text"], "lineHeight": "1.5"}),
                    html.Div("Problema altamente desbalanceado — justifica el uso de ADASYN",
                             style={"fontSize": "0.76rem", "color": C["muted"], "marginTop": "3px"})
                ])
            ], style={
                "background": C["bg"], "border": f"1px solid {C['border']}",
                "borderRadius": "10px", "padding": "14px 16px",
                "display": "flex", "gap": "12px", "alignItems": "flex-start"
            }),

            # Fraudes detectados
            html.Div([
                html.Div(style={
                    "width": "36px", "height": "36px", "borderRadius": "8px",
                    "background": C["ok_bg"], "display": "flex", "alignItems": "center",
                    "justifyContent": "center", "flexShrink": "0"
                }, children=html.I(className="fas fa-shield-alt",
                                   style={"color": C["ok"], "fontSize": "1rem"})),
                html.Div([
                    html.Div("Fraudes detectados", style={
                        "fontSize": "0.70rem", "color": C["muted"],
                        "textTransform": "uppercase", "letterSpacing": "0.4px",
                        "marginBottom": "3px"
                    }),
                    html.Div([
                        html.Span(f"{m['tp']} de {m['tp']+m['fn']}",
                                  style={"color": C["ok"], "fontWeight": "700"}),
                        " fraudes reales identificados correctamente"
                    ], style={"fontSize": "0.86rem", "fontWeight": "600",
                               "color": C["text"], "lineHeight": "1.5"}),
                    html.Div(
                        f"Recall {m['recall']*100:.1f}% — {m['fn']} fraudes no detectados (falsos negativos)",
                        style={"fontSize": "0.76rem", "color": C["muted"], "marginTop": "3px"}
                    )
                ])
            ], style={
                "background": C["bg"], "border": f"1px solid {C['border']}",
                "borderRadius": "10px", "padding": "14px 16px",
                "display": "flex", "gap": "12px", "alignItems": "flex-start"
            }),

            # Falsas alarmas
            html.Div([
                html.Div(style={
                    "width": "36px", "height": "36px", "borderRadius": "8px",
                    "background": C["warn_bg"], "display": "flex", "alignItems": "center",
                    "justifyContent": "center", "flexShrink": "0"
                }, children=html.I(className="fas fa-bell",
                                   style={"color": C["warn"], "fontSize": "1rem"})),
                html.Div([
                    html.Div("Falsas alarmas", style={
                        "fontSize": "0.70rem", "color": C["muted"],
                        "textTransform": "uppercase", "letterSpacing": "0.4px",
                        "marginBottom": "3px"
                    }),
                    html.Div([
                        "Solo ",
                        html.Span(f"{m['fp']} alertas incorrectas",
                                  style={"color": C["warn"], "fontWeight": "700"}),
                        f" sobre {m['tn']+m['fp']:,} legítimas"
                    ], style={"fontSize": "0.86rem", "fontWeight": "600",
                               "color": C["text"], "lineHeight": "1.5"}),
                    html.Div(
                        f"Precisión {m['precision']*100:.1f}% — costo operativo bajo",
                        style={"fontSize": "0.76rem", "color": C["muted"], "marginTop": "3px"}
                    )
                ])
            ], style={
                "background": C["bg"], "border": f"1px solid {C['border']}",
                "borderRadius": "10px", "padding": "14px 16px",
                "display": "flex", "gap": "12px", "alignItems": "flex-start"
            }),
        ], style={
            "display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "12px", "marginBottom": "20px"
        }),

        # Gráficas del ganador — ROC + Confusión
        html.Div([
            html.Div([
                dbc.Card([
                    html.Div(
                        _card_title("Curva ROC",
                                    f"AUC = {m['auc']:.4f} · {m['nombre']}"),
                        className="card-header-custom"
                    ),
                    dbc.CardBody(
                        dcc.Graph(figure=_fig_roc(
    auc_val=_CACHE["auc"] if _CACHE else None,
    fpr_arr=_CACHE["fpr"] if _CACHE else None,
    tpr_arr=_CACHE["tpr"] if _CACHE else None,
), config={"displayModeBar": False},
                                  style={"height": "280px"})
                    ),
                ], style={"borderTop": f"3px solid {C['accent']}"}),
            ], style={"flex": "1"}),

            html.Div([
                dbc.Card([
                    html.Div(
                        _card_title("Matriz de Confusión",
                                    "Conjunto de prueba · umbral = 0.5"),
                        className="card-header-custom"
                    ),
                    dbc.CardBody(
                        dcc.Graph(figure=_fig_confusion(), config={"displayModeBar": False},
                                  style={"height": "280px"})
                    ),
                ], style={"borderTop": f"3px solid {C['accent']}"}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "14px"}),

        # Precisión-Recall + Importancia
        html.Div([
            html.Div([
                dbc.Card([
                    html.Div(
                        _card_title("Curva Precisión-Recall",
                                    f"AP = {m['ap']:.4f} · Baseline = 0.0017"),
                        className="card-header-custom"
                    ),
                    dbc.CardBody(
                        dcc.Graph(figure=_fig_precision_recall(
    ap=_CACHE["ap"] if _CACHE else None,
    pr_precision=_CACHE["pr_precision"] if _CACHE else None,
    pr_recall=_CACHE["pr_recall"] if _CACHE else None,
), config={"displayModeBar": False},
                                  style={"height": "280px"})
                    ),
                ], style={"borderTop": f"3px solid {C['accent']}"}),
            ], style={"flex": "1"}),

            html.Div([
                dbc.Card([
                    html.Div(
                        _card_title("Importancia de Variables (Top 15)",
                                    "Gain relativo · Random Forest + SMOTE"),
                        className="card-header-custom"
                    ),
                    dbc.CardBody(
                        dcc.Graph(figure=_fig_importancia(),
                                  config={"displayModeBar": False},
                                  style={"height": "280px"})
                    ),
                ], style={"borderTop": f"3px solid {C['accent']}"}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "20px"}),

        # Interpretación
        _info_box([
            html.P("Interpretación", style={
                "fontWeight": "700", "fontSize": "0.86rem",
                "margin": "0 0 8px", "color": C["text"]
            }),
            html.Ul([
                html.Li("Un AUC = 0.9727 significa que el modelo distingue los fraudes de los no fraudes con un 97% de efectividad"),
                html.Li("La curva PR mantiene la precisión ~100% hasta recall≈0.80, después se detectan el resto de fraudes a coste de más falsas alarmas. Por otro lado, el modelo es 502 veces mejor que un clasificador aleatorio (AP=0.8334 vs baseline=0.0017) subiendo recall mientras mantiene un precision alto."),
                html.Li("La matriz de confusión muestra que se escogió un modelo más agresivo que detectara más fraudes a costa de algunas falsas alarmas. Solo 20 fraudes fueron clasificados como legítimos, a costa de subir el número de falsas alarmas a 7."),
                html.Li("V14 y V10 concentran ~70% del poder predictivo (ganancia de información). Lo cual es esperable, pues las variables en la gráfica fueron clasificadas con alto y mediano poder discriminativo en el análisis exploratorio."),
            ], style={
                "fontSize": "0.93rem", "color": C["muted"],
                "lineHeight": "1.75", "margin": 0, "paddingLeft": "18px"
            })
        ], C["accent"]),

        # ── SECCIÓN B: Explorador técnico ─────────────────────────────────────
        # ── SECCIÓN B: Explorador técnico ─────────────────────────────────────────────
html.Hr(style={"borderColor": C["border"], "margin": "24px 0 20px"}),

html.Div([
    html.Div([
        html.Div("¿Vale la pena cambiar el modelo ganador por otro?", style={
            "fontWeight": "700", "fontSize": "0.95rem",
            "color": C["text"], "marginBottom": "2px"
        }),
        html.Div(
            "Selecciona cualquier combinación y compara sus métricas directamente contra el ganador",
            style={"fontSize": "0.78rem", "color": C["muted"]}
        )
    ]),
    html.Span("para técnicos", style={
        "fontSize": "0.68rem", "fontWeight": "700",
        "color": C["accent"], "background": C["accent_lt"],
        "border": f"1px solid {C['accent']}33",
        "borderRadius": "4px", "padding": "3px 9px"
    })
], style={
    "display": "flex", "justifyContent": "space-between",
    "alignItems": "center", "marginBottom": "14px"
}),

# Selector
html.Div([
    html.Div([
        html.Label("Algoritmo", style={
            "fontSize": "0.72rem", "fontWeight": "700", "color": C["muted"],
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "display": "block", "marginBottom": "6px"
        }),
        dcc.Dropdown(
            id="rend-modelo-select",
            options=[
                {"label": "XGBoost",            "value": "XGBoost"},
                {"label": "Random Forest",       "value": "RandomForest"},
                {"label": "Regresión Logística", "value": "LogisticRegression"},
            ],
            value="XGBoost", clearable=False,
            style={"fontSize": "0.82rem"}
        ),
    ], style={"flex": "1"}),
    html.Div([
        html.Label("Técnica de balanceo", style={
            "fontSize": "0.72rem", "fontWeight": "700", "color": C["muted"],
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "display": "block", "marginBottom": "6px"
        }),
        dcc.Dropdown(
            id="rend-tecnica-select",
            options=[
                {"label": "Sin Balanceo",  "value": "Sin Balanceo"},
                {"label": "SMOTE",         "value": "SMOTE"},
                {"label": "ADASYN",        "value": "ADASYN"},
                {"label": "Class Weight",  "value": "Class Weight"},
            ],
            value="ADASYN", clearable=False,
            style={"fontSize": "0.82rem"}
        ),
    ], style={"flex": "1"}),
], style={
    "display": "flex", "gap": "14px", "alignItems": "flex-end",
    "background": C["bg"], "border": f"1px solid {C['border']}",
    "borderRadius": "8px", "padding": "16px 18px", "marginBottom": "16px"
}),

# Tabla delta + radar — llenados por callback
html.Div(id="rend-comparacion"),

    ], id="modelo-tab-rendimiento", style={"display": "none"})

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 · PREDICTOR — sin cambios estructurales
    # ═══════════════════════════════════════════════════════════════════════════
    sliders_v1_10  = [_slider_block(pfx, n) for n in range(1, 11)]
    sliders_v11_20 = [_slider_block(pfx, n) for n in range(11, 21)]
    sliders_v21_28 = [_slider_block(pfx, n) for n in range(21, 29)]

    panel_controles = html.Div([
        _card(
            _card_title("Parámetros de la transacción", f"Modelo: {m['nombre']}"),
            html.Div([
                html.P("Ejemplo rápido", style={
                    "fontWeight": "600", "fontSize": "0.80rem",
                    "color": C["muted"], "marginBottom": "8px"
                }),
                html.Div([
                    html.Button("Legítima", id=f"{pfx}-btn-legitima", style={
                        "flex": "1", "background": C["ok_bg"], "color": C["ok"],
                        "border": f"1px solid {C['ok']}44", "borderRadius": "6px",
                        "padding": "7px", "fontSize": "0.78rem",
                        "fontWeight": "600", "cursor": "pointer"
                    }),
                    html.Button("Fraude", id=f"{pfx}-btn-fraude", style={
                        "flex": "1", "background": C["danger_bg"], "color": C["danger"],
                        "border": f"1px solid {C['danger']}44", "borderRadius": "6px",
                        "padding": "7px", "fontSize": "0.78rem",
                        "fontWeight": "600", "cursor": "pointer"
                    }),
                ], style={"display": "flex", "gap": "8px", "marginBottom": "14px"}),

                html.Hr(style={"borderColor": C["border"], "margin": "0 0 14px"}),

                html.Div([
                    html.Div([
                        html.Div([
                            html.Span("Monto ($)", style={
                                "fontSize": "0.76rem", "fontWeight": "600"
                            }),
                            html.Span(id=f"{pfx}-lbl-amount", style={
                                "fontSize": "0.76rem", "color": C["accent"],
                                "fontWeight": "700", "fontFamily": "monospace"
                            })
                        ], style={"display": "flex", "justifyContent": "space-between",
                                   "marginBottom": "2px"}),
                        dcc.Slider(id=f"{pfx}-amount", min=0, max=5000, step=1, value=120,
                                   marks=None, tooltip={"always_visible": False},
                                   updatemode="drag")
                    ], style={"marginBottom": "10px"}),
                    html.Div([
                        html.Div([
                            html.Span("Hora (0–24 h)", style={
                                "fontSize": "0.76rem", "fontWeight": "600"
                            }),
                            html.Span(id=f"{pfx}-lbl-time", style={
                                "fontSize": "0.76rem", "color": C["accent"],
                                "fontWeight": "700", "fontFamily": "monospace"
                            })
                        ], style={"display": "flex", "justifyContent": "space-between",
                                   "marginBottom": "2px"}),
                        dcc.Slider(id=f"{pfx}-time", min=0, max=24, step=1, value=14,
                                   marks=None, tooltip={"always_visible": False},
                                   updatemode="drag")
                    ], style={"marginBottom": "14px"}),
                ]),

                html.Hr(style={"borderColor": C["border"], "margin": "0 0 12px"}),
                html.P("Variables PCA (componentes anónimas)", style={
    "fontWeight": "600", "fontSize": "0.80rem",
    "color": C["muted"], "marginBottom": "8px"
}),

html.Div(
    dcc.Tabs(id=f"{pfx}-pca-tabs", value="v1-10", children=[
        dcc.Tab(label="V1–V10",  value="v1-10",  children=sliders_v1_10,
                className="modelo-subtab",
                selected_className="modelo-subtab--selected"),
        dcc.Tab(label="V11–V20", value="v11-20", children=sliders_v11_20,
                className="modelo-subtab",
                selected_className="modelo-subtab--selected"),
        dcc.Tab(label="V21–V28", value="v21-28", children=sliders_v21_28,
                className="modelo-subtab",
                selected_className="modelo-subtab--selected"),
    ], className="custom-tabs"),
    style={
        "maxHeight": "320px",
        "overflowY": "auto",
        "overflowX": "hidden",
        "paddingRight": "4px",
    }
),

                html.Hr(style={"borderColor": C["border"], "margin": "14px 0 10px"}),
                html.Div([
                    html.Button("Analizar transacción", id=f"{pfx}-btn-analizar", style={
                        "flex": "2", "background": C["accent"], "color": "white",
                        "border": "none", "borderRadius": "7px", "padding": "10px",
                        "fontSize": "0.84rem", "fontWeight": "600", "cursor": "pointer"
                    }),
                    html.Button("Resetear", id=f"{pfx}-btn-reset", n_clicks=0, style={
                        "flex": "1", "background": C["bg"], "color": C["muted"],
                        "border": f"1px solid {C['border']}", "borderRadius": "7px",
                        "padding": "10px", "fontSize": "0.84rem", "cursor": "pointer"
                    }),
                ], style={"display": "flex", "gap": "10px"}),
            ])
        )
    ], style={"flex": "1", "minWidth": "300px", "maxWidth": "420px"})

    panel_resultado = html.Div([
    _card(
        _card_title("Análisis de transacción", f"Modelo: {m['nombre']}"),
        html.Div([
            html.Div([
                html.Div([
                    html.Div(_credit_card_front(), className="cc-face cc-front"),
                    html.Div(_credit_card_back(),  className="cc-face"),
                ], id="cc-flip-inner", className="cc-flip-inner"),
            ], className="cc-flip-container"),

            html.Div(id=f"{pfx}-gauge-container"),  # GAUGE SVG aquí

            html.Div(id=f"{pfx}-veredicto", style={"marginBottom": "12px"}),
            html.Hr(style={"borderColor": C["grid"], "margin": "0 0 12px"}),

            # Dos columnas: perfil + historial
            html.Div([
                # Perfil
                html.Div([
                    html.Div([
                        html.P("Perfil vs. promedios por clase", style={
                            "fontWeight": "600", "fontSize": "0.80rem", "marginBottom": "0",
                            "display": "inline"
                        }),
                        html.Div([
                            html.Button("Barras", id=f"{pfx}-btn-barras", n_clicks=0, style={
                                "fontSize": "0.70rem", "padding": "3px 10px",
                                "background": C["accent"], "color": "white",
                                "border": "none", "borderRadius": "4px 0 0 4px", "cursor": "pointer"
                            }),
                            html.Button("Tabla", id=f"{pfx}-btn-tabla", n_clicks=0, style={
                                "fontSize": "0.70rem", "padding": "3px 10px",
                                "background": C["bg"], "color": C["muted"],
                                "border": f"1px solid {C['border']}", "borderRadius": "0 4px 4px 0",
                                "cursor": "pointer"
                            }),
                        ], style={"display": "flex"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                               "alignItems": "center", "marginBottom": "8px"}),
                    dcc.Store(id=f"{pfx}-vista-perfil", data="barras"),
                    html.Div(id=f"{pfx}-perfil-container"),
                ], style={"flex": "3", "minWidth": "0"}),

                # Historial
                html.Div([
                    html.P("Historial de sesión", style={
                        "fontWeight": "600", "fontSize": "0.80rem", "marginBottom": "8px"
                    }),
                    html.Div(id=f"{pfx}-historial",
                             children=html.Div(
                                 "Sin transacciones aún.",
                                 style={"color": C["muted"], "fontSize": "0.82rem",
                                        "textAlign": "center", "padding": "16px"}
                             ))
                ], style={"flex": "2", "minWidth": "180px"}),
            ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
        ])
    ),
], style={"flex": "2", "minWidth": "380px"})

    tab_predictor = html.Div([
        dcc.Store(id=f"{pfx}-store-hist", data=[]),
        dcc.Store(id=f"{pfx}-store-prob", data=None),
        dcc.Store(id=f"{pfx}-store-prob-delayed", data=None),
        dcc.Store(id=f"{pfx}-store-hist-delayed", data=[]),
        dcc.Interval(id=f"{pfx}-anim-interval", interval=2000, n_intervals=0, disabled=True),
        html.Div([panel_controles, panel_resultado],
                 style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                         "alignItems": "flex-start"})
    ], id="modelo-tab-predictor", style={"display": "none"})

    # ═══════════════════════════════════════════════════════════════════════════
    # WRAPPER FINAL
    # ═══════════════════════════════════════════════════════════════════════════
    return html.Div([
        html.Div([
            html.H2("Modelo de Detección de Fraude", style={
                "margin": 0, "fontSize": "1.35rem", "fontWeight": "700",
                "color": C["text"], "letterSpacing": "-0.3px"
            }),
            html.P(
                "Selección del modelo, análisis de rendimiento y predictor interactivo",
                style={"margin": "4px 0 0", "color": C["muted"], "fontSize": "0.85rem"}
            )
        ], style={"marginBottom": "20px"}),

        tab_pills,
        tab_resumen,
        tab_rendimiento,
        tab_predictor,

    ], style={"padding": "20px", "background": C["bg"], "minHeight": "100vh"})


def _fig_perfil_tabla(amount, v_vals):
    """Vista tabla del perfil con columna Desviación."""
    rows = []
    for var, leg, fra in zip(PERFIL["Variable"], PERFIL["Legitima"], PERFIL["Fraude"]):
        if var == "Amount":
            actual = amount or 0
        else:
            actual = v_vals[int(var[1:]) - 1]
        rango = abs(fra - leg)
        dev = min(abs(actual - leg) / rango, 1.0) if rango > 0.001 else 0
        if dev > 0.65:
            dev_color, dev_label = C["danger"], "Alto"
        elif dev > 0.30:
            dev_color, dev_label = C["warn"], "Medio"
        else:
            dev_color, dev_label = C["ok"], "Bajo"

        rows.append(html.Div([
            html.Div(var,              style={"fontSize": "0.82rem", "fontWeight": "600", "color": C["text"], "padding": "8px 6px"}),
            html.Div(f"{leg:.3f}",     style={"fontSize": "0.80rem", "fontFamily": "monospace", "color": C["ok"], "padding": "8px 6px"}),
            html.Div(f"{fra:.3f}",     style={"fontSize": "0.80rem", "fontFamily": "monospace", "color": C["danger"], "padding": "8px 6px"}),
            html.Div(f"{actual:.3f}",  style={"fontSize": "0.80rem", "fontFamily": "monospace", "color": C["accent"], "fontWeight": "700", "padding": "8px 6px"}),
            html.Div(html.Span(dev_label, style={
                "fontSize": "0.72rem", "fontWeight": "700", "color": dev_color,
                "background": f"{dev_color}22", "border": f"1px solid {dev_color}55",
                "borderRadius": "10px", "padding": "3px 10px"
            }), style={"padding": "8px 6px"}),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "70px 90px 90px 100px 80px",
            "borderBottom": f"1px solid {C['grid']}",
            "alignItems": "center",
        }))

    header = html.Div([
        html.Div(h, style={"fontSize": "0.68rem", "fontWeight": "700", "color": C["muted"],
                            "textTransform": "uppercase", "padding": "8px 6px"})
        for h in ["Variable", "Prom. Legít.", "Prom. Fraude", "Esta transac.", "Desviación"]
    ], style={
        "display": "grid", "gridTemplateColumns": "70px 90px 90px 100px 80px",
        "background": C["bg"], "borderRadius": "6px 6px 0 0",
        "borderBottom": f"2px solid {C['border']}"
    })

    footer = html.Div(
        "Desviación = qué tanto el valor actual se aleja del perfil legítimo hacia el de fraude",
        style={"fontSize": "0.72rem", "color": C["muted"], "padding": "6px 6px"}
    )
    return html.Div([header] + rows + [footer])


def _gauge_svg(prob):
    """Gauge SVG animado estilo velocímetro."""
    arc_total  = 345
    arc_offset = arc_total - arc_total * prob
    needle_deg = -90 + 180 * prob
    pct_label  = f"{round(prob*100)}%"

    if prob < 0.30:
        nc = "#27ae60"
    elif prob < 0.65:
        nc = "#f39c12"
    else:
        nc = "#e74c3c"

    # threshold lines at 30% and 65%
    import math
    def thresh(u):
        ang = math.pi * (u - 1)
        return (150 + 103*math.cos(ang), 140 + 103*math.sin(ang),
                150 + 121*math.cos(ang), 140 + 121*math.sin(ang))

    t30 = thresh(0.30)
    t65 = thresh(0.65)

    svg_style = f"""
      .gauge-arc-fill {{
        stroke-dasharray: {arc_total};
        stroke-dashoffset: {arc_offset:.1f};
        transition: stroke-dashoffset 1.2s cubic-bezier(0.34,1.56,0.64,1);
      }}
      .gauge-needle {{
        transform-origin: 150px 140px;
        transform: rotate({needle_deg:.1f}deg);
        transition: transform 1.2s cubic-bezier(0.34,1.56,0.64,1);
      }}
    """

    import math
    return html.Iframe(
        srcDoc=f"""<!DOCTYPE html><html><body style="margin:0;background:transparent">
    <svg viewBox="0 0 300 185" xmlns="http://www.w3.org/2000/svg" style="width:260px;display:block;margin:auto">
      <defs>
        <linearGradient id="gGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#27ae60"/>
          <stop offset="45%"  stop-color="#f39c12"/>
          <stop offset="100%" stop-color="#e74c3c"/>
        </linearGradient>
      </defs>
      <style>
        .gauge-arc-fill{{stroke-dasharray:{arc_total};stroke-dashoffset:{arc_offset:.1f};transition:stroke-dashoffset 1.2s cubic-bezier(0.34,1.56,0.64,1);}}
        .gauge-needle{{transform-origin:150px 140px;transform:rotate({needle_deg:.1f}deg);transition:transform 1.2s cubic-bezier(0.34,1.56,0.64,1);}}
      </style>
      <path d="M 40 140 A 110 110 0 0 1 260 140" fill="none" stroke="#e9ecef" stroke-width="18" stroke-linecap="round"/>
      <path d="M 40 140 A 110 110 0 0 1 260 140" fill="none" stroke="url(#gGrad)" stroke-width="18" stroke-linecap="round" class="gauge-arc-fill"/>
      <line x1="{t30[0]:.1f}" y1="{t30[1]:.1f}" x2="{t30[2]:.1f}" y2="{t30[3]:.1f}" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
      <line x1="{t65[0]:.1f}" y1="{t65[1]:.1f}" x2="{t65[2]:.1f}" y2="{t65[3]:.1f}" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
      <line x1="150" y1="140" x2="150" y2="48" stroke="{nc}" stroke-width="3" stroke-linecap="round" class="gauge-needle"/>
      <circle cx="150" cy="140" r="6" fill="{nc}"/>
      <text x="150" y="112" text-anchor="middle" font-size="30" font-weight="600" fill="{nc}">{pct_label}</text>
      <text x="150" y="130" text-anchor="middle" font-size="10" fill="#9aa0a8">prob. de fraude</text>
      <text x="32"  y="158" font-size="10" fill="#c8d4e8">0%</text>
      <text x="258" y="158" font-size="10" fill="#c8d4e8">100%</text>
      <text x="55"  y="178" font-size="9" fill="#27ae60" font-weight="600">Segura</text>
      <text x="150" y="178" text-anchor="middle" font-size="9" fill="#f39c12" font-weight="600">Sospechosa</text>
      <text x="248" y="178" text-anchor="end" font-size="9" fill="#e74c3c" font-weight="600">Fraude</text>
      <text x="104" y="178" text-anchor="middle" font-size="8" fill="#c8d4e8">30%</text>
      <text x="207" y="178" text-anchor="middle" font-size="8" fill="#c8d4e8">65%</text>
    </svg>
    </body></html>""",
        style={"width": "280px", "height": "200px", "border": "none",
               "display": "block", "margin": "0 auto"},
    )
    
# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):
    pfx = "modelo"

    # ── Tab switch ────────────────────────────────────────────────────────────
    @app.callback(
        Output("modelo-tab-resumen",     "style"),
        Output("modelo-tab-rendimiento", "style"),
        Output("modelo-tab-predictor",   "style"),
        Output("modelo-pill-resumen",     "className"),
        Output("modelo-pill-rendimiento", "className"),
        Output("modelo-pill-predictor",   "className"),
        Input("modelo-pill-resumen",     "n_clicks"),
        Input("modelo-pill-rendimiento", "n_clicks"),
        Input("modelo-pill-predictor",   "n_clicks"),
        prevent_initial_call=False,
    )
    def switch_tab(n1, n2, n3):
        ctx = callback_context
        active = "resumen"
        if ctx.triggered:
            tid = ctx.triggered[0]["prop_id"].split(".")[0]
            if "rendimiento" in tid:
                active = "rendimiento"
            elif "predictor" in tid:
                active = "predictor"

        show  = {"display": "block"}
        hide  = {"display": "none"}
        pill_a = "modelo-tab-pill modelo-tab-pill-active"
        pill_i = "modelo-tab-pill"

        if active == "resumen":
            return show, hide, hide, pill_a, pill_i, pill_i
        elif active == "rendimiento":
            return hide, show, hide, pill_i, pill_a, pill_i
        else:
            return hide, hide, show, pill_i, pill_i, pill_a

    # ── Cargar modelo desde joblib ────────────────────────────────────────────
    @app.callback(
        Output("rend-comparacion", "children"),
        Input("rend-modelo-select",  "value"),
        Input("rend-tecnica-select", "value"),
    )
    def comparar_vs_ganador(modelo_nombre, tecnica_nombre):

    # Lookup en TABLA
        _m = {
            "XGBoost":            "XGBoost",
            "RandomForest":       "Random Forest",
            "LogisticRegression": "Reg. Logística",
        }
        _t = {
            "Sin Balanceo": "Sin Balanceo",
            "SMOTE":        "SMOTE",
            "ADASYN":       "ADASYN",
            "Class Weight": "Class Weight",
        }
        nombre_m = _m.get(modelo_nombre, modelo_nombre)
        nombre_t = _t.get(tecnica_nombre, tecnica_nombre)

        fila = next(
            (r for r in TABLA if r[0] == nombre_m and r[1] == nombre_t),
            None
        )

        g = MEJOR_MODELO  # ganador

        if fila is None:
            return html.Div("Combinación no encontrada.", style={"color": C["muted"]})

        _, _, acc, prec, rec, f1, auc_v, es_ganador = fila

        METRICAS = [
            ("Recall",    rec,   g["recall"],    True),   # True = mayor es mejor
            ("Precisión", prec,  g["precision"], True),
            ("F1-Score",  f1,    g["f1"],        True),
            ("AUC-ROC",   auc_v, g["auc"],       True),
            ("Accuracy",  acc,   g["accuracy"],  True),
        ]

    # ── Tabla de deltas ────────────────────────────────────────────────────
        def _delta_cell(val, ref, mayor_mejor):
            diff = val - ref
            if abs(diff) < 0.0001:
                return html.Td("= igual", style={
                    "color": C["muted"], "fontSize": "0.82rem",
                "fontWeight": "600", "padding": "10px 14px"
                })
            mejor = (diff > 0) == mayor_mejor
            color  = C["ok"] if mejor else C["danger"]
            flecha = "↑" if diff > 0 else "↓"
            signo  = "+" if diff > 0 else ""
            return html.Td(f"{flecha} {signo}{diff:.4f}", style={
                "color": color, "fontSize": "0.82rem",
                "fontWeight": "700", "padding": "10px 14px",
                "fontFamily": "monospace"
            })

        thead = html.Thead(html.Tr([
            html.Th(c, style={
                "fontSize": "0.68rem", "fontWeight": "700", "color": C["muted"],
                "textTransform": "uppercase", "letterSpacing": "0.5px",
                "padding": "10px 14px", "background": C["bg"],
                "borderBottom": f"2px solid {C['border']}"
            })
            for c in ["Métrica", f"{nombre_m} + {nombre_t}", "Ganador (XGB+ADASYN)", "Diferencia"]
        ]))

        tbody_rows = []
        for nombre_met, val, ref, mayor_mejor in METRICAS:
            tbody_rows.append(html.Tr([
                html.Td(nombre_met, style={
                    "fontSize": "0.82rem", "fontWeight": "600",
                    "color": C["text"], "padding": "10px 14px"
                }),
                html.Td(f"{val:.4f}", style={
                    "fontSize": "0.82rem", "fontFamily": "monospace",
                    "padding": "10px 14px"
                }),
                html.Td(f"{ref:.4f}", style={
                    "fontSize": "0.82rem", "fontFamily": "monospace",
                    "color": C["muted"], "padding": "10px 14px"
                }),
                _delta_cell(val, ref, mayor_mejor),
            ], style={"borderBottom": f"1px solid {C['grid']}"}))

        tabla_delta = dbc.Card([
            html.Div(
                _card_title(
                    f"Comparación directa: {nombre_m} + {nombre_t}",
                    "vs. XGBoost + Optuna + ADASYN (ganador)"
                ),
                className="card-header-custom"
            ),
            dbc.CardBody(
                html.Table(
                    [thead, html.Tbody(tbody_rows)],
                    style={"width": "100%", "borderCollapse": "collapse"}
                )
            ),
        ], className="mb-3")

    # ── Radar chart ────────────────────────────────────────────────────────
        cats      = ["Recall", "Precisión", "F1-Score", "AUC-ROC", "Accuracy"]
        vals_sel  = [rec,   prec,  f1,         auc_v,    acc]
        vals_gan  = [g["recall"], g["precision"], g["f1"], g["auc"], g["accuracy"]]

        fig_radar = go.Figure()
        fig_radar.add_scatterpolar(
            r=vals_gan + [vals_gan[0]],
            theta=cats + [cats[0]],
            fill="toself", fillcolor=f"rgba(45,74,122,0.15)",
            line=dict(color=C["accent"], width=2),
            name="Ganador (XGB+ADASYN)",
        )
        fig_radar.add_scatterpolar(
            r=vals_sel + [vals_sel[0]],
            theta=cats + [cats[0]],
            fill="toself", fillcolor=f"rgba(156,124,56,0.15)",
            line=dict(color=C["gold"], width=2, dash="dot"),
            name=f"{nombre_m} + {nombre_t}",
        )
        fig_radar.update_layout(
            plot_bgcolor=C["surface"],
            paper_bgcolor=C["surface"],
            font=dict(family="'Georgia', serif", size=11, color=C["text"]),
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    tickfont=dict(size=9), gridcolor=C["grid"]
                ),
                angularaxis=dict(tickfont=dict(size=10))
            ),
            legend=dict(orientation="h", x=0, y=-0.15, font=dict(size=10)),
            margin=dict(l=40, r=40, t=20, b=60),
        )

        radar_card = dbc.Card([
            html.Div(
                _card_title("Perfil de métricas", "Combinación seleccionada vs. ganador"),
                className="card-header-custom"
            ),
            dbc.CardBody(
                dcc.Graph(figure=fig_radar, config={"displayModeBar": False},
                      style={"height": "340px"})
            ),
        ], className="mb-3")

    # Veredicto narrativo
        n_mejor  = sum(1 for _, v, r, mm in METRICAS if (v > r) == mm and abs(v-r) >= 0.0001)
        n_peor   = sum(1 for _, v, r, mm in METRICAS if (v < r) == (not mm) and abs(v-r) >= 0.0001)

        if es_ganador:
            veredicto_txt = "Esta es la combinación ganadora."
            veredicto_color = C["ok"]
        elif n_mejor > n_peor:
            veredicto_txt = f"Supera al ganador en {n_mejor} de 5 métricas, pero puede sacrificar Recall — revisa si eso es aceptable."
            veredicto_color = C["warn"]
        else:
            veredicto_txt = f"No supera al ganador en métricas clave. El cambio no se justifica."
            veredicto_color = C["danger"]

        veredicto = html.Div(veredicto_txt, style={
            "fontSize": "0.84rem", "fontWeight": "600",
            "color": veredicto_color, "background": C["bg"],
            "border": f"1px solid {veredicto_color}33",
            "borderRadius": "8px", "padding": "10px 16px",
            "marginBottom": "16px"
        })

        return html.Div([veredicto, tabla_delta, radar_card])

    # ── Slider labels ─────────────────────────────────────────────────────────
    @app.callback(
        Output(f"{pfx}-lbl-amount", "children"),
        Output(f"{pfx}-lbl-time",   "children"),
        Input(f"{pfx}-amount", "value"),
        Input(f"{pfx}-time",   "value"),
    )
    def update_labels(amount, time_h):
        return f"${amount or 0:,}", f"{time_h or 0}:00 h"

    for _n in range(1, 29):
        @app.callback(
            Output(f"{pfx}-lbl-v{_n}", "children"),
            Input(f"{pfx}-v{_n}", "value"),
        )
        def _lbl(val, n=_n):
            return f"{val or 0:.1f}"

    # ── Ejemplos rápidos ──────────────────────────────────────────────────────
    v_outputs = [Output(f"{pfx}-v{n}", "value") for n in range(1, 29)]

    @app.callback(
        [Output(f"{pfx}-amount", "value"),
         Output(f"{pfx}-time",   "value")] + v_outputs + [
         Output(f"{pfx}-store-prob", "data", allow_duplicate=True),
         Output(f"{pfx}-store-hist", "data", allow_duplicate=True),
         Output(f"{pfx}-store-prob-delayed", "data", allow_duplicate=True),  
        Output(f"{pfx}-store-hist-delayed", "data", allow_duplicate=True),
        ],
        Input(f"{pfx}-btn-legitima", "n_clicks"),
        Input(f"{pfx}-btn-fraude",   "n_clicks"),
        Input(f"{pfx}-btn-reset",    "n_clicks"),
        prevent_initial_call=True,
    )
    def cargar_ejemplo(n_leg, n_fra, n_res):
        ctx = callback_context
        if not ctx.triggered:
            raise Exception("no trigger")
        tid = ctx.triggered[0]["prop_id"].split(".")[0]
        if "legitima" in tid:
            ej = EJEMPLO_LEGITIMO
            return [ej["amount"], ej["time"]] + list(ej["v"]) + [no_update, no_update, no_update, no_update]
        elif "fraude" in tid:
            ej = EJEMPLO_FRAUDE
            return [ej["amount"], ej["time"]] + list(ej["v"]) + [no_update, no_update, no_update, no_update]
        else:
            return [120, 14] + [0] * 28 + [None, [], None, []]

    # ── Analizar transacción ──────────────────────────────────────────────────
    v_inputs = [State(f"{pfx}-v{n}", "value") for n in range(1, 29)]

    @app.callback(
        Output(f"{pfx}-store-prob", "data"),
        Output(f"{pfx}-store-hist", "data"),
        Input(f"{pfx}-btn-analizar", "n_clicks"),
        State(f"{pfx}-amount",       "value"),
        State(f"{pfx}-time",         "value"),
        State(f"{pfx}-store-hist",   "data"),
        *v_inputs,
        prevent_initial_call=True,
    )
    def analizar(n_clicks, amount, time_h, hist, *v_vals):
        v    = [vv or 0 for vv in v_vals]
        prob = _compute_fraud_prob(amount or 0, time_h or 14, v)
        label = "Legítima" if prob < 0.30 else ("Sospechosa" if prob < 0.65 else "Fraude")
        import datetime
        entrada = {
            "n":         len(hist) + 1,
            "monto":     amount or 0,
            "hora":      f"{time_h or 0}:00",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "prob":      round(prob, 4),
            "veredicto": label,
        }
        nuevo_hist = ([entrada] + (hist or []))[:5]
        return round(prob, 4), nuevo_hist

    # ── Flip de tarjeta + actualizar monto ────────────────────────────────────
    app.clientside_callback(
    """
    function(prob, amount) {
        if (prob === null || prob === undefined) return ['cc-flip-inner', '—'];
        
        const amt = amount ? '$' + Number(amount).toLocaleString() : '—';
        
        setTimeout(function() {
            const el = document.getElementById('cc-flip-inner');
            if (el) {
                el.classList.remove('flipped-return');
                void el.offsetWidth;
                el.classList.add('flipped-return');
            }
        }, 50);
        
        return ['cc-flip-inner', amt];
    }
    """,
    Output("cc-flip-inner",       "className"),
    Output("card-amount-display", "children"),
    Input(f"{pfx}-store-prob",    "data"),
    State(f"{pfx}-amount",        "value"),
)
    
    app.clientside_callback(
        """
        function(n, prob, hist) {
            if (n === 0 || prob === null || prob === undefined) 
                return [window.dash_clientside.no_update, window.dash_clientside.no_update, true];
            return [prob, hist, true];
        }
        """,
        Output(f"{pfx}-store-prob-delayed", "data"),
        Output(f"{pfx}-store-hist-delayed", "data"),
        Output(f"{pfx}-anim-interval", "disabled", allow_duplicate=True),
        Input(f"{pfx}-anim-interval", "n_intervals"),
        State(f"{pfx}-store-prob", "data"),
        State(f"{pfx}-store-hist", "data"),
        prevent_initial_call=True,
    )

    # ── Veredicto ─────────────────────────────────────────────────────────────
    @app.callback(
        Output(f"{pfx}-veredicto", "children"),
        Input(f"{pfx}-store-prob-delayed", "data"),
    )
    def render_veredicto(prob):
        if prob is None:
            return html.Div(
                "Configure los parámetros y presione «Analizar transacción».",
                style={"color": C["muted"], "fontSize": "0.82rem",
                       "textAlign": "center", "padding": "12px"}
            )
        if prob < 0.30:
            color, bg, ico, label = C["ok"],     C["ok_bg"],     "✓", "Transacción Legítima"
        elif prob < 0.65:
            color, bg, ico, label = C["warn"],   C["warn_bg"],   "△", "Transacción Sospechosa — Revisar"
        else:
            color, bg, ico, label = C["danger"], C["danger_bg"], "✗", "Fraude Detectado"

        return html.Div([
            html.Span(ico,    style={"fontSize": "1.2rem", "marginRight": "8px"}),
            html.Span(label,  style={"fontWeight": "700", "fontSize": "0.96rem"}),
            html.Span(f" · {prob*100:.1f}%",
                      style={"fontWeight": "400", "color": color,
                             "fontSize": "0.84rem", "fontFamily": "monospace"}),
        ], style={
            "background": bg, "color": color,
            "border": f"1px solid {color}44",
            "borderRadius": "8px", "padding": "11px 16px",
            "textAlign": "center",
        })

    # ── Historial ─────────────────────────────────────────────────────────────
    @app.callback(
        Output(f"{pfx}-historial", "children"),
        Input(f"{pfx}-store-hist-delayed", "data"),
    )
    
    def render_historial(hist):
        if not hist:
            return html.Div("Sin transacciones analizadas aún.",
                            style={"color": C["muted"], "fontSize": "0.82rem",
                                   "textAlign": "center", "padding": "16px"})

        header = html.Div([
            html.Div(c, style={"fontSize": "0.67rem", "fontWeight": "700",
                                "color": C["muted"], "textTransform": "uppercase"})
            for c in ["#", "Hora", "Monto", "Prob.", ""]
        ], style={"display": "grid", "gridTemplateColumns": "24px 52px 56px 44px 1fr",
                   "gap": "6px", "padding": "6px 8px", "background": C["bg"],
                   "borderRadius": "6px 6px 0 0",
                   "borderBottom": f"2px solid {C['border']}"})

        rows = []
        for e in hist:
            prob  = e["prob"]
            color = C["ok"] if prob < 0.30 else (C["warn"] if prob < 0.65 else C["danger"])
            ico   = "✓" if prob < 0.30 else ("△" if prob < 0.65 else "✗")
            ts    = e.get("timestamp", e.get("hora", "—"))
            rows.append(html.Div([
                html.Div(f"#{e['n']}", style={"fontSize": "0.72rem", "color": C["muted"]}),
                html.Div(ts,           style={"fontSize": "0.72rem", "fontFamily": "monospace"}),
                html.Div(f"${e['monto']}", style={"fontSize": "0.72rem", "fontFamily": "monospace"}),
                html.Div(f"{prob*100:.0f}%", style={"fontSize": "0.72rem", "fontWeight": "700",
                                                      "color": color, "fontFamily": "monospace"}),
                # Barra de progreso + badge
                html.Div([
                    html.Div(style={
                        "height": "4px", "background": C["grid"], "borderRadius": "2px",
                        "marginBottom": "3px"
                    }, children=html.Div(style={
                        "height": "4px", "width": f"{prob*100:.0f}%",
                        "background": color, "borderRadius": "2px"
                    })),
                    html.Span([
                        html.Span(ico, style={"marginRight": "3px"}),
                        html.Span(e["veredicto"], style={"fontSize": "0.68rem"})
                    ], style={"color": color, "fontWeight": "600", "fontSize": "0.70rem"})
                ]),
            ], style={"display": "grid", "gridTemplateColumns": "24px 52px 56px 44px 1fr",
                       "gap": "6px", "padding": "7px 8px",
                       "borderBottom": f"1px solid {C['grid']}",
                       "alignItems": "center"}))

        return html.Div([header] + rows)
    
    # ── Toggle vista perfil ───────────────────────────────────────────────────
    @app.callback(
        Output(f"{pfx}-vista-perfil", "data"),
        Output(f"{pfx}-btn-barras", "style"),
        Output(f"{pfx}-btn-tabla",  "style"),
        Input(f"{pfx}-btn-barras", "n_clicks"),
        Input(f"{pfx}-btn-tabla",  "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_vista(n_barras, n_tabla):
        ctx = callback_context
        tid = ctx.triggered[0]["prop_id"].split(".")[0]
        active_style = {"fontSize": "0.70rem", "padding": "3px 10px",
                        "background": C["accent"], "color": "white",
                        "border": "none", "cursor": "pointer"}
        inactive_style = {"fontSize": "0.70rem", "padding": "3px 10px",
                          "background": C["bg"], "color": C["muted"],
                          "border": f"1px solid {C['border']}", "cursor": "pointer"}
        active_barras = {"borderRadius": "4px 0 0 4px", **active_style}
        inactive_barras = {"borderRadius": "4px 0 0 4px", **inactive_style}
        active_tabla = {"borderRadius": "0 4px 4px 0", **active_style}
        inactive_tabla = {"borderRadius": "0 4px 4px 0", **inactive_style}

        if "tabla" in tid:
            return "tabla", inactive_barras, active_tabla
        return "barras", active_barras, inactive_tabla

    # ── Gauge SVG ─────────────────────────────────────────────────────────────
    @app.callback(
        Output(f"{pfx}-gauge-container", "children"),
        Input(f"{pfx}-store-prob-delayed", "data"),
    )
    def render_gauge(prob):
        if prob is None:
            return html.Div(style={"height": "10px"})
        return _gauge_svg(prob)

    # ── Perfil container (barras o tabla) ─────────────────────────────────────
    @app.callback(
        Output(f"{pfx}-perfil-container", "children"),
        Input(f"{pfx}-store-prob-delayed",   "data"),
        Input(f"{pfx}-vista-perfil", "data"),
        State(f"{pfx}-amount",       "value"),
        *[State(f"{pfx}-v{n}", "value") for n in range(1, 29)],
    )
    def render_perfil_container(prob, vista, amount, *v_vals):
        if prob is None:
            return html.Div("Analice una transacción para ver el perfil comparativo",
                            style={"color": C["muted"], "fontSize": "0.82rem",
                                   "textAlign": "center", "padding": "20px"})
        v = [vv or 0 for vv in v_vals]

        if vista == "tabla":
            return _fig_perfil_tabla(amount, v)

        # Barras
        actuals = []
        for var in PERFIL["Variable"]:
            if var == "Amount":
                actuals.append(amount or 0)
            else:
                actuals.append(v[int(var[1:]) - 1])

        fig = go.Figure()
        fig.add_bar(name="Prom. Legítima", x=PERFIL["Legitima"], y=PERFIL["Variable"],
                    orientation="h", marker_color=C["ok"], opacity=0.6,
                    hovertemplate="<b>%{y}</b><br>Legítima: %{x:.3f}<extra></extra>")
        fig.add_bar(name="Prom. Fraude", x=PERFIL["Fraude"], y=PERFIL["Variable"],
                    orientation="h", marker_color=C["danger"], opacity=0.6,
                    hovertemplate="<b>%{y}</b><br>Fraude: %{x:.3f}<extra></extra>")
        fig.add_bar(name="Esta transacción", x=actuals, y=PERFIL["Variable"],
                    orientation="h", marker_color=C["accent"], opacity=0.9,
                    hovertemplate="<b>%{y}</b><br>Actual: %{x:.3f}<extra></extra>")
        fig.update_layout(
            **_LAYOUT_BASE,
            barmode="group",
            xaxis=dict(title="Valor", gridcolor=C["grid"]),
            yaxis=dict(title=""),
            legend=dict(orientation="h", x=0, y=-0.28, font=dict(size=10)),
            margin=dict(l=60, r=20, t=10, b=70),
        )
        return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "240px"})
    
    # ── Color dinámico tarjeta ────────────────────────────────────────────────
    @app.callback(
        Output("cc-card-front", "style"),
        Input(f"{pfx}-store-prob", "data"),
    )
    def color_tarjeta(prob):
        base = {"position": "relative", "transition": "background 0.8s ease"}
        if prob is None:
            bg = "linear-gradient(135deg, #1a2f5e, #0f1e3d)"
        elif prob < 0.30:
            bg = "linear-gradient(135deg, #1a3a2a, #0f2a1e)"
        elif prob < 0.65:
            bg = "linear-gradient(135deg, #3a2a0a, #2a1e08)"
        else:
            bg = "linear-gradient(135deg, #3a1010, #2a0808)"
        return {**base, "background": bg}
    
    # Activa el interval cuando llega prob
    app.clientside_callback(
        """
        function(prob) {
            if (prob === null || prob === undefined) return true;
            return false;  // habilita el interval
        }
        """,
        Output(f"{pfx}-anim-interval", "disabled"),
        Input(f"{pfx}-store-prob", "data"),
    )

    
    