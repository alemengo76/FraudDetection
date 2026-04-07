"""
tabs/resultados.py
------------------
Pestaña – Métricas del modelo de regresión logística.
"""

import os
import numpy as np
import joblib
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
)

from data.data_loader import get_df

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

C_BLUE = "#2e86c1"
C_RED  = "#c0392b"
LAYOUT_BASE = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=50, b=40, l=40, r=20),
    font=dict(family="Segoe UI, Arial", size=11),
)


#graficosss

def _get_model_results():
    if not os.path.exists(MODEL_PATH):
        return None

    model = joblib.load(MODEL_PATH)
    df    = get_df().copy()
    df["Class"] = df["Class"].astype("float64")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    _, X_test, _, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)[:, 1]
    y_pred_13 = (y_prob >= 0.13).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc         = roc_auc_score(y_test, y_prob)

    return dict(
        y_test=y_test, y_pred=y_pred, y_prob=y_prob,
        y_pred_13=y_pred_13,
        fpr=fpr, tpr=tpr, auc=auc,
        cm=confusion_matrix(y_test, y_pred),
        cm_13=confusion_matrix(y_test, y_pred_13),
        precision   =precision_score(y_test, y_pred),
        recall      =recall_score(y_test, y_pred),
        f1          =f1_score(y_test, y_pred),
        accuracy    =accuracy_score(y_test, y_pred),
        precision_13=precision_score(y_test, y_pred_13),
        recall_13   =recall_score(y_test, y_pred_13),
        f1_13       =f1_score(y_test, y_pred_13),
    )


def _fig_roc(res: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res["fpr"], y=res["tpr"],
        mode="lines", name=f"ROC (AUC = {res['auc']:.3f})",
        line=dict(color=C_BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="#aaa"),
        name="Azar",
    ))
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos (Recall)",
        yaxis=dict(gridcolor="#eee"),
        legend=dict(x=0.6, y=0.1),
        **LAYOUT_BASE,
    )
    return fig


def _fig_cm(cm: np.ndarray, title: str):
    labels = ["No fraude", "Fraude"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        colorscale=[[0, "#eaf4fb"], [1, "#1a5276"]],
        showscale=False,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicción",
        yaxis_title="Real",
        height=320,
        **LAYOUT_BASE,
    )
    return fig


#layout

def layout():
    res = _get_model_results()
    model_section = _model_unavailable() if res is None else _model_section(res)

    return html.Div([

        dbc.Row(dbc.Col(html.H5("Métricas del modelo",
                                className="section-title",
                                style={"marginTop": 0}), width=12), className="mb-4"),

        model_section,

    ], className="tab-content-wrapper")


def _model_unavailable():
    return dbc.Alert(
        "El modelo aún no ha sido entrenado. Ejecuta model/train_model.py para generar model.pkl.",
        color="warning",
    )


def _model_section(res: dict):
    return html.Div([

        # KPIs umbral 0.5
        dbc.Row([
            _kpi(f"{res['accuracy']:.3f}",  "Accuracy",  ""),
            _kpi(f"{res['precision']:.3f}", "Precision", ""),
            _kpi(f"{res['recall']:.3f}",    "Recall",    "kpi-danger"),
            _kpi(f"{res['f1']:.3f}",        "F1-Score",  "kpi-accent"),
            _kpi(f"{res['auc']:.3f}",       "ROC-AUC",   "kpi-accent"),
        ], className="mb-4"),

        # ROC + CM umbral 0.5
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div("Curva ROC", className="card-header-custom"),
                dbc.CardBody(dcc.Graph(figure=_fig_roc(res),
                                       config={"displayModeBar": False})),
            ]), md=6),
            dbc.Col(dbc.Card([
                html.Div("Matriz de confusión (umbral 0.50)", className="card-header-custom"),
                dbc.CardBody(dcc.Graph(figure=_fig_cm(res["cm"], "Umbral 0.50"),
                                       config={"displayModeBar": False})),
            ]), md=6),
        ], className="mb-4"),

        # KPIs umbral 0.13
        dbc.Row(dbc.Col(html.Div([
            html.H5("Umbral ajustado (0.13)", className="section-title"),
            html.P(
                "Reduciendo el umbral de decisión a 0.13 se incrementa el recall "
                "(detecta más fraudes), a costa de reducir la precision. Este ajuste "
                "es relevante cuando el costo de un falso negativo supera al de un "
                "falso positivo.",
                className="section-body",
            ),
        ]), width=12), className="mb-2"),

        dbc.Row([
            _kpi(f"{res['precision_13']:.3f}", "Precision (0.13)", ""),
            _kpi(f"{res['recall_13']:.3f}",    "Recall (0.13)",    "kpi-danger"),
            _kpi(f"{res['f1_13']:.3f}",        "F1 (0.13)",        "kpi-accent"),
        ], className="mb-4"),

        dbc.Row(dbc.Col(dbc.Card([
            html.Div("Matriz de confusión (umbral 0.13)", className="card-header-custom"),
            dbc.CardBody(dcc.Graph(
                figure=_fig_cm(res["cm_13"], "Umbral 0.13"),
                config={"displayModeBar": False},
                style={"height": "320px"},
            )),
        ]), md=6)),
    ])


def _kpi(value: str, label: str, accent: str) -> dbc.Col:
    return dbc.Col(html.Div([
        html.Div(value, className=f"kpi-value {accent}"),
        html.Div(label, className="kpi-label"),
    ], className="kpi-card"), md=2, className="mb-3")