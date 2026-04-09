"""
tabs/eda.py
-----------
Pestaña – Análisis Exploratorio de Datos.
Univariado, Bivariado, Análisis Temporal y Estructura de Variables.
"""

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache

from data.data_loader import get_df

C_BLUE = "#2e86c1"
C_RED = "#c0392b"
LAYOUT_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=50, b=40, l=40, r=20),
    font=dict(family="Segoe UI, Arial", size=11),
)

# Skewness y Kurtosis pre-calculados
SKEW_KURT = {
    "V1": {"skew": -2.07, "kurt": 10.72},
    "V2": {"skew": 1.63, "kurt": 13.31},
    "V3": {"skew": -3.27, "kurt": 42.35},
    "V4": {"skew": 2.79, "kurt": 45.27},
    "V5": {"skew": -0.79, "kurt": 25.17},
    "V6": {"skew": 0.12, "kurt": 8.07},
    "V7": {"skew": 0.47, "kurt": 32.57},
    "V8": {"skew": -0.30, "kurt": 20.43},
    "V9": {"skew": -0.04, "kurt": 7.96},
    "V10": {"skew": -0.41, "kurt": 17.00},
    "V11": {"skew": 0.03, "kurt": 4.64},
    "V12": {"skew": -0.48, "kurt": 10.97},
    "V13": {"skew": 0.07, "kurt": 4.73},
    "V14": {"skew": -0.56, "kurt": 11.99},
    "V15": {"skew": 0.07, "kurt": 4.58},
    "V16": {"skew": -0.39, "kurt": 9.05},
    "V17": {"skew": -0.67, "kurt": 28.29},
    "V18": {"skew": -0.12, "kurt": 8.36},
    "V19": {"skew": 0.04, "kurt": 5.05},
    "V20": {"skew": -0.07, "kurt": 20.19},
    "V21": {"skew": 0.45, "kurt": 21.35},
    "V22": {"skew": 0.01, "kurt": 5.50},
    "V23": {"skew": -0.41, "kurt": 26.65},
    "V24": {"skew": 0.06, "kurt": 5.40},
    "V25": {"skew": 0.10, "kurt": 5.85},
    "V26": {"skew": 0.05, "kurt": 3.65},
    "V27": {"skew": 4.03, "kurt": 87.31},
    "V28": {"skew": 5.61, "kurt": 96.75},
    "Amount": {"skew": 16.98, "kurt": 844.47},
    "Time": {"skew": -0.04, "kurt": -1.29},
}

# RBC pre-calculados
RBC_DATA = [
    {"Variable": "V1", "RBC": -0.579, "p": 2.571e-105},
    {"Variable": "V2", "RBC": 0.700, "p": 8.275e-153},
    {"Variable": "V3", "RBC": -0.817, "p": 8.739e-208},
    {"Variable": "V4", "RBC": 0.873, "p": 1.599e-236},
    {"Variable": "V5", "RBC": -0.400, "p": 2.830e-51},
    {"Variable": "V6", "RBC": -0.537, "p": 5.344e-91},
    {"Variable": "V7", "RBC": -0.659, "p": 6.513e-136},
    {"Variable": "V8", "RBC": 0.325, "p": 2.369e-34},
    {"Variable": "V9", "RBC": -0.683, "p": 1.484e-145},
    {"Variable": "V10", "RBC": -0.821, "p": 9.598e-210},
    {"Variable": "V11", "RBC": 0.830, "p": 4.678e-214},
    {"Variable": "V12", "RBC": -0.869, "p": 1.358e-234},
    {"Variable": "V13", "RBC": -0.044, "p": 9.743e-2},
    {"Variable": "V14", "RBC": -0.894, "p": 2.305e-248},
    {"Variable": "V15", "RBC": -0.029, "p": 2.674e-1},
    {"Variable": "V16", "RBC": -0.683, "p": 1.111e-145},
    {"Variable": "V17", "RBC": -0.601, "p": 2.854e-113},
    {"Variable": "V18", "RBC": -0.466, "p": 5.453e-69},
    {"Variable": "V19", "RBC": 0.313, "p": 5.230e-32},
    {"Variable": "V20", "RBC": 0.313, "p": 4.106e-32},
    {"Variable": "V21", "RBC": 0.486, "p": 6.872e-75},
    {"Variable": "V22", "RBC": 0.035, "p": 1.914e-1},
    {"Variable": "V23", "RBC": -0.082, "p": 2.080e-3},
    {"Variable": "V24", "RBC": -0.129, "p": 1.255e-6},
    {"Variable": "V25", "RBC": 0.064, "p": 1.544e-2},
    {"Variable": "V26", "RBC": 0.075, "p": 4.573e-3},
    {"Variable": "V27", "RBC": 0.403, "p": 4.334e-52},
    {"Variable": "V28", "RBC": 0.293, "p": 3.174e-28},
    {"Variable": "Amount", "RBC": -0.112, "p": 2.686e-5},
    {"Variable": "Time", "RBC": -0.169, "p": 2.233e-10},
]

RBC_MAP = {
    "V1": {"U": 28197486.0, "p": 2.571e-105, "RBC": -0.579},
    "V2": {"U": 113855524.0, "p": 8.275e-153, "RBC": 0.700},
    "V3": {"U": 12240264.0, "p": 8.739e-208, "RBC": -0.817},
    "V4": {"U": 125438246.0, "p": 1.599e-236, "RBC": 0.873},
    "V5": {"U": 40179560.0, "p": 2.830e-51, "RBC": -0.400},
    "V6": {"U": 30983681.0, "p": 5.344e-91, "RBC": -0.537},
    "V7": {"U": 22826763.0, "p": 6.513e-136, "RBC": -0.659},
    "V8": {"U": 88742548.0, "p": 2.369e-34, "RBC": 0.325},
    "V9": {"U": 21264298.0, "p": 1.484e-145, "RBC": -0.683},
    "V10": {"U": 11980128.0, "p": 9.598e-210, "RBC": -0.821},
    "V11": {"U": 122566787.0, "p": 4.678e-214, "RBC": 0.830},
    "V12": {"U": 8781437.0, "p": 1.358e-234, "RBC": -0.869},
    "V13": {"U": 64039341.0, "p": 9.743e-2, "RBC": -0.044},
    "V14": {"U": 7082225.0, "p": 2.305e-248, "RBC": -0.894},
    "V15": {"U": 65015403.0, "p": 2.674e-1, "RBC": -0.029},
    "V16": {"U": 21244264.0, "p": 1.111e-145, "RBC": -0.683},
    "V17": {"U": 26732456.0, "p": 2.854e-113, "RBC": -0.601},
    "V18": {"U": 35744597.0, "p": 5.453e-69, "RBC": -0.466},
    "V19": {"U": 87947406.0, "p": 5.230e-32, "RBC": 0.313},
    "V20": {"U": 87983679.0, "p": 4.106e-32, "RBC": 0.313},
    "V21": {"U": 99578153.0, "p": 6.872e-75, "RBC": 0.486},
    "V22": {"U": 69314702.0, "p": 1.914e-1, "RBC": 0.035},
    "V23": {"U": 61509979.0, "p": 2.080e-3, "RBC": -0.082},
    "V24": {"U": 58362910.0, "p": 1.255e-6, "RBC": -0.129},
    "V25": {"U": 71299741.0, "p": 1.544e-2, "RBC": 0.064},
    "V26": {"U": 72036301.0, "p": 4.573e-3, "RBC": 0.075},
    "V27": {"U": 94018959.0, "p": 4.334e-52, "RBC": 0.403},
    "V28": {"U": 86597204.0, "p": 3.174e-28, "RBC": 0.293},
    "Amount": {"U": 59517129.5, "p": 2.686e-5, "RBC": -0.112},
    "Time": {"U": 55697454.0, "p": 2.233e-10, "RBC": -0.169},
}

# Grupos discriminativos
ALTO = {"V2", "V3", "V4", "V7", "V9", "V10", "V11", "V12", "V14", "V16", "V17"}
MODERADO = {"V1", "V5", "V6", "V8", "V18", "V19", "V20", "V21", "V27"}
BAJO = {"V13", "V15", "V22", "V23", "V24", "V25", "V26", "V28", "Amount", "Time"}

# Listas de variables por sección
VARS_NUM_UNI = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount", "log_Amount", "Time", "Class",
]

VARS_NUM_BIV = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount", "log_Amount", "Time",
]

# VIF pre-calculados
VIF_DATA = [
    {"Variable": "Time", "VIF": 1.883851},
    {"Variable": "V1", "VIF": 1.656669},
    {"Variable": "V2", "VIF": 4.467685},
    {"Variable": "V3", "VIF": 1.879467},
    {"Variable": "V4", "VIF": 1.141301},
    {"Variable": "V5", "VIF": 2.866851},
    {"Variable": "V6", "VIF": 1.578711},
    {"Variable": "V7", "VIF": 2.923828},
    {"Variable": "V8", "VIF": 1.134124},
    {"Variable": "V9", "VIF": 1.024321},
    {"Variable": "V10", "VIF": 1.125354},
    {"Variable": "V11", "VIF": 1.115713},
    {"Variable": "V12", "VIF": 1.030677},
    {"Variable": "V13", "VIF": 1.008479},
    {"Variable": "V14", "VIF": 1.032034},
    {"Variable": "V15", "VIF": 1.064241},
    {"Variable": "V16", "VIF": 1.000915},
    {"Variable": "V17", "VIF": 1.011412},
    {"Variable": "V18", "VIF": 1.031429},
    {"Variable": "V19", "VIF": 1.039877},
    {"Variable": "V20", "VIF": 2.406956},
    {"Variable": "V21", "VIF": 1.140172},
    {"Variable": "V22", "VIF": 1.089913},
    {"Variable": "V23", "VIF": 1.164070},
    {"Variable": "V24", "VIF": 1.000816},
    {"Variable": "V25", "VIF": 1.131789},
    {"Variable": "V26", "VIF": 1.003516},
    {"Variable": "V27", "VIF": 1.011060},
    {"Variable": "V28", "VIF": 1.001726},
    {"Variable": "Amount", "VIF": 12.296464},
]


# Preparación del DataFrame
def _preparar_df(df: pd.DataFrame):
    df2 = df.copy()
    df2["log_Amount"] = np.log1p(df2["Amount"])
    df2["Class"] = df2["Class"].astype(str)
    return df2


_DF = _preparar_df(get_df())


# Figuras estáticas pre-calculadas
def _build_fig_rbc():
    df_rbc = pd.DataFrame(RBC_DATA).sort_values("RBC")
    colors = [C_RED if v < 0 else C_BLUE for v in df_rbc["RBC"]]
    fig = go.Figure(
        go.Bar(
            x=df_rbc["RBC"],
            y=df_rbc["Variable"],
            orientation="h",
            marker_color=colors,
            text=df_rbc["RBC"].round(3),
            textposition="outside",
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#27ae60", line_width=1.5)
    fig.add_vline(x=-0.5, line_dash="dash", line_color="#27ae60", line_width=1.5)
    fig.update_layout(
        title="Coeficiente RBC por variable (capacidad discriminativa)",
        xaxis_title="RBC",
        xaxis=dict(range=[-1.1, 1.1], gridcolor="#eee"),
        height=700,
        **LAYOUT_BASE,
    )
    return fig


def _build_fig_correlacion(df: pd.DataFrame):
    num_cols = [c for c in df.columns if c not in ("Class", "log_Amount")]
    corr = df[num_cols].corr(method="spearman")
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(title="Spearman"),
        )
    )
    fig.update_layout(
        height=550,
        xaxis=dict(tickangle=45),
        **LAYOUT_BASE,
    )
    return fig


def _build_fig_vif():
    df_vif = pd.DataFrame(VIF_DATA).sort_values("VIF", ascending=True)
    colors = [
        C_RED if v >= 10 else "#f39c12" if v >= 5 else C_BLUE for v in df_vif["VIF"]
    ]
    fig = go.Figure(
        go.Bar(
            x=df_vif["VIF"],
            y=df_vif["Variable"],
            orientation="h",
            marker_color=colors,
            text=df_vif["VIF"].round(2),
            textposition="outside",
        )
    )
    fig.add_vline(x=5, line_dash="dash", line_color="#f39c12", line_width=1.5)
    fig.add_vline(x=10, line_dash="dash", line_color=C_RED, line_width=1.5)
    fig.update_layout(
        title="Factor de Inflación de la Varianza (VIF)",
        xaxis_title="VIF",
        xaxis=dict(range=[0, 14], gridcolor="#eee"),
        height=700,
        **LAYOUT_BASE,
    )
    return fig


def _build_fig_temporal():
    df2 = _DF.copy()
    df2["Hour"] = (df2["Time"] // 3600) % 24
    tiempo = df2.groupby(["Hour", "Class"])["Amount"].median().reset_index()
    tiempo["Tipo"] = tiempo["Class"].map({"0": "No fraude", "1": "Fraude"})
    fig = px.line(
        tiempo, x="Hour", y="Amount", color="Tipo", markers=True,
        color_discrete_map={"No fraude": C_BLUE, "Fraude": C_RED},
        labels={
            "Hour": "Hora relativa al inicio del registro",
            "Amount": "Mediana del monto de la transacción",
            "Tipo": "Tipo de transacción",
        },
    )
    fig.update_layout(height=420, **LAYOUT_BASE)
    return fig


# Pre-calcular matrices de correlación
_num_cols = [c for c in _DF.columns if c not in ("Class", "log_Amount")]
_CORR_0 = _DF[_DF["Class"] == "0"][_num_cols].corr(method="spearman")
_CORR_1 = _DF[_DF["Class"] == "1"][_num_cols].corr(method="spearman")


def _build_fig_correlacion_clase(corr: pd.DataFrame, titulo: str):
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        showscale=True, colorbar=dict(title="Spearman"),
    ))
    fig.update_layout(
        title=titulo,
        height=500,
        xaxis=dict(tickangle=45),
        **LAYOUT_BASE,
    )
    return fig


_FIG_CORRELACION_0 = _build_fig_correlacion_clase(_CORR_0, "Transacciones NO fraudulentas")
_FIG_CORRELACION_1 = _build_fig_correlacion_clase(_CORR_1, "Transacciones fraudulentas")
_FIG_TEMPORAL = _build_fig_temporal()
_FIG_RBC = _build_fig_rbc()
_FIG_CORRELACION = _build_fig_correlacion(_DF)
_FIG_VIF = _build_fig_vif()


# Figuras dinámicas
@lru_cache(maxsize=None)
def _fig_boxplot_uni(var: str):
    fig = px.box(
        _DF,
        y=var,
        color_discrete_sequence=[C_BLUE],
        points=False,
        boxmode="overlay",
    )
    fig.update_layout(
        title=f"Distribución de {var}",
        yaxis=dict(showgrid=True, gridcolor="lightgrey"),
        showlegend=False,
        **LAYOUT_BASE,
    )
    fig.update_traces(boxmean=True)
    return fig


@lru_cache(maxsize=None)
def _fig_kde_uni(var: str):
    vals = _DF[var].values
    fig = go.Figure(
        go.Histogram(
            x=vals,
            histnorm="probability density",
            marker_color=C_BLUE,
            opacity=0.7,
            nbinsx=80,
            name=var,
        )
    )
    fig.update_layout(
        title=f"Distribución de {var}",
        xaxis_title=var,
        yaxis_title="Densidad",
        yaxis=dict(gridcolor="#eee", tickformat=".6f"),
        **LAYOUT_BASE,
    )
    return fig


@lru_cache(maxsize=None)
def _fig_bar_class():
    fig = go.Figure(
        go.Bar(
            x=["No fraude (0)", "Fraude (1)"],
            y=[283253, 473],
            marker_color=[C_BLUE, C_RED],
            text=["283.253 (99.83%)", "473 (0.17%)"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Distribución de Class",
        yaxis=dict(showgrid=True, gridcolor="lightgrey", title="Frecuencia"),
        showlegend=False,
        **LAYOUT_BASE,
    )
    return fig


@lru_cache(maxsize=None)
def _fig_violin_biv(var: str):
    fig = px.violin(
        _DF,
        x="Class",
        y=var,
        color="Class",
        color_discrete_map={"0": C_BLUE, "1": C_RED},
        box=True,
        points=False,
    )
    fig.update_layout(
        title=f"Distribución de {var} por clase",
        yaxis=dict(showgrid=True, gridcolor="lightgrey"),
        xaxis=dict(tickvals=["0", "1"], ticktext=["No fraude", "Fraude"]),
        showlegend=False,
        **LAYOUT_BASE,
    )
    fig.update_traces(
        meanline_visible=True,
        meanline=dict(color="#1a2540", width=2),
        box=dict(line_color="#1a2540"),
    )
    return fig


@lru_cache(maxsize=None)
def _fig_kde_biv_time():
    fig = go.Figure()
    for cls, label, color in [("0", "No fraude", C_BLUE), ("1", "Fraude", C_RED)]:
        vals = _DF[_DF["Class"] == cls]["Time"].values
        fig.add_trace(
            go.Histogram(
                x=vals,
                histnorm="probability density",
                name=label,
                marker_color=color,
                opacity=0.6,
                nbinsx=80,
            )
        )
    fig.update_layout(
        barmode="overlay",
        title="Distribución de Time por clase",
        xaxis_title="Time (segundos)",
        yaxis_title="Densidad",
        yaxis=dict(gridcolor="#eee", tickformat=".6f"),
        legend=dict(x=0.75, y=0.95),
        **LAYOUT_BASE,
    )
    return fig


for _v in VARS_NUM_UNI:
    if _v not in ("Class", "log_Amount", "Time"):
        _fig_boxplot_uni(_v)
for _v in VARS_NUM_BIV:
    if _v not in ("Time", "log_Amount"):
        _fig_violin_biv(_v)
_fig_kde_biv_time()
_fig_kde_uni("Time")
_fig_bar_class()


# Funciones auxiliares
def _grupo_badge(var: str):
    if var in ALTO:
        label, color = "Alto poder discriminante", "#27ae60"
    elif var in MODERADO:
        label, color = "Moderado poder discriminante", "#f39c12"
    else:
        label, color = "Bajo poder discriminante", "#c0392b"
    return html.Span(
        label,
        style={
            "backgroundColor": color,
            "color": "white",
            "padding": "3px 10px",
            "borderRadius": "12px",
            "fontSize": "0.78rem",
            "fontWeight": "600",
        },
    )


def _kpi_small(value, label, id=None):
    div_props = {"className": "kpi-value"}
    if id:
        div_props["id"] = id
    return dbc.Col(
        html.Div(
            [
                html.Div(value, **div_props),
                html.Div(label, className="kpi-label"),
            ],
            className="kpi-card",
        ),
        className="mb-2",
    )


# Interpretaciones
def _interpretacion_uni(var: str):
    if var in SKEW_KURT and var not in ("Amount", "Time"):
        return html.P(
            "En los boxplots de las variables V1-V28 se nota un patrón que es consistente con la naturaleza de este tipo de variables (PCA). Las primeras componentes (V1,V2,V3) son las que tienen la mayor dispersión, valores extremos y alta asimetría, mientras que las siguientes tienen distribuciones relativamente más centradas en cero. Las variables que tienen kurtosisi alta  (V1, V2, V5, V7, V8, V20, V21, V23, V28) tienen colas muy pesadas y un número mucho mayor de valores extremos, por otro lado, variables como V11, V13, V15, V18 y V19 tienen distribuciones más simétricas y mesocúrticas.",
            className="section-body",
        )
    if var == "Time":
        return html.P(
            "La variable Time tiene un promedio de 94,811.08 s (DE = 47,481.05 s), con una mediana de 84,692.50 s e IQR de 85,093.25 s. El rango va de 0 a 172,792 s, correspondiente exactamente a los dos días de observación. En el gráfico se ve una concentración en las transacciones al inicio y al final del intervalo temporal, con una menor densidad en la parte intermedia, que puede ser el final del primer día. Se observa una distribución más plana con colas más ligeras que una normal, por lo que los tiempos se distribuyen de manera relativamente homogénea a lo largo de los dos días.",
            className="section-body",
        )
    if var == "Amount":
        return html.P(
            "El monto promedio transferido fue de 88.47 euros (DE = 250.39), con un rango de 0 a 25,691.16. El 50% de las transacciones estuvo por debajo de 22 euros (IQR = 71.91). Se evidencia esta gran asimetría positiva que se concentra en montos bajos y tiene una cola larga a valores altos. Con el fin de tener una mejor visualización, se puede usar log_Amount.",
            className="section-body",
        )
    if var == "log_Amount":
        return html.P(
            "Con la transformación logarítmica, se puede ver claramente la distribución de los datos. No obstante, todavía hay valores atípicos; sin embargo, la media y mediana son similares, lo que dice que los valores extremos no están influyendo de manera tan significativa en la tendencia central de la distribución.",
            className="section-body",
        )
    if var == "Class":
        return html.P(
            "Se observa un desbalance extremo en la variable respuesta Class, pues con un total de 283.726 transacciones, 283.253 (99.83%) corresponden a transacciones no fraudulentas (Clase 0) y 473 (0.17%) corresponden a transacciones fraudulentas (Clase 1).",
            className="section-body",
        )
    return html.Div()


def _interpretacion_biv(var: str):
    if var in SKEW_KURT and var not in ("Amount", "log_Amount", "Time"):
        return html.P(
            "Los violin plots confirman visualmente la separación entre clases. En las componentes con alto "
            "poder discriminante no hay solapamiento entre las distribuciones de Fraude y No Fraude. "
            "La mediana en No Fraude (clase 0) es aproximadamente 0 en todas las componentes, mientras que "
            "en Fraude (clase 1) toma valores positivos o negativos según la componente. Las cajas del grupo "
            "Fraude son notablemente más amplias, lo que indica mayor variabilidad en las transacciones "
            "fraudulentas. En ambos grupos se evidencia una gran cantidad de valores atípicos, especialmente "
            "en No Fraude, lo cual es esperable dado el fuerte desbalance de clases.",
            className="section-body",
        )
    if var == "Time":
        return html.P(
            "Las transacciones fraudulentas (n = 473) tienen un tiempo promedio de 80,450.51 s "
            "(DE = 48,636.18 s), con mediana de 73,408 s (IQR = 87,892 s). Las no fraudulentas "
            "(n = 283,253) presentan un tiempo promedio mayor (94,835.06 s, DE = 47,475.55 s) con mediana "
            "de 84,711 s. Ambas distribuciones tienen gran dispersión y solapamiento en prácticamente todo "
            "el intervalo temporal. Aunque la diferencia es estadísticamente significativa (p < 0.05), el "
            "RBC = -0.169 indica un efecto de magnitud pequeña: el momento de la transacción no distingue "
            "de manera marcada entre fraude y no fraude.",
            className="section-body",
        )
    if var == "Amount":
        return html.P(
            "La mediana en transacciones fraudulentas es de 9.82 euros frente a 22 euros en no fraudulentas, "
            "lo que sugiere que el fraude típico no ocurre en transacciones de alto valor. La media en fraude "
            "(123.87, DE = 260.21) es superior a la de no fraude (88.41, DE = 250.38), diferencia influenciada "
            "por valores extremos. El IQR en fraude (1–105.89) presenta solapamiento considerable con el de "
            "no fraude (5.67–77.46), confirmando baja capacidad discriminativa. Aunque la diferencia es "
            "estadísticamente significativa (U = 59,517,129.5, p ≈ 0.003), el RBC = -0.112 indica que los "
            "grupos están muy solapados.",
            className="section-body",
        )
    if var == "log_Amount":
        return html.P(
            "Debido a la alta asimetría de la variable Amount, se aplicó una transformación logarítmica para visualizar de manera más clara la distribución de los datos. En este caso, se sigue evidenciando que la mediana en las transacciones no fraudulentas es mayor que en la de transacciones fraudulentas. A simple vista, se observa un considerable solapamiento entre ambas cajas, lo que reafirma más que Amount no separa fuertemente entre clases. Por otro lado, la clase Fraude parece más disperso hacia valores bajos, mientras que No fraude parece que tiene valores intermedios.",
            className="section-body",
        )
    return html.Div()


def _kpis_biv(var: str):
    var_lookup = "Amount" if var == "log_Amount" else var
    d = RBC_MAP.get(var_lookup)
    if not d:
        return html.Div()

    u = d["U"]
    rbc = d["RBC"]
    p = d["p"]
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    u_str = f"{u:,.0f}"

    return dbc.Row(
        [
            _kpi_small(u_str, "Estadístico U"),
            _kpi_small(p_str, "p-valor"),
            _kpi_small(f"{rbc:.3f}", "RBC"),
            _kpi_small(_grupo_badge(var_lookup), "Poder discriminante"),
        ],
        className="mb-3",
    )




def _panel_univariado():
    """Contenido del tab Análisis Univariado."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label(
                    "Selecciona una variable:",
                    style={"fontSize": "0.85rem", "fontWeight": "600", "color": "#1a2540"},
                ),
                dcc.Dropdown(
                    id="uni-var-selector",
                    options=[{"label": v, "value": v} for v in VARS_NUM_UNI],
                    value="V1",
                    clearable=False,
                    style={"fontSize": "0.85rem"},
                ),
            ], md=4),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("", className="kpi-value", id="uni-kpi-skew"),
                    html.Div("", className="kpi-label", id="uni-kpi-skew-label"),
                ], className="kpi-card"),
                className="mb-2",
            ),
            dbc.Col(
                html.Div([
                    html.Div("", className="kpi-value", id="uni-kpi-kurt"),
                    html.Div("", className="kpi-label", id="uni-kpi-kurt-label"),
                ], className="kpi-card"),
                className="mb-2",
            ),
        ], className="mb-3"),

        dbc.Row(
            dbc.Col(
                dbc.Card([
                    html.Div(id="uni-graph-title", className="card-header-custom"),
                    dbc.CardBody(dcc.Graph(id="uni-graph", config={"displayModeBar": False})),
                ]),
                width=12,
            ),
            className="mb-4",
        ),

        html.Div(id="uni-interpretacion", className="mb-2"),
    ])


def _panel_bivariado():
    """Contenido del tab Análisis Bivariado."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label(
                    "Selecciona una variable:",
                    style={"fontSize": "0.85rem", "fontWeight": "600", "color": "#1a2540"},
                ),
                dcc.Dropdown(
                    id="biv-var-selector",
                    options=[{"label": v, "value": v} for v in VARS_NUM_BIV],
                    value="V1",
                    clearable=False,
                    style={"fontSize": "0.85rem"},
                ),
            ], md=4),
        ], className="mb-3"),

        html.Div(id="biv-kpis", className="mb-3"),

        dbc.Row(
            dbc.Col(
                dbc.Card([
                    html.Div(id="biv-graph-title", className="card-header-custom"),
                    dbc.CardBody(dcc.Graph(id="biv-graph", config={"displayModeBar": False})),
                ]),
                width=12,
            ),
            className="mb-4",
        ),

        html.Div(id="biv-interpretacion", className="mb-2"),
    ])


def _panel_temporal():
    """Contenido del tab Análisis Temporal."""
    return html.Div([
        dbc.Row(
            dbc.Col(
                dbc.Card([
                    html.Div("Análisis temporal de las transacciones", className="card-header-custom"),
                    dbc.CardBody([
                        html.P(
                            "El análisis exploratorio previo permite plantear una pregunta adicional: "
                            "¿tienden a concentrarse montos más elevados en ciertos intervalos de tiempo "
                            "o el comportamiento del monto es independiente del tiempo en estos dos días? "
                            "Para este análisis se usa la mediana de Amount debido al gran sesgo de esta variable.",
                            className="section-body",
                        ),
                        dcc.Graph(figure=_FIG_TEMPORAL, config={"displayModeBar": False}),
                        html.P(
                            "Note que Time representa los segundos transcurridos desde la primera transacción, luego, "
                            "las horas son relativas al inicio del registro y no corresponden necesariamente a horas "
                            "del día reales. Con esto en mente, a simple vista parece que en las horas relativas 0, 6 "
                            "y 14 las medianas de los montos de las transacciones fraudulentas son muy elevadas, y "
                            "presentan una alta variabilidad, con picos pronunciados, lo que implica que estos no "
                            "siguen un patrón uniforme y pueden concentrarse en intervalos específicos con valores "
                            "elevados. En contraste, el monto en las transacciones no fraudulentas presenta un "
                            "comportamiento relativamente estable a lo largo del tiempo. No obstante, en el futuro "
                            "se realizarán pruebas estadísticas para verificar si las diferencias son a causa de un "
                            "patrón significativo o se atribuyen a la variabilidad aleatoria derivada del tamaño "
                            "muestral reducido en la clase minoritaria.",
                            className="section-body",
                        ),
                    ]),
                ]),
                width=12,
            ),
            className="mb-4",
        ),
    ])


def _panel_estructura():
    """Contenido del tab Estructura de Variables (correlación + VIF + RBC)."""
    return html.Div([
        # Correlación
        html.H6(
            "Correlación",
            style={
                "fontWeight": "700",
                "borderLeft": f"4px solid {C_BLUE}",
                "paddingLeft": "12px",
                "marginBottom": "12px",
                "color": "#1a2540",
            },
        ),

        dbc.Row([
            dbc.Col([
                html.Label(
                    "Selecciona una vista:",
                    style={"fontSize": "0.85rem", "fontWeight": "600", "color": "#1a2540"},
                ),
                dcc.Dropdown(
                    id="corr-selector",
                    options=[
                        {"label": "General", "value": "general"},
                        {"label": "Por clase (Fraude vs No fraude)", "value": "clases"},
                    ],
                    value="general",
                    clearable=False,
                    style={"fontSize": "0.85rem"},
                ),
            ], md=4),
        ], className="mb-3"),

        dbc.Row(
            dbc.Col(
                dbc.Card([
                    html.Div("Matriz de correlación Spearman", className="card-header-custom"),
                    dbc.CardBody(html.Div(id="corr-content")),
                ]),
                width=12,
            ),
            className="mb-3",
        ),

        html.Div(id="corr-interpretacion", className="mb-4"),

        # VIF
        html.H6(
            "Factor de Inflación de la Varianza (VIF)",
            style={
                "fontWeight": "700",
                "borderLeft": f"4px solid {C_BLUE}",
                "paddingLeft": "12px",
                "marginBottom": "12px",
                "color": "#1a2540",
            },
        ),

        html.P(
            "El Factor de Inflación de la Varianza (VIF) mide la multicolinealidad entre las variables "
            "explicativas. Un VIF ≥ 5 indica multicolinealidad moderada y un VIF ≥ 10 indica multicolinealidad "
            "severa. En este dataset, solo Amount tiene un VIF mayor a 10 (VIF = 12.30), mientras que "
            "el resto de variables tienen valores bajos, lo que sugiere que no hay multicolinealidad problemática.",
            className="section-body",
        ),

        dbc.Row(
            dbc.Col(
                dbc.Card([
                    html.Div("VIF por variable", className="card-header-custom"),
                    dbc.CardBody(dcc.Graph(figure=_FIG_VIF, config={"displayModeBar": False})),
                ]),
                width=12,
            ),
            className="mb-4",
        ),

        # RBC
        html.H6(
            "Capacidad discriminativa (RBC)",
            style={
                "fontWeight": "700",
                "borderLeft": f"4px solid {C_BLUE}",
                "paddingLeft": "12px",
                "marginBottom": "12px",
                "color": "#1a2540",
            },
        ),

        html.P(
            "El coeficiente RBC (Rank Biserial Correlation) mide qué tan bien separa "
            "cada variable las transacciones fraudulentas de las no fraudulentas. "
            "Las líneas verdes marcan el umbral |RBC| > 0.5, que indica alta capacidad discriminativa. "
            "Las 5 variables con mayor capacidad discriminativa son: V14, V4, V12, V11 y V10",
            className="section-body",
        ),

        dbc.Row(
            dbc.Col(
                dbc.Card([
                    html.Div("RBC por variable", className="card-header-custom"),
                    dbc.CardBody(dcc.Graph(figure=_FIG_RBC, config={"displayModeBar": False})),
                ]),
                width=12,
            ),
            className="mb-4",
        ),
    ])




def layout():
    return html.Div(
        [
            # Título de la pestaña
            dbc.Row(
                dbc.Col(
                    html.Div([
                        html.H5(
                            "Análisis Exploratorio de Datos",
                            className="section-title",
                            style={"marginTop": 0},
                        ),
                        html.P(
                            "Para ver de mejor manera la distribución de Amount, se recomienda seleccionar "
                            "log_Amount en el análisis bivariado, pues esta tiene una alta asimetría que no "
                            "permite comparar las clases.",
                            className="section-body",
                        ),
                    ]),
                    width=12,
                ),
                className="mb-4",
            ),

            # Navbar con tabs (equivalente al navset_card_tab de Shiny)
            dbc.Card(
                [
                    dbc.CardHeader(
                        dbc.Tabs(
                            [
                                dbc.Tab(label="Análisis Univariado",   tab_id="tab-uni"),
                                dbc.Tab(label="Análisis Bivariado",    tab_id="tab-biv"),
                                dbc.Tab(label="Análisis Temporal",     tab_id="tab-temporal"),
                                dbc.Tab(label="Diagnóstico multivariado", tab_id="tab-estructura"),
                            ],
                            id="eda-tabs",
                            active_tab="tab-uni",
                            className="custom-tabs",
                            style={"borderBottom": "none"},
                        )
                    ),
                    dbc.CardBody(html.Div(id="eda-tab-content")),
                ],
                className="mb-4",
                # Borde superior de acento en azul, igual que las section-title
                style={"border": "1px solid #dee2e6", "borderTop": f"3px solid {C_BLUE}"},
            ),
        ],
        className="tab-content-wrapper tab-fade-in",
    )



def register_callbacks(app):

    # Renderiza el panel activo
    @app.callback(
        Output("eda-tab-content", "children"),
        Input("eda-tabs", "active_tab"),
    )
    def render_tab(active_tab):
        if active_tab == "tab-uni":
            return _panel_univariado()
        if active_tab == "tab-biv":
            return _panel_bivariado()
        if active_tab == "tab-temporal":
            return _panel_temporal()
        if active_tab == "tab-estructura":
            return _panel_estructura()
        return html.Div()

    # Univariado
    @app.callback(
        Output("uni-graph", "figure"),
        Output("uni-graph-title", "children"),
        Output("uni-kpi-skew", "children"),
        Output("uni-kpi-skew-label", "children"),
        Output("uni-kpi-kurt", "children"),
        Output("uni-kpi-kurt-label", "children"),
        Output("uni-interpretacion", "children"),
        Input("uni-var-selector", "value"),
    )
    def update_uni(var):
        if var == "Class":
            return (
                _fig_bar_class(),
                "Class",
                "99.83%", "No fraude",
                "0.17%", "Fraude",
                _interpretacion_uni(var),
            )
        if var == "log_Amount":
            return (
                _fig_boxplot_uni(var),
                "log_Amount",
                "", "Asimetría",
                "", "Curtosis",
                _interpretacion_uni("log_Amount"),
            )
        fig = _fig_kde_uni(var) if var == "Time" else _fig_boxplot_uni(var)
        sk = SKEW_KURT.get(var, {})
        return (
            fig,
            var,
            f"{sk.get('skew', '')}", "Asimetría",
            f"{sk.get('kurt', '')}", "Curtosis",
            _interpretacion_uni(var),
        )

    # Bivariado
    @app.callback(
        Output("biv-kpis", "children"),
        Output("biv-graph", "figure"),
        Output("biv-graph-title", "children"),
        Output("biv-interpretacion", "children"),
        Input("biv-var-selector", "value"),
    )
    def update_biv(var):
        fig = _fig_kde_biv_time() if var == "Time" else _fig_violin_biv(var)
        return (
            _kpis_biv(var),
            fig,
            f"{var} vs Class",
            _interpretacion_biv(var),
        )

    # Correlación
    @app.callback(
        Output("corr-content", "children"),
        Output("corr-interpretacion", "children"),
        Input("corr-selector", "value"),
    )
    def update_correlacion(vista):
        if vista == "general":
            content = dcc.Graph(figure=_FIG_CORRELACION, config={"displayModeBar": False})
            interp = html.P(
                "En este análisis de correlación inicial usando Spearman, muchas de las variables V1-V28 "
                "muestran correlaciones bajas entre sí y con las variables Amount y Time, es decir no hay "
                "relaciones monótónicas fuertes. No obstante, hay correlaciones visibles como: V21 y V22 "
                "tienen una correlación positiva moderada (r = 0.68), mientras que V2 con Amount muestra "
                "una correlación negativa moderada (r = -0.50). De todas formas, no se observan patrones "
                "de correlación generalizados marcados. Cabe resaltar que la correlación de Pearson es "
                "aproximadamente 0 entre las variables V1 hasta V28, debido a la ortogonalidad del proceso PCA.",
                className="section-body",
            )
        else:
            content = dbc.Row([
                dbc.Col(dcc.Graph(figure=_FIG_CORRELACION_0, config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=_FIG_CORRELACION_1, config={"displayModeBar": False}), md=6),
            ])
            interp = html.P(
                "En la vista por clases se observa un contraste claro: en las transacciones no fraudulentas las "
                "correlaciones son en su mayoría cercanas a 0, sin patrones consistentes entre variables, lo cual "
                "es esperable dado que dominan el dataset y determinan el comportamiento global. En cambio, en las "
                "transacciones fraudulentas emergen correlaciones moderadas a fuertes entre las primeras componentes "
                "principales (V1–V18), destacando V18–V17, V17–V16 y V18–V16 (r ≈ 0.94–0.96), pues estas capturan "
                "mayor parte de la variabilidad de los datos y permiten evidenciar patrones internos consistentes "
                "en esta clase que no son visibles en el análisis general.",
                className="section-body",
            )
        return content, interp