"""
tabs/problema.py
----------------
Pestaña 3 – Planteamiento del problema con visualizaciones descriptivas.
Incluye: distribución fraude vs no fraude, Amount, Time y análisis por hora.
"""

import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from data.data_loader import get_df


#Gráficos 
def _fig_barras_clase():
    df = get_df()
    frecuencia = df['Class'].value_counts().sort_index()
    proporciones = df['Class'].value_counts(normalize=True).sort_index() * 100
    #combinar en dataframe
    df_plot = pd.DataFrame({
        "Tipo": ["No fraude", "Fraude"],
        "Frecuencia": frecuencia.values,
        "Proporcion": proporciones.values
    })

    fig = px.bar(
        df_plot,
        x="Tipo",
        y="Frecuencia",
        color="Tipo",
        color_discrete_sequence=["#2e86c1", "#c0392b"],
        text=df_plot.apply(lambda x: f"{x['Frecuencia']:,}<br>{x['Proporcion']:.2f}%", axis=1),
    )


    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Cantidad: %{y:,}<br>%{text}<extra></extra>"
    )

    fig.update_layout(
        title="Distribución de transacciones: fraude vs no fraude",
        xaxis_title="Tipo de transacción",
        yaxis_title="Número de transacciones",
        plot_bgcolor="white",
        yaxis=dict(gridcolor="lightgrey"),
        showlegend=False
    )
    return fig


def _fig_amount_box():
    df = get_df()
    fig = px.box(
        df, x="Class", y=np.log1p(df["Amount"].values),
        color="Class",
        color_discrete_map={"0": "#2e86c1", "1": "#c0392b"},
        labels={"Class": "Tipo", "y": "log(Amount + 1)"},
        title="Distribución del monto (escala logarítmica) por clase",
    )
    fig.update_xaxes(tickvals=["0", "1"], ticktext=["No fraude", "Fraude"])
    fig.update_layout(showlegend=False, plot_bgcolor="white",
                      yaxis=dict(gridcolor="#eeeeee"), paper_bgcolor="white",
                      margin=dict(t=50, b=40))
    return fig


def _fig_time_distplot():
    df = get_df()
    df2 = df.copy()
    df2["Tipo"]= df["Class"].map({"0": "No fraude", "1": "Fraude"})

    fig = px.histogram(
        df2, x="Time", color="Tipo",
        barmode="overlay",
        nbins=80,
        histnorm="probability density",
        color_discrete_map={"No fraude": "#2e86c1", "Fraude": "#c0392b"},
        opacity=0.6,
        title="Distribución de Time por tipo de transacción",
        labels={"Time": "Tiempo (segundos desde la primera transacción)",
                "probability density": "Densidad"},
    )
    fig.update_layout(
        yaxis_title="Densidad",
        yaxis=dict(gridcolor="#eee", tickformat=".2e"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=50, b=40, l=40, r=20),
        font=dict(family="Segoe UI, Arial", size=11),
    )
    return fig


def _fig_hora():
    df   = get_df().copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    tiempo = (
        df.groupby(["Hour", "Class"])["Amount"]
        .median()
        .reset_index()
    )
    tiempo["Tipo"] = tiempo["Class"].map({"0": "No fraude", "1": "Fraude"})

    fig = px.line(
        tiempo, x="Hour", y="Amount", color="Tipo",
        markers=True,
        color_discrete_map={"No fraude": "#2e86c1", "Fraude": "#c0392b"},
        labels={
            "Hour": "Hora relativa al inicio del registro",
            "Amount": "Mediana del monto",
            "Tipo": "Tipo de transacción",
        },
        title="Mediana del monto de transacciones por hora del día",
    )
    fig.update_layout(
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#eeeeee"),
        paper_bgcolor="white",
        margin=dict(t=50, b=40),
    )
    return fig

_figbarrasclase = _fig_barras_clase()
_figamountbox = _fig_amount_box()
_figtimedistplot = _fig_time_distplot()
_fighora = _fig_hora()
#Layout

def layout():
    return html.Div([

        dbc.Row(dbc.Col(html.Div([
            html.H5("Planteamiento del problema", className="section-title",
                    style={"marginTop": 0}),
            html.P(
                "El principal desafío es poder extraer patrones para poder identificar transacciones fraudulentas, a pesar del desbalance de clases. A continuación se presentan distribuciones claves del dataset:",
                className="section-body",
            ),
        ]), width=12), className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div("Frecuencia por clase", className="card-header-custom"),
                dbc.CardBody(dcc.Graph(figure=_figbarrasclase,
                                       config={"displayModeBar": False})),
            ], className="hallazgo-card"), md=6),
            dbc.Col(dbc.Card([
                html.Div("Monto de la transacción por clase", className="card-header-custom"),
                dbc.CardBody(dcc.Graph(figure=_figamountbox,
                                       config={"displayModeBar": False})),
            ], className="hallazgo-card"), md=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div("Distribución temporal (Time)", className="card-header-custom"),
                dbc.CardBody(dcc.Graph(figure=_figtimedistplot,
                                       config={"displayModeBar": False})),
            ], className="hallazgo-card"), md=6),
            dbc.Col(dbc.Card([
                html.Div("Análisis por hora del día", className="card-header-custom"),
                dbc.CardBody(dcc.Graph(figure=_fighora,
                                       config={"displayModeBar": False})),
            ], className="hallazgo-card"), md=6),
        ]),

    ], className="tab-content-wrapper tab-fade-in")