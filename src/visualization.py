"""
Plotly visualization helpers.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def line_chart(df: pd.DataFrame, title: str):
    fig = px.line(df, title=title)
    fig.update_layout(legend_title_text="", hovermode="x unified")
    return fig


def bar_chart(series: pd.Series, title: str):
    fig = px.bar(series.reset_index(), x=series.index.name or "index", y=series.name, title=title)
    fig.update_layout(showlegend=False)
    return fig


def heatmap(corr: pd.DataFrame, title: str):
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title=title, zmin=-1, zmax=1)
    return fig


def weights_pie(weights: pd.Series, title: str):
    fig = px.pie(values=weights.values, names=weights.index, title=title, hole=0.45)
    return fig


def efficient_frontier_plot(frontier: pd.DataFrame, selected_points: dict[str, tuple[float, float]] | None = None):
    fig = px.scatter(
        frontier,
        x="Volatility",
        y="Return",
        color="Sharpe",
        title="Efficient Frontier Simulation",
        hover_data={"Sharpe": ":.2f", "Volatility": ":.2%", "Return": ":.2%"},
    )
    if selected_points:
        for name, (vol, ret) in selected_points.items():
            fig.add_trace(go.Scatter(
                x=[vol], y=[ret], mode="markers+text", text=[name], textposition="top center",
                marker=dict(size=14, symbol="star"), name=name
            ))
    fig.update_layout(hovermode="closest")
    return fig
