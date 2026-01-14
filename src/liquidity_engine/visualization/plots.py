"""
Visualization Utilities
=======================
Plotly-based visualizations for demand curves and SHAP explanations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple


def create_demand_curve(
    base_demand: float,
    current_price: float,
    optimal_price: float,
    elasticity: float,
    price_range: np.ndarray
) -> go.Figure:
    """Create interactive demand and revenue curve visualization."""
    
    def calc_demand(p):
        return np.clip(base_demand + elasticity * (p - current_price), 0, 1)
    
    def calc_revenue(p):
        return p * calc_demand(p)
    
    demands = [calc_demand(p) * 100 for p in price_range]
    revenues = [calc_revenue(p) for p in price_range]
    
    current_demand = calc_demand(current_price)
    optimal_demand = calc_demand(optimal_price)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📈 Demand Curve", "💰 Revenue Curve"),
        horizontal_spacing=0.1
    )
    
    # Demand curve
    fig.add_trace(
        go.Scatter(x=price_range, y=demands, mode='lines', name='Demand',
                   line=dict(color='#3498db', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[current_price], y=[current_demand * 100], mode='markers',
                   name='Current', marker=dict(color='#e74c3c', size=15)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[optimal_price], y=[optimal_demand * 100], mode='markers',
                   name='Optimal', marker=dict(color='#2ecc71', size=15, symbol='star')),
        row=1, col=1
    )
    
    # Revenue curve
    fig.add_trace(
        go.Scatter(x=price_range, y=revenues, mode='lines',
                   line=dict(color='#9b59b6', width=3), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[current_price], y=[calc_revenue(current_price)], mode='markers',
                   marker=dict(color='#e74c3c', size=15), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[optimal_price], y=[calc_revenue(optimal_price)], mode='markers',
                   marker=dict(color='#2ecc71', size=15, symbol='star'), showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    fig.update_xaxes(title_text="Price ($)")
    fig.update_yaxes(title_text="Booking Probability (%)", row=1, col=1)
    fig.update_yaxes(title_text="Expected Revenue ($)", row=1, col=2)
    
    return fig


def create_shap_chart(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: np.ndarray
) -> go.Figure:
    """Create SHAP feature importance bar chart."""
    
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values,
        'Feature Value': feature_values
    }).sort_values('SHAP Value', key=abs, ascending=True)
    
    shap_scaled = shap_df['SHAP Value'] * 10000
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in shap_df['SHAP Value']]
    
    fig = go.Figure(go.Bar(
        x=shap_scaled,
        y=shap_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' for v in shap_df['SHAP Value']],
        textposition='outside'
    ))
    
    max_abs = max(abs(shap_scaled.min()), abs(shap_scaled.max()), 1)
    
    fig.update_layout(
        title="Feature Contributions to Elasticity (×10,000)",
        xaxis=dict(range=[-max_abs * 1.3, max_abs * 1.3]),
        height=400, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
