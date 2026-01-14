"""
Marketplace Liquidity Engine - Streamlit Dashboard
===================================================

Interactive dashboard for real estate price optimization using
causal inference and machine learning.

Features:
- Property parameter inputs (sidebar)
- Demand curve visualization with AI-recommended price
- Impact simulation with liquidity preference slider
- SHAP explainability for elasticity predictions

Author: Causal Inference Project
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from econml.dml import LinearDML
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Marketplace Liquidity Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #9CA3AF;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Fix metric cards for dark mode */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #475569;
    }
    [data-testid="stMetric"] label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #4ade80 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] svg {
        fill: #4ade80;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource(show_spinner=False)
def load_causal_model():
    """Load and train the causal model (cached for performance)."""
    df = pd.read_csv('marketplace_data.csv')
    
    le_room = LabelEncoder()
    le_tier = LabelEncoder()
    df['room_type_encoded'] = le_room.fit_transform(df['room_type'])
    df['property_tier_encoded'] = le_tier.fit_transform(df['property_tier'])
    
    feature_cols = [
        'location_score', 'capacity', 'host_rating', 'seasonality_index',
        'room_type_encoded', 'property_tier_encoded',
        'competitor_density', 'host_experience_days', 'platform_fee_rate', 'cleaning_cost'
    ]
    
    X = df[feature_cols].values
    T = df['historical_price'].values
    Y = df['is_booked'].values
    
    model_t = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_y = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    
    est = LinearDML(model_y=model_y, model_t=model_t, random_state=42, cv=3)
    est.fit(Y, T, X=X)
    
    return est, feature_cols, df, le_room, le_tier


def calculate_demand(price, base_demand, current_price, elasticity):
    """Calculate demand at a given price using linear approximation."""
    demand = base_demand + elasticity * (price - current_price)
    return np.clip(demand, 0.0, 1.0)


def calculate_revenue(price, base_demand, current_price, elasticity):
    """Calculate expected revenue at a given price."""
    demand = calculate_demand(price, base_demand, current_price, elasticity)
    return price * demand


def find_optimal_price(base_demand, current_price, elasticity, liquidity_pref, price_bounds=(20, 500)):
    """
    Find optimal price with liquidity preference adjustment.
    
    liquidity_pref: 0 = pure margin focus, 1 = pure occupancy focus
    """
    if elasticity >= 0:
        return price_bounds[1]
    
    # Pure revenue-maximizing price
    revenue_optimal = current_price / 2 - base_demand / (2 * elasticity)
    
    # Pure occupancy-maximizing price (lower is better for occupancy)
    # Set to a price that maintains ~90% of base demand
    occupancy_optimal = current_price - 0.1 * base_demand / elasticity
    
    # Blend based on liquidity preference
    optimal_price = (1 - liquidity_pref) * revenue_optimal + liquidity_pref * occupancy_optimal
    
    return np.clip(optimal_price, price_bounds[0], price_bounds[1])


def create_demand_curve(base_demand, current_price, optimal_price, elasticity, price_range):
    """Create interactive demand curve plot with Plotly."""
    
    demands = [calculate_demand(p, base_demand, current_price, elasticity) for p in price_range]
    revenues = [calculate_revenue(p, base_demand, current_price, elasticity) for p in price_range]
    
    current_demand = calculate_demand(current_price, base_demand, current_price, elasticity)
    optimal_demand = calculate_demand(optimal_price, base_demand, current_price, elasticity)
    
    current_revenue = calculate_revenue(current_price, base_demand, current_price, elasticity)
    optimal_revenue = calculate_revenue(optimal_price, base_demand, current_price, elasticity)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📈 Demand Curve", "💰 Revenue Curve"),
        horizontal_spacing=0.1
    )
    
    # Demand curve
    fig.add_trace(
        go.Scatter(
            x=price_range, y=[d * 100 for d in demands],
            mode='lines',
            name='Demand',
            line=dict(color='#3498db', width=3),
            hovertemplate='Price: $%{x:.0f}<br>Demand: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Current price marker
    fig.add_trace(
        go.Scatter(
            x=[current_price], y=[current_demand * 100],
            mode='markers+text',
            name='Current Price',
            marker=dict(color='#e74c3c', size=15, symbol='circle'),
            text=['Current'],
            textposition='top center',
            hovertemplate=f'Current: ${current_price:.0f}<br>Demand: {current_demand*100:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Optimal price marker
    fig.add_trace(
        go.Scatter(
            x=[optimal_price], y=[optimal_demand * 100],
            mode='markers+text',
            name='AI Recommended',
            marker=dict(color='#2ecc71', size=15, symbol='star'),
            text=['AI Optimal'],
            textposition='top center',
            hovertemplate=f'Optimal: ${optimal_price:.0f}<br>Demand: {optimal_demand*100:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Revenue curve
    fig.add_trace(
        go.Scatter(
            x=price_range, y=revenues,
            mode='lines',
            name='Revenue',
            line=dict(color='#9b59b6', width=3),
            showlegend=False,
            hovertemplate='Price: $%{x:.0f}<br>Revenue: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Revenue markers
    fig.add_trace(
        go.Scatter(
            x=[current_price], y=[current_revenue],
            mode='markers',
            marker=dict(color='#e74c3c', size=15, symbol='circle'),
            showlegend=False,
            hovertemplate=f'Current Revenue: ${current_revenue:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=[optimal_price], y=[optimal_revenue],
            mode='markers',
            marker=dict(color='#2ecc71', size=15, symbol='star'),
            showlegend=False,
            hovertemplate=f'Optimal Revenue: ${optimal_revenue:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Booking Probability (%)", row=1, col=1)
    fig.update_yaxes(title_text="Expected Revenue ($)", row=1, col=2)
    
    return fig, current_demand, optimal_demand, current_revenue, optimal_revenue


def create_shap_explanation(model, features, feature_names):
    """Generate SHAP explanation for the property's elasticity."""
    try:
        shap_values = model.shap_values(features.reshape(1, -1))
        
        # Handle EconML dictionary return format
        if isinstance(shap_values, dict):
            while isinstance(shap_values, dict):
                shap_values = next(iter(shap_values.values()))
        
        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values.flatten()
        else:
            shap_array = np.array(shap_values).flatten()
        
        return shap_array
    except Exception as e:
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Marketplace Liquidity Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Price Optimization using Causal Inference</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading Causal Model..."):
        model, feature_cols, df, le_room, le_tier = load_causal_model()
    
    # Sidebar - Property Inputs
    st.sidebar.header("🏡 Property Configuration")
    
    st.sidebar.subheader("Location & Quality")
    location_score = st.sidebar.slider("Location Score", 0.0, 10.0, 7.0, 0.1,
                                       help="Quality of location (0=Poor, 10=Excellent)")
    host_rating = st.sidebar.slider("Host Rating", 3.0, 5.0, 4.5, 0.1,
                                    help="Average host rating from guests")
    
    st.sidebar.subheader("Property Details")
    room_type = st.sidebar.selectbox("Room Type", ["Entire", "Private", "Shared"])
    capacity = st.sidebar.slider("Guest Capacity", 1, 12, 4)
    
    st.sidebar.subheader("Market Conditions")
    seasonality = st.sidebar.slider("Seasonality Index", 0.5, 2.0, 1.2, 0.1,
                                    help="Demand multiplier (0.5=Low, 2.0=Peak season)")
    competitor_density = st.sidebar.slider("Nearby Competitors", 0, 50, 15)
    
    st.sidebar.subheader("Current Pricing")
    current_price = st.sidebar.number_input("Current Price ($)", 20, 500, 180)
    base_demand = st.sidebar.slider("Current Occupancy Rate", 0.1, 1.0, 0.75, 0.05,
                                    help="Current booking probability")
    
    st.sidebar.divider()
    
    # Liquidity Preference Slider
    st.sidebar.subheader("⚖️ Optimization Preference")
    liquidity_pref = st.sidebar.slider(
        "Liquidity Priority",
        0.0, 1.0, 0.3, 0.1,
        help="0 = Maximize Revenue | 1 = Maximize Occupancy"
    )
    
    pref_label = "Revenue Focus" if liquidity_pref < 0.4 else "Balanced" if liquidity_pref < 0.7 else "Occupancy Focus"
    st.sidebar.caption(f"Strategy: **{pref_label}**")
    
    # Encode inputs
    room_type_encoded = le_room.transform([room_type])[0]
    
    # Determine property tier
    luxury_score = (location_score / 10 * 0.4 + (host_rating - 3) / 2 * 0.3 + 
                   (0.3 if room_type == 'Entire' else 0.15 if room_type == 'Private' else 0))
    property_tier = 'Luxury' if luxury_score > 0.6 else 'Economy'
    property_tier_encoded = le_tier.transform([property_tier])[0]
    
    # Create feature vector
    features = np.array([
        location_score,
        capacity,
        host_rating,
        seasonality,
        room_type_encoded,
        property_tier_encoded,
        competitor_density,
        365,  # host_experience_days (default)
        0.14,  # platform_fee_rate (default)
        15 + capacity * 5  # cleaning_cost (derived)
    ])
    
    # Predict elasticity
    elasticity = model.effect(features.reshape(1, -1))[0]
    
    # Find optimal price
    optimal_price = find_optimal_price(base_demand, current_price, elasticity, liquidity_pref)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    current_demand = calculate_demand(current_price, base_demand, current_price, elasticity)
    optimal_demand = calculate_demand(optimal_price, base_demand, current_price, elasticity)
    current_revenue = calculate_revenue(current_price, base_demand, current_price, elasticity)
    optimal_revenue = calculate_revenue(optimal_price, base_demand, current_price, elasticity)
    
    price_change = optimal_price - current_price
    revenue_change = optimal_revenue - current_revenue
    occupancy_change = (optimal_demand - current_demand) * 100
    
    with col1:
        st.metric(
            "🎯 AI Recommended Price",
            f"${optimal_price:.2f}",
            f"{price_change:+.2f} ({price_change/current_price*100:+.1f}%)"
        )
    
    with col2:
        st.metric(
            "💰 Projected Revenue",
            f"${optimal_revenue:.2f}",
            f"{revenue_change:+.2f} ({revenue_change/current_revenue*100 if current_revenue > 0 else 0:+.1f}%)"
        )
    
    with col3:
        st.metric(
            "📊 Projected Occupancy",
            f"{optimal_demand*100:.1f}%",
            f"{occupancy_change:+.1f}%"
        )
    
    with col4:
        st.metric(
            "📉 Price Elasticity (β)",
            f"{elasticity:.5f}",
            f"{'Less' if abs(elasticity) < 0.001 else 'More'} sensitive"
        )
    
    st.divider()
    
    # Property classification
    tier_color = "🟢" if property_tier == "Luxury" else "🟡"
    st.info(f"{tier_color} **Property Classification:** {property_tier} | "
            f"A $10 price increase is expected to change booking probability by **{elasticity*10*100:.2f}%**")
    
    # Demand Curve Visualization
    st.subheader("📈 Price-Demand Relationship")
    
    price_range = np.linspace(20, 500, 100)
    fig, _, _, _, _ = create_demand_curve(
        base_demand, current_price, optimal_price, elasticity, price_range
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # SHAP Explainability
    st.subheader("🔍 Why This Elasticity?")
    
    shap_values = create_shap_explanation(model, features, feature_cols)
    
    if shap_values is not None:
        # Create SHAP bar chart
        shap_df = pd.DataFrame({
            'Feature': feature_cols,
            'SHAP Value': shap_values,
            'Feature Value': features
        }).sort_values('SHAP Value', key=abs, ascending=True)
        
        # Color based on direction
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in shap_df['SHAP Value']]
        
        # Scale SHAP values for visibility (multiply by 10000 for display)
        shap_scaled = shap_df['SHAP Value'] * 10000
        
        fig_shap = go.Figure(go.Bar(
            x=shap_scaled,
            y=shap_df['Feature'],
            orientation='h',
            marker_color=colors,
            text=[f'{v:.4f}' for v in shap_df['SHAP Value']],
            textposition='outside',
            hovertemplate='%{y}: %{text}<extra></extra>'
        ))
        
        # Calculate range for visibility
        max_abs = max(abs(shap_scaled.min()), abs(shap_scaled.max()), 1)
        
        fig_shap.update_layout(
            title="Feature Contributions to Elasticity (scaled ×10,000)",
            xaxis_title="Impact on Elasticity (×10,000)",
            xaxis=dict(range=[-max_abs * 1.3, max_abs * 1.3]),
            yaxis_title="",
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Top explanations
        top_positive = shap_df[shap_df['SHAP Value'] > 0].tail(2)
        top_negative = shap_df[shap_df['SHAP Value'] < 0].head(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Factors Making Demand MORE Elastic:**")
            for _, row in top_negative.iterrows():
                st.markdown(f"- **{row['Feature']}** = {row['Feature Value']:.2f}")
        
        with col2:
            st.markdown("**🔴 Factors Making Demand LESS Elastic:**")
            for _, row in top_positive.iterrows():
                st.markdown(f"- **{row['Feature']}** = {row['Feature Value']:.2f}")
    else:
        st.warning("SHAP explanation not available for this configuration.")
    
    # Footer
    st.divider()
    st.caption("Built with Causal Inference (EconML) • Double Machine Learning • GradientBoosting")


if __name__ == "__main__":
    main()
