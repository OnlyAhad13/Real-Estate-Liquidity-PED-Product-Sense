"""
Price Optimization Engine
=========================
Revenue maximization with occupancy constraints using causal elasticity estimates.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class OptimizationResult:
    """Container for price optimization results."""
    listing_id: int
    current_price: float
    optimal_price: float
    price_change_pct: float
    current_demand: float
    optimal_demand: float
    current_revenue: float
    optimal_revenue: float
    revenue_lift_pct: float
    elasticity: float
    constraint_active: bool
    recommendation: str


class PriceOptimizer:
    """Revenue-maximizing price optimizer with occupancy constraints."""
    
    def __init__(
        self,
        causal_model,
        price_bounds: Tuple[float, float] = (20.0, 500.0)
    ):
        self.causal_model = causal_model
        self.price_bounds = price_bounds
    
    def predict_elasticity(self, features: np.ndarray) -> float:
        """Predict price elasticity for given features."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return float(self.causal_model.effect(features)[0])
    
    def demand_function(self, price: float, base_demand: float, 
                        current_price: float, elasticity: float) -> float:
        """Calculate demand at a given price."""
        demand = base_demand + elasticity * (price - current_price)
        return np.clip(demand, 0.0, 1.0)
    
    def revenue_function(self, price: float, base_demand: float,
                         current_price: float, elasticity: float) -> float:
        """Calculate expected revenue at a given price."""
        demand = self.demand_function(price, base_demand, current_price, elasticity)
        return price * demand
    
    def find_optimal_price(
        self,
        base_demand: float,
        current_price: float,
        elasticity: float,
        liquidity_pref: float = 0.0
    ) -> float:
        """Find optimal price with liquidity preference adjustment."""
        if elasticity >= 0:
            return self.price_bounds[1]
        
        # Revenue-maximizing price
        revenue_optimal = current_price / 2 - base_demand / (2 * elasticity)
        
        # Occupancy-focused price
        occupancy_optimal = current_price - 0.1 * base_demand / elasticity
        
        # Blend based on preference
        optimal = (1 - liquidity_pref) * revenue_optimal + liquidity_pref * occupancy_optimal
        return np.clip(optimal, *self.price_bounds)
    
    def recommend_price(
        self,
        features: np.ndarray,
        current_price: float,
        base_demand: float,
        min_occupancy: float = 0.2,
        liquidity_pref: float = 0.0,
        listing_id: int = 0
    ) -> OptimizationResult:
        """Generate price recommendation for a listing."""
        elasticity = self.predict_elasticity(features)
        optimal_price = self.find_optimal_price(base_demand, current_price, elasticity, liquidity_pref)
        
        # Check occupancy constraint
        optimal_demand = self.demand_function(optimal_price, base_demand, current_price, elasticity)
        constraint_active = optimal_demand < min_occupancy
        
        if constraint_active and elasticity < 0:
            optimal_price = current_price + (min_occupancy - base_demand) / elasticity
            optimal_price = np.clip(optimal_price, *self.price_bounds)
            optimal_demand = self.demand_function(optimal_price, base_demand, current_price, elasticity)
        
        # Calculate metrics
        current_revenue = current_price * base_demand
        optimal_revenue = optimal_price * optimal_demand
        
        price_change_pct = (optimal_price - current_price) / current_price * 100
        revenue_lift_pct = (optimal_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0
        
        # Generate recommendation
        if abs(price_change_pct) < 2:
            recommendation = "HOLD"
        elif price_change_pct > 0:
            recommendation = f"INCREASE to ${optimal_price:.2f}"
        else:
            recommendation = f"DECREASE to ${optimal_price:.2f}"
        
        return OptimizationResult(
            listing_id=listing_id,
            current_price=current_price,
            optimal_price=optimal_price,
            price_change_pct=price_change_pct,
            current_demand=base_demand,
            optimal_demand=optimal_demand,
            current_revenue=current_revenue,
            optimal_revenue=optimal_revenue,
            revenue_lift_pct=revenue_lift_pct,
            elasticity=elasticity,
            constraint_active=constraint_active,
            recommendation=recommendation
        )
