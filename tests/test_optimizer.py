"""Tests for the optimization module."""

import numpy as np
import pytest
from unittest.mock import MagicMock


class TestPriceOptimizer:
    """Tests for PriceOptimizer class."""
    
    def test_demand_function_decreases_with_price(self):
        """Verify demand decreases as price increases (negative elasticity)."""
        from src.liquidity_engine.optimization.optimizer import PriceOptimizer
        
        mock_model = MagicMock()
        mock_model.effect.return_value = np.array([-0.001])
        
        optimizer = PriceOptimizer(mock_model)
        
        base_demand = 0.8
        current_price = 100
        elasticity = -0.001
        
        demand_at_100 = optimizer.demand_function(100, base_demand, current_price, elasticity)
        demand_at_150 = optimizer.demand_function(150, base_demand, current_price, elasticity)
        
        assert demand_at_150 < demand_at_100
    
    def test_optimal_price_bounded(self):
        """Verify optimal price stays within bounds."""
        from src.liquidity_engine.optimization.optimizer import PriceOptimizer
        
        mock_model = MagicMock()
        mock_model.effect.return_value = np.array([-0.0001])  # Very inelastic
        
        optimizer = PriceOptimizer(mock_model, price_bounds=(50, 300))
        
        optimal = optimizer.find_optimal_price(
            base_demand=0.9,
            current_price=100,
            elasticity=-0.0001
        )
        
        assert 50 <= optimal <= 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
