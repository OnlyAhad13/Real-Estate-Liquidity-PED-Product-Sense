"""
Causal Estimator using Double Machine Learning
===============================================
Estimates Price Elasticity of Demand using EconML's LinearDML.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from econml.dml import LinearDML
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CausalEstimator:
    """Double Machine Learning estimator for price elasticity."""
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        cv: int = 5,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv = cv
        self.random_state = random_state
        self.model = None
        self.feature_cols = None
        self._encoders = {}
    
    def prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for DML estimation."""
        
        # Encode categoricals
        self._encoders['room'] = LabelEncoder()
        self._encoders['tier'] = LabelEncoder()
        
        df = df.copy()
        df['room_type_encoded'] = self._encoders['room'].fit_transform(df['room_type'])
        df['property_tier_encoded'] = self._encoders['tier'].fit_transform(df['property_tier'])
        
        self.feature_cols = [
            'location_score', 'capacity', 'host_rating', 'seasonality_index',
            'room_type_encoded', 'property_tier_encoded',
            'competitor_density', 'host_experience_days', 'platform_fee_rate', 'cleaning_cost'
        ]
        
        X = df[self.feature_cols].values
        T = df['historical_price'].values
        Y = df['is_booked'].values
        
        return X, T, Y
    
    def fit(self, df: pd.DataFrame) -> 'CausalEstimator':
        """Fit the DML model."""
        X, T, Y = self.prepare_data(df)
        
        model_t = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        model_y = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        self.model = LinearDML(
            model_y=model_y,
            model_t=model_t,
            cv=self.cv,
            random_state=self.random_state
        )
        
        self.model.fit(Y, T, X=X)
        return self
    
    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.effect(X)
    
    def get_ate(self, X: np.ndarray) -> float:
        """Get Average Treatment Effect."""
        return float(np.mean(self.estimate_cate(X)))


def main():
    """CLI entry point."""
    # Load data
    df = pd.read_csv("artifacts/marketplace_data.csv")
    
    # Fit estimator
    estimator = CausalEstimator()
    estimator.fit(df)
    
    # Estimate effects
    X, _, _ = estimator.prepare_data(df)
    cate = estimator.estimate_cate(X)
    ate = estimator.get_ate(X)
    
    print(f"Average Treatment Effect: {ate:.6f}")
    print(f"CATE range: [{cate.min():.6f}, {cate.max():.6f}]")


if __name__ == "__main__":
    main()
