"""
Synthetic Marketplace Data Generator
=====================================
Generates realistic Airbnb-style data with confounding for causal inference.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

np.random.seed(42)


class MarketplaceDataGenerator:
    """Generator for synthetic marketplace data with realistic confounding."""
    
    def __init__(self, n_samples: int = 10000, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
    
    def generate(self) -> pd.DataFrame:
        """Generate complete marketplace dataset."""
        n = self.n_samples
        
        # Base features
        location_score = np.round(np.random.beta(2, 2, n) * 10, 2)
        room_type = np.random.choice(['Shared', 'Private', 'Entire'], n, p=[0.10, 0.25, 0.65])
        capacity = np.clip(np.random.poisson(3, n) + 1, 1, 12)
        host_rating = np.round(3.0 + np.random.beta(5, 2, n) * 2, 2)
        seasonality_index = np.round(0.5 + np.random.beta(2, 2, n) * 1.5, 2)
        
        # Property tier
        luxury_score = (location_score / 10 * 0.4 + (host_rating - 3) / 2 * 0.3 + 
                       np.where(room_type == 'Entire', 0.3, np.where(room_type == 'Private', 0.15, 0)))
        property_tier = np.where(luxury_score > 0.6, 'Luxury', 'Economy')
        
        df = pd.DataFrame({
            'listing_id': np.arange(1, n + 1),
            'location_score': location_score,
            'room_type': room_type,
            'capacity': capacity,
            'host_rating': host_rating,
            'seasonality_index': seasonality_index,
            'property_tier': property_tier
        })
        
        # Confounded pricing
        df = self._add_confounded_pricing(df)
        
        # Demand outcome
        df = self._add_demand_outcome(df)
        
        # Instrumental variables
        df = self._add_instruments(df)
        
        return df
    
    def _add_confounded_pricing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price column with seasonality confounding."""
        room_premium = df['room_type'].map({'Shared': 0, 'Private': 25, 'Entire': 60})
        
        historical_price = (
            50 +  # base
            df['location_score'] * 4 +
            room_premium +
            (df['capacity'] - 1) * 8 +
            (df['host_rating'] - 3.0) * 15 +
            (df['seasonality_index'] - 0.5) * 40 +  # CONFOUNDING
            np.random.normal(0, 15, len(df))
        )
        
        df['historical_price'] = np.clip(historical_price, 20, 500).round(2)
        return df
    
    def _add_demand_outcome(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add booking outcome with heterogeneous price effects."""
        price_coef = np.where(df['property_tier'] == 'Luxury', -0.008, -0.015)
        
        logit = (
            1.5 +
            price_coef * df['historical_price'] +
            0.8 * df['seasonality_index'] +
            0.15 * df['location_score'] +
            df['room_type'].map({'Shared': -0.5, 'Private': 0, 'Entire': 0.3}) +
            0.05 * df['capacity'] +
            0.4 * (df['host_rating'] - 3.0) +
            np.random.normal(0, 0.3, len(df))
        )
        
        booking_probability = 1 / (1 + np.exp(-logit))
        df['booking_probability'] = np.round(booking_probability, 4)
        df['is_booked'] = np.random.binomial(1, booking_probability)
        return df
    
    def _add_instruments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add instrumental variables."""
        n = len(df)
        df['competitor_density'] = np.random.poisson(15, n)
        df['host_experience_days'] = np.random.exponential(365, n).astype(int)
        df['platform_fee_rate'] = np.round(np.random.uniform(0.10, 0.18, n), 3)
        df['cleaning_cost'] = np.clip(15 + df['capacity'] * 5 + np.random.normal(0, 5, n), 10, 100).round(2)
        return df
    
    def save(self, df: pd.DataFrame, path: str = "artifacts/marketplace_data.csv"):
        """Save dataset to CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"✓ Saved {len(df):,} records to {path}")


def generate_marketplace_data(n_samples: int = 10000, save_path: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to generate marketplace data."""
    generator = MarketplaceDataGenerator(n_samples)
    df = generator.generate()
    if save_path:
        generator.save(df, save_path)
    return df


def main():
    """CLI entry point."""
    df = generate_marketplace_data(n_samples=10000, save_path="artifacts/marketplace_data.csv")
    print(f"Generated {len(df):,} samples")
    print(df.head())


if __name__ == "__main__":
    main()
