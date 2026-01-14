"""
Configuration Settings
======================
Central configuration for the liquidity engine.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """Application settings."""
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    DATA_PATH: Path = ARTIFACTS_DIR / "marketplace_data.csv"
    
    # Data generation
    N_SAMPLES: int = 10000
    RANDOM_SEED: int = 42
    
    # Model parameters
    N_ESTIMATORS: int = 200
    MAX_DEPTH: int = 5
    CV_FOLDS: int = 5
    
    # Optimization
    PRICE_MIN: float = 20.0
    PRICE_MAX: float = 500.0
    DEFAULT_MIN_OCCUPANCY: float = 0.2
    
    def __post_init__(self):
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
