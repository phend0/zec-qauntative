import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__version__ = "0.0.1"

@dataclass
class APICredentials:
    coinmarketcap_api_key: str = os.getenv("CMC_API_KEY", "")
    ccxt_kraken_api_key: str = os.getenv("KRAKEN_API_KEY", "")
    ccxt_kraken_secret: str = os.getenv("KRAKEN_SECRET", "")
    ccxt_binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    ccxt_binance_secret: str = os.getenv("BINANCE_SECRET", "")

@dataclass
class DateRangeConfig:
    start_date: str = "2020-01-01"
    end_date: str = datetime.now(UTC).strftime("%Y-%m-%d")
    interval: str = "1h"

@dataclass
class FeatureConfig:
    assets: list[str] = field(default_factory=lambda: ["ZEC", "BTC", "ETH"])
    target_asset: str = "ZEC"
    lookback_windows: list[int] = field(default_factory=lambda: [7, 14, 20, 30])
    
@dataclass
class TargetConfig:
    atr_multiplier_upper: float = 1.5
    atr_multiplier_lower: float = 1.5
    vertical_barrier_periods: int = 24  # Assuming 1h interval -> 24 hours

@dataclass
class FeeConfig:
    maker_fee: float = 0.0010  # 10 bps
    taker_fee: float = 0.0010  # 10 bps
    slippage: float = 0.0005   # 5 bps
    initial_capital: float = 10000.0

@dataclass
class GridConfig:
    lightgbm_grid: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    })
    pytorch_rnn_grid: dict[str, Any] = field(default_factory=lambda: {
        "hidden_size": [32, 64, 128],
        "num_layers": [1, 2],
        "dropout": [0.2, 0.3, 0.5],
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "batch_size": [32, 64, 128]
    })
    logistic_regression_grid: dict[str, Any] = field(default_factory=lambda: {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced"]
    })

@dataclass
class PipelineConfig:
    api: APICredentials = field(default_factory=APICredentials)
    dates: DateRangeConfig = field(default_factory=DateRangeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    grids: GridConfig = field(default_factory=GridConfig)
    random_seed: int = 42

# Global configuration instance
config = PipelineConfig()
