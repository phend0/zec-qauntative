import logging

import numpy as np
import pandas as pd
import ta
from statsmodels.tsa.stattools import adfuller

from config import config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target = config.features.target_asset
        self.interval = config.dates.interval
        # Determine number of periods in a day for rolling metrics
        self.periods_per_day = 24 if self.interval == "1h" else 1
        
    def add_log_returns(self):
        """Calculates log returns to ensure stationarity of price features."""
        for asset in config.features.assets:
            close_col = f'{asset}_close'
            self.df[f'{asset}_log_return'] = np.log(self.df[close_col] / self.df[close_col].shift(1))
            
    def add_technical_indicators(self):
        """Generates momentum, trend, and volatility indicators."""
        close = self.df[f'{self.target}_close']
        high = self.df[f'{self.target}_high']
        low = self.df[f'{self.target}_low']
        
        # Momentum
        self.df['RSI_14'] = ta.momentum.rsi(close, window=14)
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        self.df['Stoch_K'] = stoch.stoch()
        self.df['Stoch_D'] = stoch.stoch_signal()
        
        # Trend
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        self.df['MACD'] = macd.macd()
        self.df['MACD_diff'] = macd.macd_diff()
        
        ema_short = ta.trend.ema_indicator(close, window=12)
        ema_long = ta.trend.ema_indicator(close, window=26)
        self.df['EMA_ratio'] = ema_short / ema_long
        
        # Volatility
        self.df['ATR_14'] = ta.volatility.average_true_range(high, low, close, window=14)
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        self.df['BB_width'] = bb.bollinger_wband()
        
        # Rolling annualized volatility
        log_ret = self.df[f'{self.target}_log_return']
        self.df['Vol_7d'] = log_ret.rolling(window=7 * self.periods_per_day).std() * np.sqrt(365 * self.periods_per_day)
        self.df['Vol_30d'] = log_ret.rolling(window=30 * self.periods_per_day).std() * np.sqrt(365 * self.periods_per_day)

    def add_cross_asset_features(self):
        """Generates cross-asset dynamics (Betas, Correlations, Differentials)."""
        target_ret = self.df[f'{self.target}_log_return']
        window_30d = 30 * self.periods_per_day
        
        for asset in config.features.assets:
            if asset == self.target:
                continue
            
            asset_ret = self.df[f'{asset}_log_return']
            
            # Rolling correlation
            self.df[f'corr_{self.target}_{asset}_30d'] = target_ret.rolling(window=window_30d).corr(asset_ret)
            
            # Rolling Beta
            cov = target_ret.rolling(window=window_30d).cov(asset_ret)
            var = asset_ret.rolling(window=window_30d).var()
            self.df[f'beta_{self.target}_{asset}_30d'] = cov / var
            
            # Return differentials
            self.df[f'ret_diff_{self.target}_{asset}'] = target_ret - asset_ret

    def add_microstructure_features(self):
        """Generates volume and microstructure signals."""
        close = self.df[f'{self.target}_close']
        high = self.df[f'{self.target}_high']
        low = self.df[f'{self.target}_low']
        volume = self.df[f'{self.target}_volume']
        
        # VWAP approximation (rolling over a standard daily cycle)
        window = self.periods_per_day
        typical_price = (high + low + close) / 3
        # Ensure no division by zero
        rolling_vol = volume.rolling(window=window).sum().replace(0, np.nan)
        self.df['VWAP'] = (typical_price * volume).rolling(window=window).sum() / rolling_vol
        
        # Normalized volume ratios
        sma_vol = volume.rolling(window=20).mean().replace(0, np.nan)
        self.df['Volume_ratio'] = volume / sma_vol

    def generate_features(self) -> pd.DataFrame:
        """Executes the complete feature engineering pipeline."""
        logger.info("Generating features...")
        self.add_log_returns()
        self.add_technical_indicators()
        self.add_cross_asset_features()
        self.add_microstructure_features()
        
        # Drop rows where rolling windows introduced NaNs
        initial_len = len(self.df)
        self.df.dropna(inplace=True)
        logger.info(f"Dropped {initial_len - len(self.df)} rows due to rolling window NaNs.")
        return self.df

    @staticmethod
    def check_stationarity(df: pd.DataFrame, significance_level: float = 0.05) -> list[str]:
        """
        Runs the Augmented Dickey-Fuller (ADF) test on features.
        Returns a list of non-stationary features.
        """
        logger.info("Running ADF stationarity checks...")
        non_stationary = []
        
        for col in df.columns:
            # Skip absolute price level columns (they are inherently non-stationary)
            if any(price_col in col for price_col in ['_open', '_high', '_low', '_close']) and 'log_return' not in col:
                continue
                
            series = df[col].dropna()
            if len(series) < 30:
                continue
                
            try:
                # autolag='AIC' allows ADF to choose optimal lags
                result = adfuller(series, maxlag=1, autolag=None)
                p_value = result[1]
                
                if p_value > significance_level:
                    non_stationary.append(col)
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning(f"ADF test failed for {col}: {e}")
                
        if non_stationary:
            logger.warning(f"Detected {len(non_stationary)} non-stationary features (p > {significance_level}):")
            logger.warning(f"{non_stationary}")
        else:
            logger.info("All engineered features passed stationarity checks.")
            
        return non_stationary
