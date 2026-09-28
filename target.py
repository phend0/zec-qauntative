import logging

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

class TargetGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target_asset = config.features.target_asset
        self.atr_col = 'ATR_14'
        self.v_barrier = config.target.vertical_barrier_periods
        self.mult_upper = config.target.atr_multiplier_upper
        self.mult_lower = config.target.atr_multiplier_lower

    def generate_targets(self) -> pd.DataFrame:
        """
        Implements the Volatility-Adjusted Triple-Barrier Method.
        Returns the dataframe with the added 'label' column.
        
        Labels:
         1: Hit Upper Barrier first
        -1: Hit Lower Barrier first
         0: Time Expired (Vertical Barrier hit) or both hit simultaneously (Neutral)
        """
        logger.info("Generating labels using Volatility-Adjusted Triple-Barrier Method...")
        
        if self.atr_col not in self.df.columns:
            raise ValueError(f"ATR column '{self.atr_col}' not found. Please run FeatureEngineer first.")
            
        close_col = f'{self.target_asset}_close'
        high_col = f'{self.target_asset}_high'
        low_col = f'{self.target_asset}_low'
        
        prices = self.df[close_col].values
        highs = self.df[high_col].values
        lows = self.df[low_col].values
        atrs = self.df[self.atr_col].values
        
        n = len(self.df)
        labels = np.zeros(n, dtype=float)
        
        for t in range(n - self.v_barrier):
            p_t = prices[t]
            atr_t = atrs[t]
            
            # If ATR is missing, we can't formulate barriers
            if np.isnan(atr_t) or atr_t == 0:
                labels[t] = np.nan
                continue
                
            upper_barrier = p_t + self.mult_upper * atr_t
            lower_barrier = p_t - self.mult_lower * atr_t
            
            hit = 0
            for i in range(1, self.v_barrier + 1):
                idx = t + i
                h_idx = highs[idx]
                l_idx = lows[idx]
                
                u_hit = h_idx >= upper_barrier
                l_hit = l_idx <= lower_barrier
                
                if u_hit and l_hit:
                    # Ambiguous volatile candle: mark as neutral/0
                    labels[t] = 0
                    hit = 1
                    break
                elif u_hit:
                    labels[t] = 1
                    hit = 1
                    break
                elif l_hit:
                    labels[t] = -1
                    hit = 1
                    break
            
            if hit == 0:
                labels[t] = 0
                
        # The last `v_barrier` periods cannot be fully evaluated without looking into the future
        labels[-self.v_barrier:] = np.nan
        
        self.df['label'] = labels
        
        # Drop rows where target couldn't be generated
        initial_len = len(self.df)
        self.df.dropna(subset=['label'], inplace=True)
        self.df['label'] = self.df['label'].astype(int)
        
        logger.info(f"Dropped {initial_len - len(self.df)} rows due to unobservable future bounds.")
        logger.info(f"Target distribution:\n{self.df['label'].value_counts(normalize=True).to_string()}")
        
        return self.df
