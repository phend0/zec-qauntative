import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, data: pd.DataFrame, predictions: np.ndarray):
        """
        Args:
            data: Original OHLCV + returns Dataframe
            predictions: Array of discrete positions (-1, 0, 1) perfectly aligned with data's index
        """
        self.data = data.copy()
        self.predictions = predictions
        self.target = config.features.target_asset
        
        # Convert bps to float multiplier
        self.trade_cost = config.fees.taker_fee + config.fees.slippage
        self.periods_per_year = 365 * (24 if config.dates.interval == '1h' else 1)

    def run(self):
        logger.info("Initializing Vectorized Trading Simulator...")
        
        # Position mapping
        self.data['position'] = self.predictions
        
        # A position taken at time `t` earns the return of period `t+1`
        market_returns = self.data[f'{self.target}_log_return']
        self.data['strategy_return_gross'] = self.data['position'].shift(1) * market_returns
        
        # Frictions: Apply to turnover explicitly
        self.data['turnover'] = self.data['position'].diff().abs()
        self.data['strategy_return_net'] = self.data['strategy_return_gross'] - (self.data['turnover'] * self.trade_cost)
        
        self.data.fillna({'strategy_return_gross': 0, 'strategy_return_net': 0, 'turnover': 0}, inplace=True)
        
        # Compounding Growth
        self.data['cum_market'] = np.exp(market_returns.cumsum())
        self.data['cum_strategy'] = np.exp(self.data['strategy_return_net'].cumsum())
        
        self._calculate_metrics()
        self._plot()

    def _calculate_metrics(self):
        strat_ret = self.data['strategy_return_net']
        
        # Returns / Vol
        ann_ret = strat_ret.mean() * self.periods_per_year
        ann_vol = strat_ret.std() * np.sqrt(self.periods_per_year)
        
        # Sharpe
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        
        # Sortino
        downside_ret = strat_ret[strat_ret < 0]
        downside_vol = downside_ret.std() * np.sqrt(self.periods_per_year)
        sortino = ann_ret / downside_vol if downside_vol != 0 else 0
        
        # Drawdowns
        cum_ret = self.data['cum_strategy']
        rolling_max = cum_ret.cummax()
        drawdowns = (cum_ret - rolling_max) / rolling_max
        mdd = drawdowns.min()
        
        # Calmar
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        
        # Win metrics
        winning_trades = (self.data['strategy_return_gross'] > 0).sum()
        losing_trades = (self.data['strategy_return_gross'] < 0).sum()
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_profit = self.data.loc[self.data['strategy_return_gross'] > 0, 'strategy_return_gross'].sum()
        gross_loss = abs(self.data.loc[self.data['strategy_return_gross'] < 0, 'strategy_return_gross'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        metrics = {
            "Annualized Return": f"{np.exp(ann_ret)-1:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Max Drawdown": f"{mdd:.2%}",
            "Calmar Ratio": f"{calmar:.2f}",
            "Win Rate": f"{win_rate:.2%}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Total Trades Executed": int(self.data['turnover'].sum())
        }
        
        logger.info("============== BACKTEST RESULTS ==============")
        for k, v in metrics.items():
            logger.info(f"{k:<25} {v}")
        logger.info("==============================================")

    def _plot(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data.index, self.data['cum_market'], label='Benchmark (Buy & Hold)', color='gray', alpha=0.6)
        plt.plot(self.data.index, self.data['cum_strategy'], label='ML Strategy (Net of Friction)', color='darkblue')
        plt.yscale('log')
        plt.title('Cumulative PnL: Predictive ML Strategy vs Buy & Hold Benchmark (Log Scale)')
        plt.ylabel('Cumulative Return (Multiplier)')
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        logger.info("Saved equity curve projection to 'backtest_results.png'")
