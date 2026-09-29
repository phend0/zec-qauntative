import logging
import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import requests
import yfinance as yf

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.cmc_api_key = config.api.coinmarketcap_api_key
        self.interval = config.dates.interval
        
    def _fetch_cmc_historical(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        if not self.cmc_api_key:
            logger.warning("No CMC API key provided.")
            return None
            
        symbol_map = {"ZEC": "1437", "BTC": "1", "ETH": "1027"}
        cmc_id = symbol_map.get(symbol)
        if not cmc_id:
            return None
            
        url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": self.cmc_api_key,
        }
        
        # Note: CoinMarketCap API limits and format might require adjustments in a true production environment
        params = {
            "id": cmc_id,
            "time_start": pd.to_datetime(start_date).isoformat(),
            "time_end": pd.to_datetime(end_date).isoformat(),
            "interval": self.interval,
            "count": 10000
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            quotes = data.get("data", {}).get("quotes", [])
            if not quotes:
                return None
                
            records = []
            for q in quotes:
                quote = q["quote"]["USD"]
                records.append({
                    "timestamp": pd.to_datetime(q["time_open"]),
                    "open": quote["open"],
                    "high": quote["high"],
                    "low": quote["low"],
                    "close": quote["close"],
                    "volume": quote["volume"],
                })
            
            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.index = df.index.tz_convert(None) # Make tz-naive for uniform alignment
            return df
        except (requests.RequestException, KeyError, TypeError, ValueError) as e:
            logger.error(f"CMC API error for {symbol}: {e}")
            return None

    def _fetch_ccxt_historical(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        ccxt_interval_map = {"1h": "1h", "1d": "1d"}
        ccxt_interval = ccxt_interval_map.get(self.interval, "1h")
        
        # Try Binance first, then Kraken
        exchanges = [
            (ccxt.binance(), f"{symbol}/USDT"),
            (ccxt.kraken(), f"{symbol}/USD")
        ]
        
        for exchange, pair in exchanges:
            try:
                since = exchange.parse8601(pd.to_datetime(start_date).isoformat() + "Z")
                end_timestamp = exchange.parse8601(pd.to_datetime(end_date).isoformat() + "Z")
                all_ohlcv = []
                
                while True:
                    ohlcv = exchange.fetch_ohlcv(pair, ccxt_interval, since=since, limit=1000)
                    if not ohlcv:
                        break
                        
                    all_ohlcv += ohlcv
                    since = ohlcv[-1][0] + 1
                    
                    time.sleep(exchange.rateLimit / 1000) # Respect rate limits
                    
                    if len(ohlcv) < 1000 or since > end_timestamp:
                        break
                        
                if all_ohlcv:
                    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    df = df.loc[df.index <= pd.to_datetime(end_date)]
                    if not df.empty:
                        return df
            except (ccxt.BaseError, ValueError, TypeError) as e:
                logger.warning(f"CCXT {exchange.id} error for {pair}: {e}")
                continue
                
        return None
            
    def _fetch_yfinance_historical(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        yf_symbol = f"{symbol}-USD"
        yf_interval = self.interval
        try:
            # yfinance restricts 1h intraday data to the last 720 days
            if yf_interval == "1h":
                max_start = (datetime.utcnow() - timedelta(days=720)).strftime("%Y-%m-%d")
                if pd.to_datetime(start_date) < pd.to_datetime(max_start):
                    logger.info(f"yfinance 1h requests limited to 720 days. Adjusting start_date to {max_start}")
                    start_date = max_start

            # yfinance progress=False to keep logs clean
            df = yf.download(yf_symbol, start=start_date, end=end_date, interval=yf_interval, progress=False)
            if df.empty:
                return None
                
            df.index = df.index.tz_localize(None)
            df.index.name = "timestamp"
            
            # For multi-index columns returned by yfinance if multiple symbols
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
            return df[["open", "high", "low", "close", "volume"]]
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            return None

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info(f"Fetching data for {symbol}...")
        
        # Primary: CoinMarketCap
        df = self._fetch_cmc_historical(symbol, start_date, end_date)
        if df is not None and not df.empty:
            logger.info(f"Loaded {symbol} from CMC.")
            return df
            
        # Fallback 1: CCXT (Binance / Kraken)
        df_ccxt = self._fetch_ccxt_historical(symbol, start_date, end_date)
        if df_ccxt is not None and len(df_ccxt) >= 2000:
            logger.info(f"Loaded {symbol} from CCXT ({len(df_ccxt)} rows).")
            return df_ccxt
            
        # Fallback 2: yfinance (provides up to ~17,000 hourly candles)
        df_yf = self._fetch_yfinance_historical(symbol, start_date, end_date)
        if df_yf is not None and not df_yf.empty:
            if df_ccxt is None or len(df_yf) >= len(df_ccxt):
                logger.info(f"Loaded {symbol} from yfinance ({len(df_yf)} rows).")
                return df_yf
                
        if df_ccxt is not None and not df_ccxt.empty:
            logger.info(f"Loaded {symbol} from CCXT ({len(df_ccxt)} rows).")
            return df_ccxt
            
        raise ValueError(f"Failed to fetch data for {symbol} across all providers.")

    def load_and_align(self) -> pd.DataFrame:
        start_date = config.dates.start_date
        end_date = config.dates.end_date
        
        dfs = {}
        for asset in config.features.assets:
            df = self.fetch_data(asset, start_date, end_date)
            dfs[asset] = df
            
        freq = "h" if self.interval == "1h" else "d"
        
        # Align edges (intersection of all indices)
        common_start = max(df.index.min() for df in dfs.values())
        common_end = min(df.index.max() for df in dfs.values())
        
        unified_index = pd.date_range(start=common_start, end=common_end, freq=freq)
        aligned_df = pd.DataFrame(index=unified_index)
        
        for asset, df in dfs.items():
            # Drop duplicates just in case
            df = df[~df.index.duplicated(keep='last')]
            
            # Reindex to unified UTC index
            df_reindexed = df.reindex(unified_index)
            
            # Forward fill bounded to max 2 intervals
            df_filled = df_reindexed.ffill(limit=2)
            
            df_filled.columns = [f"{asset}_{col}" for col in df_filled.columns]
            aligned_df = pd.concat([aligned_df, df_filled], axis=1)
            
        # Drop rows with NaN (unaligned edges or missing more than 2 intervals)
        aligned_df.dropna(inplace=True)
        return aligned_df
