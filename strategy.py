"""
DEBUG Broom Breakout Strategy
Simplified with extensive logging to find the issue
"""

import pandas as pd
import numpy as np
import warnings
import gc
import time
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import requests
from pathlib import Path
import traceback

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    """Minimal configuration"""
    
    def __init__(self):
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # Backtest Period
        self.backtest_start = '2018-01-01'
        self.data_start = '2013-01-01'  # More data for 200 EMA
        
        # Universe
        self.universe_size = 10
        self.batch_size = 5
        
        # VERY RELAXED filters to find ANY trades
        self.compression_threshold = 0.15  # 15% - very relaxed
        self.base_lookback = 100  # 5 months
        self.base_duration_min = 15  # 3 weeks minimum
        self.base_duration_max = 180  # 9 months maximum
        self.consolidation_height = 0.20  # 20% box height
        
        # Entry - VERY RELAXED
        self.volume_surge = 1.1  # Only 10% above average
        self.rsi_max = 75  # Not overbought
        
        # Risk
        self.stop_loss_atr = 2.0
        self.trailing_stop_atr = 3.0
        self.max_holding_days = 120
        
        # Minimal filters
        self.min_price = 10  # Very low minimum
        self.min_avg_volume = 50000  # 50k minimum
        
        # Rate Limiting
        self.request_delay = 1
        self.batch_delay = 5
        
        # Cache
        self.cache_enabled = True
        self.cache_expiry_days = 7
        
        # Directories
        self.data_dir = Path('data_cache')
        self.results_dir = Path('results')
        self.logs_dir = Path('logs')
        
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

# ==================== LOGGING ====================

def setup_logging(config: Config):
    """Setup logging with DEBUG level"""
    logging.basicConfig(
        level=logging.DEBUG,  # DEBUG level to see everything
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logs_dir / 'debug_backtest.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

config = Config()
logger = setup_logging(config)

# ==================== SIMPLE DATA FETCHER ====================

class DataFetcher:
    """Simple data fetcher"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_data(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance"""
        try:
            import yfinance as yf
            
            logger.debug(f"Fetching {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, auto_adjust=True, timeout=30)
            
            if df is not None and not df.empty:
                logger.debug(f"Got {len(df)} rows for {ticker}")
                logger.debug(f"Columns: {df.columns.tolist()}")
                logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")
                logger.debug(f"First few rows:\n{df.head(3)}")
                logger.debug(f"Last few rows:\n{df.tail(3)}")
                
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    df = df.dropna()
                    
                    if len(df) > 250:
                        return df
                    else:
                        logger.warning(f"Only {len(df)} rows after cleaning")
                else:
                    logger.error(f"Missing columns. Have: {df.columns.tolist()}")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            logger.error(traceback.format_exc())
            return None

# ==================== DEBUG BACKTEST ====================

class DebugBacktest:
    """Debug backtest with extensive logging"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.results = []
        
        logger.info("=" * 80)
        logger.info("DEBUG BROOM BREAKOUT")
        logger.info("=" * 80)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators with logging"""
        logger.debug("Calculating indicators...")
        
        # EMAs
        for period in [20, 50, 100, 200]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        logger.debug(f"EMA_20 first valid: {df['EMA_20'].first_valid_index()}")
        logger.debug(f"EMA_200 first valid: {df['EMA_200'].first_valid_index()}")
        
        # Volume MA
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Rolling high
        df['High_20'] = df['High'].rolling(window=20).max()
        
        # Check for NaN
        logger.debug(f"NaN counts:\n{df.isnull().sum()}")
        
        return df
    
    def check_setup_conditions(self, df: pd.DataFrame, idx: int, ticker: str):
        """Check each condition with logging"""
        if idx < 200:
            logger.debug(f"  idx {idx} < 200 - skip")
            return False
        
        current_price = df.iloc[idx]['Close']
        date = df.index[idx]
        
        logger.debug(f"\n  Checking {ticker} at {date} - Price: ₹{current_price:.2f}")
        
        # 1. Price above 200 EMA
        ema_200 = df.iloc[idx]['EMA_200']
        if pd.isna(ema_200):
            logger.debug(f"  ❌ EMA_200 is NaN")
            return False
        
        if current_price <= ema_200:
            logger.debug(f"  ❌ Price {current_price:.2f} <= EMA_200 {ema_200:.2f}")
            return False
        else:
            pct_above = (current_price - ema_200) / ema_200 * 100
            logger.debug(f"  ✅ Price above EMA_200 by {pct_above:.2f}%")
        
        # 2. EMA Compression
        emas = [
            df.iloc[idx]['EMA_20'],
            df.iloc[idx]['EMA_50'],
            df.iloc[idx]['EMA_100'],
            df.iloc[idx]['EMA_200']
        ]
        
        if any(pd.isna(e) for e in emas):
            logger.debug(f"  ❌ Some EMA is NaN: {emas}")
            return False
        
        ema_max = max(emas)
        ema_min = min(emas)
        spread = (ema_max - ema_min) / current_price
        
        logger.debug(f"  EMA Spread: {spread:.4f} ({spread*100:.2f}%)")
        logger.debug(f"  EMAs: 20={emas[0]:.2f}, 50={emas[1]:.2f}, 100={emas[2]:.2f}, 200={emas[3]:.2f}")
        
        if spread > self.config.compression_threshold:
            logger.debug(f"  ❌ Spread {spread:.2%} > threshold {self.config.compression_threshold:.2%}")
            return False
        else:
            logger.debug(f"  ✅ Spread {spread:.2%} <= threshold {self.config.compression_threshold:.2%}")
        
        # 3. Consolidation
        recent = df.iloc[idx-19:idx+1]
        box_height = (recent['High'].max() - recent['Low'].min()) / current_price
        
        logger.debug(f"  Box height: {box_height:.4f} ({box_height*100:.2f}%)")
        
        if box_height > self.config.consolidation_height:
            logger.debug(f"  ❌ Box {box_height:.2%} > threshold {self.config.consolidation_height:.2%}")
            return False
        else:
            logger.debug(f"  ✅ Box {box_height:.2%} <= threshold {self.config.consolidation_height:.2%}")
        
        # 4. Base duration
        lookback = df.iloc[max(0, idx-self.config.base_lookback):idx]
        if len(lookback) < 50:
            logger.debug(f"  ❌ Lookback too short: {len(lookback)}")
            return False
        
        peak_pos = lookback['High'].idxmax()
        peak_idx = df.index.get_loc(peak_pos)
        days_since_peak = idx - peak_idx
        
        logger.debug(f"  Days since peak: {days_since_peak}")
        
        if days_since_peak < self.config.base_duration_min or \
           days_since_peak > self.config.base_duration_max:
            logger.debug(f"  ❌ Base duration {days_since_peak} outside {self.config.base_duration_min}-{self.config.base_duration_max}")
            return False
        else:
            logger.debug(f"  ✅ Base duration {days_since_peak} within range")
        
        # 5. Volume
        current_volume = df.iloc[idx]['Volume']
        avg_volume = df.iloc[idx]['Volume_MA']
        
        if pd.isna(avg_volume) or avg_volume <= 0:
            logger.debug(f"  ❌ Volume MA is NaN or zero")
            return False
        
        volume_ratio = current_volume / avg_volume
        
        logger.debug(f"  Volume ratio: {volume_ratio:.2f}x")
        
        if volume_ratio < self.config.volume_surge:
            logger.debug(f"  ❌ Volume {volume_ratio:.2f}x < threshold {self.config.volume_surge}x")
            return False
        else:
            logger.debug(f"  ✅ Volume {volume_ratio:.2f}x >= threshold {self.config.volume_surge}x")
        
        # 6. RSI
        rsi = df.iloc[idx]['RSI']
        
        if pd.isna(rsi):
            logger.debug(f"  ❌ RSI is NaN")
            return False
        
        logger.debug(f"  RSI: {rsi:.2f}")
        
        if rsi > self.config.rsi_max:
            logger.debug(f"  ❌ RSI {rsi:.2f} > {self.config.rsi_max}")
            return False
        else:
            logger.debug(f"  ✅ RSI {rsi:.2f} <= {self.config.rsi_max}")
        
        # 7. Breakout
        high_20 = df.iloc[idx]['High_20']
        
        if pd.isna(high_20):
            logger.debug(f"  ❌ High_20 is NaN")
            return False
        
        if current_price <= high_20:
            logger.debug(f"  ❌ Price {current_price:.2f} <= High_20 {high_20:.2f}")
            return False
        else:
            logger.debug(f"  ✅ Price {current_price:.2f} > High_20 {high_20:.2f}")
        
        logger.debug(f"  🎯 ALL CONDITIONS PASSED!")
        return True
    
    def run_backtest(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """Run debug backtest"""
        trades = []
        
        df = self.calculate_indicators(df)
        
        # Filter to backtest period
        backtest_df = df[df.index >= self.config.backtest_start].copy()
        
        logger.info(f"\n{ticker}: {len(backtest_df)} days in backtest period")
        
        if len(backtest_df) < 100:
            return trades
        
        setup_count = 0
        checked_count = 0
        
        for date in backtest_df.index:
            idx = df.index.get_loc(date)
            
            # Check every 5th day for speed
            if idx % 5 != 0:
                continue
            
            checked_count += 1
            
            current_price = df.iloc[idx]['Close']
            
            # Minimal filters
            if current_price < self.config.min_price:
                continue
            
            avg_volume = df.iloc[max(0, idx-20):idx]['Volume'].mean()
            if avg_volume < self.config.min_avg_volume:
                continue
            
            if self.check_setup_conditions(df, idx, ticker):
                setup_count += 1
                
                # Enter trade
                entry_price = current_price
                atr = df.iloc[idx]['ATR']
                
                if pd.isna(atr) or atr <= 0:
                    atr = current_price * 0.02  # Fallback 2%
                
                stop_loss = entry_price - (self.config.stop_loss_atr * atr)
                
                logger.info(f"🚀 TRADE FOUND: {ticker} @ {date.date()} - ₹{entry_price:.2f}")
                
                trades.append({
                    'ticker': ticker,
                    'entry_date': date,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss
                })
        
        logger.info(f"{ticker}: Checked {checked_count} days, found {setup_count} setups, {len(trades)} trades")
        
        return trades
    
    def run_universe(self, tickers: List[str]) -> pd.DataFrame:
        """Run debug universe"""
        all_trades = []
        
        logger.info("=" * 80)
        logger.info(f"DEBUG BACKTEST - {len(tickers)} STOCKS")
        logger.info("=" * 80)
        
        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"[{i+1}/{len(tickers)}] {ticker}")
                logger.info(f"{'='*60}")
                
                df = self.data_fetcher.fetch_data(ticker, self.config.data_start)
                
                if df is None or len(df) < 250:
                    logger.warning(f"✗ Insufficient data for {ticker}")
                    continue
                
                trades = self.run_backtest(df, ticker)
                
                if trades:
                    all_trades.extend(trades)
                    logger.info(f"✅ {ticker}: {len(trades)} trades found")
                else:
                    logger.info(f"❌ {ticker}: No trades found")
                
                del df
                gc.collect()
                
                time.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"Error with {ticker}: {e}")
                logger.error(traceback.format_exc())
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info(f"TOTAL TRADES FOUND: {len(all_trades)}")
        logger.info("=" * 80)
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            logger.info(f"\nTrades by ticker:")
            logger.info(trades_df.groupby('ticker').size())
            
            trades_df.to_csv('debug_trades.csv', index=False)
        
        return pd.DataFrame(all_trades)

# ==================== TEST UNIVERSE ====================

def get_test_universe() -> List[str]:
    """Small test universe"""
    return [
        'RELIANCE.NS',
        'TCS.NS',
        'HDFCBANK.NS',
        'INFY.NS',
        'ICICIBANK.NS',
    ]

# ==================== MAIN ====================

def main():
    """Main execution"""
    try:
        logger.info("=" * 80)
        logger.info("DEBUG BROOM BREAKOUT - FINDING THE ISSUE")
        logger.info("=" * 80)
        
        tickers = get_test_universe()
        
        engine = DebugBacktest(config)
        
        results = engine.run_universe(tickers)
        
        logger.info("\n✅ DEBUG COMPLETED")
        logger.info(f"Check debug_backtest.log for details")
        
        # Create empty results file
        pd.DataFrame(columns=['ticker', 'total_trades']).to_csv(
            'nifty500_broom_breakout_results.csv', index=False
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        pd.DataFrame(columns=['ticker']).to_csv(
            'nifty500_broom_breakout_results.csv', index=False
        )
        
        sys.exit(1)

if __name__ == "__main__":
    main()
