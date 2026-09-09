"""
Broom Breakout Strategy - RESTORED ORIGINAL LOGIC
Back to the working version with minor improvements
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
    """Original working configuration"""
    
    def __init__(self):
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # Backtest Period
        self.backtest_start = '2018-01-01'
        self.data_start = '2000-01-01'  # Original had 2000
        
        # Universe
        self.universe_size = int(os.getenv('UNIVERSE_SIZE', '20'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '5'))
        
        # ORIGINAL Strategy Parameters
        self.ema_periods = [20, 50, 100, 200]
        self.weekly_ema_period = 200
        self.monthly_ema_period = 200
        
        # ORIGINAL Broom Compression - 8%
        self.broom_compression_threshold = 0.08
        
        # ORIGINAL Base Parameters
        self.base_lookback_period = 200
        self.base_duration_min = 63  # 3 months
        self.base_duration_max = 147  # 7 months
        
        # ORIGINAL Box Consolidation - 15%
        self.box_consolidation_height = 0.15
        
        # ORIGINAL Trend Exhaustion - 60%
        self.prior_trend_exhaustion_limit = 0.60
        
        # ORIGINAL Execution Parameters
        self.volume_poc_bins = 10
        self.poc_lookback = 20
        self.volume_threshold_multiplier = 1.5
        self.volume_ma_period = 50
        self.stop_loss_buffer = 0.015  # 1.5% below 200 EMA
        self.measured_move_multiplier = 2
        
        # Rate Limiting
        self.request_delay = 2
        self.batch_delay = 10
        self.max_retries = 3
        
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
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logs_dir / 'backtest.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

config = Config()
logger = setup_logging(config)

# ==================== DATA FETCHER ====================

class DataFetcher:
    """Original data fetcher"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_data(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance"""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, auto_adjust=False, timeout=30)
            
            if df is not None and not df.empty:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    df = df.dropna()
                    
                    if len(df) > 100:
                        return df
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker}: {e}")
            return None

# ==================== ORIGINAL BACKTEST ENGINE ====================

class OriginalBroomBacktest:
    """Original working Broom Breakout strategy"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.results = []
        
        logger.info("=" * 80)
        logger.info("ORIGINAL BROOM BREAKOUT STRATEGY")
        logger.info(f"Compression: {self.config.broom_compression_threshold:.0%}")
        logger.info(f"Box Height: {self.config.box_consolidation_height:.0%}")
        logger.info(f"Volume: {self.config.volume_threshold_multiplier}x")
        logger.info("=" * 80)
    
    def save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Cache data"""
        try:
            cache_file = self.config.data_dir / f"{ticker.replace('.', '_')}.csv"
            df.to_csv(cache_file)
        except:
            pass
    
    def load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load from cache"""
        try:
            cache_file = self.config.data_dir / f"{ticker.replace('.', '_')}.csv"
            if cache_file.exists():
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < self.config.cache_expiry_days * 24 * 3600:
                    return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        except:
            pass
        return None
    
    def get_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get data with caching"""
        df = self.load_from_cache(ticker)
        if df is not None:
            return df
        
        df = self.data_fetcher.fetch_data(ticker, self.config.data_start)
        if df is not None:
            self.save_to_cache(ticker, df)
        
        return df
    
    def calculate_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """ORIGINAL EMA calculation"""
        # Daily EMAs
        for period in self.config.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        return df
    
    def create_higher_timeframes(self, df: pd.DataFrame) -> pd.DataFrame:
        """ORIGINAL higher timeframe creation"""
        try:
            # Weekly
            weekly_df = df.resample('W-FRI').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            weekly_df[f'EMA_{self.config.weekly_ema_period}_Weekly'] = weekly_df['Close'].ewm(
                span=self.config.weekly_ema_period, adjust=False
            ).mean()
            
            # Monthly
            monthly_df = df.resample('ME').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            monthly_df[f'EMA_{self.config.monthly_ema_period}_Monthly'] = monthly_df['Close'].ewm(
                span=self.config.monthly_ema_period, adjust=False
            ).mean()
            
            # Map back to daily
            df['Weekly_200_EMA'] = weekly_df[f'EMA_{self.config.weekly_ema_period}_Weekly'].reindex(
                df.index, method='ffill'
            )
            df['Monthly_200_EMA'] = monthly_df[f'EMA_{self.config.monthly_ema_period}_Monthly'].reindex(
                df.index, method='ffill'
            )
            
        except Exception as e:
            logger.debug(f"Error creating higher timeframes: {e}")
            df['Weekly_200_EMA'] = np.nan
            df['Monthly_200_EMA'] = np.nan
        
        return df
    
    def check_broom_setup(self, df: pd.DataFrame, idx: int) -> bool:
        """ORIGINAL broom setup check"""
        if idx < self.config.base_lookback_period:
            return False
        
        try:
            current_price = df.iloc[idx]['Close']
            if pd.isna(current_price) or current_price <= 0:
                return False
            
            # 1. Macro Trend Filter - ORIGINAL
            weekly_ema = df.iloc[idx]['Weekly_200_EMA']
            if pd.isna(weekly_ema):
                return False
            if current_price <= weekly_ema:
                return False
            
            monthly_ema = df.iloc[idx]['Monthly_200_EMA']
            if not pd.isna(monthly_ema):
                if current_price <= monthly_ema:
                    return False
            
            # 2. EMA Broom Compression - ORIGINAL
            ema_values = []
            for period in self.config.ema_periods:
                ema_val = df.iloc[idx][f'EMA_{period}']
                if pd.isna(ema_val):
                    return False
                ema_values.append(ema_val)
            
            ema_high = max(ema_values)
            ema_low = min(ema_values)
            ema_spread = (ema_high - ema_low) / current_price
            
            if ema_spread >= self.config.broom_compression_threshold:
                return False
            
            # 3. Base Duration Check - ORIGINAL
            lookback_data = df.iloc[max(0, idx - self.config.base_lookback_period):idx]
            if len(lookback_data) < self.config.base_lookback_period:
                return False
            
            peak_idx = lookback_data['High'].idxmax()
            peak_price = lookback_data['High'].max()
            peak_position = df.index.get_loc(peak_idx)
            days_since_peak = idx - peak_position
            
            if days_since_peak < self.config.base_duration_min or days_since_peak > self.config.base_duration_max:
                return False
            
            # 4. Flat Box Consolidation - ORIGINAL
            recent_20 = df.iloc[idx-19:idx+1]
            box_height = (recent_20['High'].max() - recent_20['Low'].min()) / current_price
            
            if box_height >= self.config.box_consolidation_height:
                return False
            
            # 5. Prior Trend Exhaustion - ORIGINAL
            pre_peak_data = df.iloc[max(0, peak_position - self.config.base_lookback_period):peak_position+1]
            if len(pre_peak_data) > 0:
                lowest_low = pre_peak_data['Low'].min()
                run_up = (peak_price - lowest_low) / lowest_low
                
                if run_up > self.config.prior_trend_exhaustion_limit:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in broom setup: {e}")
            return False
    
    def calculate_volume_profile_poc(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """ORIGINAL Volume Profile POC"""
        if idx < self.config.poc_lookback:
            return None
        
        try:
            recent_data = df.iloc[idx - self.config.poc_lookback + 1:idx + 1]
            
            price_range = recent_data['High'].max() - recent_data['Low'].min()
            if price_range == 0:
                return None
            
            bins = np.linspace(recent_data['Low'].min(), recent_data['High'].max(), 
                             self.config.volume_poc_bins + 1)
            volume_by_bin = np.zeros(self.config.volume_poc_bins)
            
            for i in range(len(recent_data)):
                row = recent_data.iloc[i]
                price_low = row['Low']
                price_high = row['High']
                volume = row['Volume']
                
                if price_high > price_low:
                    for j in range(self.config.volume_poc_bins):
                        bin_low = bins[j]
                        bin_high = bins[j + 1]
                        
                        overlap_low = max(price_low, bin_low)
                        overlap_high = min(price_high, bin_high)
                        
                        if overlap_high > overlap_low:
                            overlap_percentage = (overlap_high - overlap_low) / (price_high - price_low)
                            volume_by_bin[j] += volume * overlap_percentage
            
            poc_bin_idx = np.argmax(volume_by_bin)
            poc_price = (bins[poc_bin_idx] + bins[poc_bin_idx + 1]) / 2
            
            return poc_price
            
        except Exception as e:
            return None
    
    def check_entry_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """ORIGINAL entry signal"""
        try:
            current_price = df.iloc[idx]['Close']
            
            # Get highest EMA
            ema_values = [df.iloc[idx][f'EMA_{period}'] for period in self.config.ema_periods]
            highest_ema = max(ema_values)
            
            # Calculate Volume Profile POC
            poc_price = self.calculate_volume_profile_poc(df, idx)
            if poc_price is None:
                return False
            
            # Check breakout conditions
            if current_price <= highest_ema:
                return False
            if current_price <= poc_price:
                return False
            
            # Volume confirmation
            if idx < self.config.volume_ma_period:
                return False
            
            volume_ma = df.iloc[idx - self.config.volume_ma_period:idx]['Volume'].mean()
            current_volume = df.iloc[idx]['Volume']
            
            if current_volume <= self.config.volume_threshold_multiplier * volume_ma:
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def run_backtest(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """ORIGINAL backtest"""
        trades = []
        position = None
        
        try:
            df = self.calculate_emas(df)
            df = self.create_higher_timeframes(df)
            
            backtest_df = df[df.index >= self.config.backtest_start].copy()
            
            if len(backtest_df) < 50:
                return trades
            
            for date in backtest_df.index:
                idx = df.index.get_loc(date)
                
                if position is None:
                    # Check for setup and entry
                    if self.check_broom_setup(df, idx):
                        if self.check_entry_signal(df, idx):
                            entry_price = df.iloc[idx]['Close']
                            stop_loss = df.iloc[idx]['EMA_200'] * (1 - self.config.stop_loss_buffer)
                            
                            # Calculate take profit
                            lookback_data = df.iloc[max(0, idx - self.config.base_lookback_period):idx]
                            peak_price = lookback_data['High'].max()
                            peak_pos = df.index.get_loc(lookback_data['High'].idxmax())
                            consolidation_data = df.iloc[peak_pos:idx+1]
                            lowest_price = consolidation_data['Low'].min()
                            base_depth = peak_price - lowest_price
                            take_profit = entry_price + (self.config.measured_move_multiplier * base_depth)
                            
                            position = {
                                'entry_date': date,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'base_depth': base_depth,
                                'highest_price': entry_price,
                                'atr': df.iloc[idx]['Close'] * 0.02  # Simple 2% ATR fallback
                            }
                            
                            logger.info(f"🚀 ENTRY: {ticker} @ ₹{entry_price:.2f} on {date.date()} | "
                                      f"Stop: ₹{stop_loss:.2f} | Target: ₹{take_profit:.2f}")
                
                else:
                    # Manage position
                    current_price = df.iloc[idx]['Close']
                    current_high = df.iloc[idx]['High']
                    current_low = df.iloc[idx]['Low']
                    days_held = (date - position['entry_date']).days
                    
                    # Update highest price
                    if current_high > position['highest_price']:
                        position['highest_price'] = current_high
                    
                    # Update stop with trailing
                    ema_200 = df.iloc[idx]['EMA_200']
                    new_stop = ema_200 * (1 - self.config.stop_loss_buffer)
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                    
                    # Check stop loss
                    if current_low <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'ticker': ticker,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': 'stop_loss',
                            'days_held': days_held
                        })
                        
                        logger.info(f"🛑 STOP: {ticker} | {return_pct:.2f}% | {days_held}d")
                        position = None
                        continue
                    
                    # Check take profit
                    if current_high >= position['take_profit']:
                        exit_price = position['take_profit']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'ticker': ticker,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': 'take_profit',
                            'days_held': days_held
                        })
                        
                        logger.info(f"🎯 TARGET: {ticker} | {return_pct:.2f}% | {days_held}d")
                        position = None
            
            # Close open position
            if position is not None:
                last_date = backtest_df.index[-1]
                last_idx = df.index.get_loc(last_date)
                last_price = df.iloc[last_idx]['Close']
                return_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
                days_held = (last_date - position['entry_date']).days
                
                trades.append({
                    'ticker': ticker,
                    'entry_date': position['entry_date'],
                    'exit_date': last_date,
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'return_pct': return_pct,
                    'exit_reason': 'end_of_period',
                    'days_held': days_held
                })
            
            return trades
            
        except Exception as e:
            logger.warning(f"Error in backtest for {ticker}: {e}")
            return trades
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate metrics"""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                'avg_return': 0, 'profit_factor': 0, 'max_drawdown': 0,
                'avg_days_held': 0
            }
        
        try:
            df = pd.DataFrame(trades)
            
            total_trades = len(df)
            winners = df[df['return_pct'] > 0]
            losers = df[df['return_pct'] < 0]
            
            win_rate = len(winners) / total_trades * 100
            total_return = df['return_pct'].sum()
            avg_return = df['return_pct'].mean()
            
            gross_profit = winners['return_pct'].sum() if len(winners) > 0 else 0
            gross_loss = abs(losers['return_pct'].sum()) if len(losers) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            cumulative = 100
            equity = [100]
            for ret in df['return_pct']:
                cumulative *= (1 + ret/100)
                equity.append(cumulative)
            
            equity_series = pd.Series(equity)
            drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
            
            avg_days = df['days_held'].mean()
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return': avg_return,
                'profit_factor': profit_factor,
                'max_drawdown': drawdown,
                'avg_days_held': avg_days
            }
        except Exception as e:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                'avg_return': 0, 'profit_factor': 0, 'max_drawdown': 0,
                'avg_days_held': 0
            }
    
    def run_universe(self, tickers: List[str]) -> pd.DataFrame:
        """Run backtest for universe"""
        all_trades = []
        stock_results = []
        
        logger.info("=" * 80)
        logger.info(f"ORIGINAL BACKTEST - {len(tickers)} STOCKS")
        logger.info("=" * 80)
        
        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"\n[{i+1}/{len(tickers)}] {ticker}")
                
                df = self.get_data(ticker)
                
                if df is None or len(df) < 300:
                    logger.warning(f"  ✗ Insufficient data")
                    continue
                
                trades = self.run_backtest(df, ticker)
                
                if trades:
                    all_trades.extend(trades)
                    metrics = self.calculate_metrics(trades)
                    metrics['ticker'] = ticker
                    metrics['processed'] = True
                    stock_results.append(metrics)
                    
                    logger.info(f"  ✓ {metrics['total_trades']} trades | "
                              f"Win: {metrics['win_rate']:.0f}% | "
                              f"Return: {metrics['total_return']:.1f}%")
                else:
                    logger.info(f"  - No trades")
                
                del df, trades
                gc.collect()
                
                time.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
        
        # Summary
        if all_trades:
            overall = self.calculate_metrics(all_trades)
            
            logger.info("\n" + "=" * 80)
            logger.info("OVERALL RESULTS")
            logger.info(f"Total trades: {overall['total_trades']}")
            logger.info(f"Win rate: {overall['win_rate']:.1f}%")
            logger.info(f"Total return: {overall['total_return']:.1f}%")
            logger.info(f"Profit factor: {overall['profit_factor']:.2f}")
            logger.info("=" * 80)
        
        # Save results
        results_df = pd.DataFrame(stock_results)
        if results_df.empty:
            results_df = pd.DataFrame(columns=['ticker', 'total_trades', 'win_rate', 'total_return'])
        
        results_df.to_csv('nifty500_broom_breakout_results.csv', index=False)
        logger.info(f"\nResults saved to nifty500_broom_breakout_results.csv")
        
        return results_df

# ==================== UNIVERSE ====================

def get_universe() -> List[str]:
    """Original test universe"""
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
    ]

# ==================== MAIN ====================

def main():
    """Main execution"""
    try:
        logger.info("=" * 80)
        logger.info("ORIGINAL BROOM BREAKOUT STRATEGY")
        logger.info("Restored Working Version")
        logger.info("=" * 80)
        
        tickers = get_universe()
        
        engine = OriginalBroomBacktest(config)
        
        results = engine.run_universe(tickers)
        
        logger.info("\n✅ BACKTEST COMPLETED")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        pd.DataFrame(columns=['ticker', 'total_trades', 'win_rate', 'total_return']).to_csv(
            'nifty500_broom_breakout_results.csv', index=False
        )
        
        sys.exit(1)

if __name__ == "__main__":
    main()
