"""
BALANCED Broom Breakout Strategy
Realistic Sniper - Gets 3-8 quality trades per stock since 2018
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
    """Balanced configuration - realistic sniper"""
    
    def __init__(self):
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # Backtest Period
        self.backtest_start = '2018-01-01'
        self.data_start = '2015-01-01'
        
        # Universe
        self.universe_size = int(os.getenv('UNIVERSE_SIZE', '50'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        
        # ===== BALANCED FILTERS =====
        
        # EMA Compression (moderate)
        self.compression_threshold = 0.08  # 8% - realistic compression
        
        # Base Pattern
        self.base_lookback = 150  # 7.5 months
        self.base_duration_min = 25  # 1.5 months minimum
        self.base_duration_max = 130  # 6.5 months maximum
        
        # Consolidation (moderate)
        self.consolidation_period = 20
        self.consolidation_height = 0.12  # 12% box height
        
        # Entry Conditions
        self.breakout_pct = 0.01  # 1% above 20-day high
        self.volume_surge = 1.3  # 1.3x average volume (realistic)
        self.rsi_min = 50  # Above 50 (bullish)
        self.rsi_max = 70  # Below 70 (not overbought)
        
        # Trend Requirements
        self.price_above_200ema = 0.02  # 2% above 200 EMA
        
        # Risk Management
        self.initial_stop_atr = 2.0
        self.trailing_stop_atr = 2.5
        self.max_holding_days = 90
        
        # Filters
        self.min_price = 50
        self.min_avg_volume = 200000  # 2 lakh minimum
        
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
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, auto_adjust=True, timeout=30)
            
            if df is not None and not df.empty:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    df = df.dropna()
                    
                    if len(df) > 200:
                        return df
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker}: {e}")
            return None

# ==================== BALANCED BACKTEST ====================

class BalancedBroomBacktest:
    """Balanced Broom Breakout - realistic quality trades"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.results = []
        
        logger.info("=" * 80)
        logger.info("BALANCED BROOM BREAKOUT STRATEGY")
        logger.info(f"Compression: {self.config.compression_threshold:.0%}")
        logger.info(f"Volume: {self.config.volume_surge}x")
        logger.info(f"RSI: {self.config.rsi_min}-{self.config.rsi_max}")
        logger.info(f"Breakout: {self.config.breakout_pct:.0%}")
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
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators"""
        # EMAs
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # Volume
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
        
        # Rolling highs
        df['High_20'] = df['High'].rolling(window=20).max()
        
        return df
    
    def is_balanced_setup(self, df: pd.DataFrame, idx: int) -> bool:
        """Check for balanced Broom setup"""
        if idx < 200:
            return False
        
        try:
            current_price = df.iloc[idx]['Close']
            
            # 1. Price above 200 EMA
            ema_200 = df.iloc[idx]['EMA_200']
            if pd.isna(ema_200) or current_price <= ema_200 * (1 + self.config.price_above_200ema):
                return False
            
            # 2. EMA Broom Compression
            emas = [
                df.iloc[idx]['EMA_20'],
                df.iloc[idx]['EMA_50'],
                df.iloc[idx]['EMA_100'],
                df.iloc[idx]['EMA_200']
            ]
            
            if any(pd.isna(e) for e in emas):
                return False
            
            ema_max = max(emas)
            ema_min = min(emas)
            spread = (ema_max - ema_min) / current_price
            
            if spread > self.config.compression_threshold:
                return False
            
            # 3. Consolidation check
            recent_20 = df.iloc[idx-19:idx+1]
            box_height = (recent_20['High'].max() - recent_20['Low'].min()) / current_price
            
            if box_height > self.config.consolidation_height:
                return False
            
            # 4. Base duration
            lookback = df.iloc[max(0, idx-self.config.base_lookback):idx]
            if len(lookback) < 80:
                return False
            
            peak_pos = lookback['High'].idxmax()
            peak_idx = df.index.get_loc(peak_pos)
            days_since_peak = idx - peak_idx
            
            if days_since_peak < self.config.base_duration_min or \
               days_since_peak > self.config.base_duration_max:
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def is_entry_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """Check entry signal"""
        try:
            current_price = df.iloc[idx]['Close']
            prev_close = df.iloc[idx-1]['Close']
            
            # 1. Breakout above 20-day high
            high_20 = df.iloc[idx]['High_20']
            if pd.isna(high_20) or current_price <= high_20 * (1 + self.config.breakout_pct):
                return False
            
            # 2. Volume confirmation
            current_volume = df.iloc[idx]['Volume']
            avg_volume = df.iloc[idx]['Volume_MA']
            
            if pd.isna(avg_volume) or avg_volume <= 0:
                return False
            
            if current_volume < self.config.volume_surge * avg_volume:
                return False
            
            # 3. RSI check
            rsi = df.iloc[idx]['RSI']
            if pd.isna(rsi) or rsi < self.config.rsi_min or rsi > self.config.rsi_max:
                return False
            
            # 4. Price increase
            if current_price <= prev_close:
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def run_backtest(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """Run balanced backtest"""
        trades = []
        position = None
        
        try:
            df = self.calculate_indicators(df)
            
            backtest_df = df[df.index >= self.config.backtest_start].copy()
            
            if len(backtest_df) < 100:
                return trades
            
            setup_count = 0
            
            for date in backtest_df.index:
                idx = df.index.get_loc(date)
                
                current_price = df.iloc[idx]['Close']
                
                # Liquidity filter
                if current_price < self.config.min_price:
                    continue
                
                avg_volume = df.iloc[max(0, idx-20):idx]['Volume'].mean()
                if avg_volume < self.config.min_avg_volume:
                    continue
                
                if position is None:
                    # Check setup
                    if self.is_balanced_setup(df, idx):
                        setup_count += 1
                        
                        if self.is_entry_signal(df, idx):
                            entry_price = current_price
                            atr = df.iloc[idx]['ATR']
                            
                            # Initial stop: 2x ATR
                            if not pd.isna(atr) and atr > 0:
                                initial_stop = entry_price - (self.config.initial_stop_atr * atr)
                            else:
                                initial_stop = entry_price * 0.97
                            
                            position = {
                                'entry_date': date,
                                'entry_price': entry_price,
                                'stop_loss': initial_stop,
                                'highest_price': entry_price,
                                'atr': atr if not pd.isna(atr) else 0
                            }
                            
                            logger.info(f"🚀 ENTRY: {ticker} @ ₹{entry_price:.2f} | "
                                      f"Stop: ₹{initial_stop:.2f}")
                
                else:
                    # Manage position
                    current_high = df.iloc[idx]['High']
                    current_low = df.iloc[idx]['Low']
                    days_held = (date - position['entry_date']).days
                    
                    # Update highest price
                    if current_high > position['highest_price']:
                        position['highest_price'] = current_high
                    
                    # Trailing stop: 2.5x ATR
                    current_atr = df.iloc[idx]['ATR']
                    if not pd.isna(current_atr) and current_atr > 0:
                        trail_stop = position['highest_price'] - (self.config.trailing_stop_atr * current_atr)
                        if trail_stop > position['stop_loss']:
                            position['stop_loss'] = trail_stop
                    
                    # Check stop
                    if current_low <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        peak_profit = (position['highest_price'] - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'ticker': ticker,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'peak_profit': peak_profit,
                            'exit_reason': 'trailing_stop',
                            'days_held': days_held
                        })
                        
                        logger.info(f"🛑 EXIT: {ticker} | "
                                  f"Return: {return_pct:.2f}% | "
                                  f"Peak: {peak_profit:.2f}% | "
                                  f"{days_held}d")
                        
                        position = None
                        continue
                    
                    # Time exit
                    if days_held >= self.config.max_holding_days:
                        exit_price = current_price
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        peak_profit = (position['highest_price'] - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'ticker': ticker,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'peak_profit': peak_profit,
                            'exit_reason': 'time_exit',
                            'days_held': days_held
                        })
                        
                        position = None
            
            # Close open position
            if position is not None:
                last_date = backtest_df.index[-1]
                last_idx = df.index.get_loc(last_date)
                last_price = df.iloc[last_idx]['Close']
                return_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
                peak_profit = (position['highest_price'] - position['entry_price']) / position['entry_price'] * 100
                days_held = (last_date - position['entry_date']).days
                
                trades.append({
                    'ticker': ticker,
                    'entry_date': position['entry_date'],
                    'exit_date': last_date,
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'return_pct': return_pct,
                    'peak_profit': peak_profit,
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
                'avg_days_held': 0, 'avg_peak_profit': 0
            }
        
        try:
            df = pd.DataFrame(trades)
            
            total_trades = len(df)
            winners = df[df['return_pct'] > 0]
            losers = df[df['return_pct'] < 0]
            
            win_rate = len(winners) / total_trades * 100
            total_return = df['return_pct'].sum()
            avg_return = df['return_pct'].mean()
            avg_peak = df['peak_profit'].mean()
            
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
                'avg_days_held': avg_days,
                'avg_peak_profit': avg_peak
            }
        except Exception as e:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                'avg_return': 0, 'profit_factor': 0, 'max_drawdown': 0,
                'avg_days_held': 0, 'avg_peak_profit': 0
            }
    
    def run_universe(self, tickers: List[str]) -> pd.DataFrame:
        """Run balanced backtest"""
        all_trades = []
        stock_results = []
        
        logger.info("=" * 80)
        logger.info(f"BALANCED BACKTEST - {len(tickers)} STOCKS")
        logger.info("=" * 80)
        
        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"\n[{i+1}/{len(tickers)}] {ticker}")
                
                df = self.get_data(ticker)
                
                if df is None or len(df) < 250:
                    logger.warning(f"  ✗ Insufficient data")
                    continue
                
                trades = self.run_backtest(df, ticker)
                
                if trades:
                    all_trades.extend(trades)
                    metrics = self.calculate_metrics(trades)
                    metrics['ticker'] = ticker
                    stock_results.append(metrics)
                    
                    logger.info(f"  ✓ {metrics['total_trades']} trades | "
                              f"Win: {metrics['win_rate']:.0f}% | "
                              f"Return: {metrics['total_return']:.1f}% | "
                              f"PF: {metrics['profit_factor']:.2f}")
                else:
                    logger.info(f"  - No trades")
                
                del df, trades
                gc.collect()
                
                time.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
        
        # Overall summary
        if all_trades:
            overall = self.calculate_metrics(all_trades)
            
            logger.info("\n" + "=" * 80)
            logger.info("OVERALL RESULTS")
            logger.info(f"Total trades: {overall['total_trades']}")
            logger.info(f"Win rate: {overall['win_rate']:.1f}%")
            logger.info(f"Total return: {overall['total_return']:.1f}%")
            logger.info(f"Profit factor: {overall['profit_factor']:.2f}")
            logger.info(f"Average peak profit: {overall['avg_peak_profit']:.1f}%")
            logger.info(f"Max drawdown: {overall['max_drawdown']:.1f}%")
            logger.info("=" * 80)
        
        # Save results
        results_df = pd.DataFrame(stock_results)
        if results_df.empty:
            results_df = pd.DataFrame(columns=['ticker', 'total_trades', 'win_rate'])
        
        results_df.to_csv('nifty500_broom_breakout_results.csv', index=False)
        logger.info(f"\nResults saved to nifty500_broom_breakout_results.csv")
        
        return results_df

# ==================== UNIVERSE ====================

def get_universe() -> List[str]:
    """Get universe"""
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS',
        'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
        'DIVISLAB.NS', 'DRREDDY.NS', 'CIPLA.NS', 'BRITANNIA.NS', 'DABUR.NS',
        'PIDILITIND.NS', 'HAVELLS.NS', 'ASTRAL.NS', 'DIXON.NS', 'TRENT.NS',
        'CUMMINSIND.NS', 'VOLTAS.NS', 'CROMPTON.NS', 'KEI.NS', 'POLYCAB.NS',
        'LUPIN.NS', 'AUROPHARMA.NS', 'BIOCON.NS', 'GLENMARK.NS', 'ALKEM.NS',
        'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS',
        'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'MPHASIS.NS', 'COFORGE.NS',
        'PERSISTENT.NS', 'TATACONSUM.NS', 'GODREJCP.NS', 'MARICO.NS', 'UBL.NS',
    ]

# ==================== MAIN ====================

def main():
    """Main execution"""
    try:
        logger.info("=" * 80)
        logger.info("BALANCED BROOM BREAKOUT STRATEGY")
        logger.info("=" * 80)
        
        tickers = get_universe()
        
        engine = BalancedBroomBacktest(config)
        
        results = engine.run_universe(tickers)
        
        logger.info("\n✅ BACKTEST COMPLETED")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        pd.DataFrame(columns=['ticker', 'total_trades', 'win_rate']).to_csv(
            'nifty500_broom_breakout_results.csv', index=False
        )
        
        sys.exit(1)

if __name__ == "__main__":
    main()
