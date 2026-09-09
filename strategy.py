"""
Broom Breakout Strategy with Nifty Sector Index Filter
Robust Version - Always generates results file
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from pathlib import Path
from io import StringIO
import traceback

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    """Configuration"""
    
    def __init__(self):
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # Backtest Period
        self.backtest_start = '2018-01-01'
        self.data_start = '2015-01-01'
        
        # Universe
        self.universe_size = int(os.getenv('UNIVERSE_SIZE', '50'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        
        # Strategy Parameters
        self.ema_periods = [20, 50, 100, 200]
        self.compression_threshold = 0.10
        self.base_lookback = 120
        self.base_duration_min = 20
        self.base_duration_max = 120
        self.consolidation_period = 20
        self.consolidation_height = 0.15
        
        # Entry Conditions
        self.volume_surge = 1.5
        self.volume_ma_period = 20
        
        # Risk Management
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.15
        self.max_holding_days = 60
        
        # Filters
        self.min_price = 50
        self.min_avg_volume = 100000
        
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

# ==================== SIMPLE DATA FETCHER ====================

class SimpleDataFetcher:
    """Simple data fetcher with error handling"""
    
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

# ==================== SIMPLE BACKTEST ENGINE ====================

class SimpleBroomBacktest:
    """Simple Broom Breakout strategy that always generates results"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = SimpleDataFetcher()
        self.results = []
        
        logger.info("=" * 80)
        logger.info("SIMPLE BROOM BREAKOUT STRATEGY")
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
        try:
            df = self.load_from_cache(ticker)
            if df is not None:
                return df
            
            df = self.data_fetcher.fetch_data(ticker, self.config.data_start)
            if df is not None:
                self.save_to_cache(ticker, df)
            
            return df
        except Exception as e:
            logger.warning(f"Error getting data for {ticker}: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators"""
        try:
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
            
            return df
        except Exception as e:
            logger.warning(f"Error calculating indicators: {e}")
            return df
    
    def is_broom_setup(self, df: pd.DataFrame, idx: int) -> bool:
        """Check Broom setup"""
        try:
            if idx < 200:
                return False
            
            current_price = df.iloc[idx]['Close']
            
            # Price above 200 EMA
            ema_200 = df.iloc[idx]['EMA_200']
            if pd.isna(ema_200) or current_price <= ema_200:
                return False
            
            # EMA Compression
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
            
            # Consolidation
            recent = df.iloc[idx-19:idx+1]
            box_height = (recent['High'].max() - recent['Low'].min()) / current_price
            
            if box_height > self.config.consolidation_height:
                return False
            
            # Base duration
            lookback = df.iloc[max(0, idx-self.config.base_lookback):idx]
            if len(lookback) < 50:
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
            
            # Breakout
            recent_high = df.iloc[idx-20:idx]['High'].max()
            if current_price <= recent_high:
                return False
            
            # Volume
            current_volume = df.iloc[idx]['Volume']
            avg_volume = df.iloc[idx]['Volume_MA']
            
            if pd.isna(avg_volume) or avg_volume <= 0:
                return False
            
            if current_volume < self.config.volume_surge * avg_volume:
                return False
            
            # RSI not overbought
            rsi = df.iloc[idx]['RSI']
            if pd.isna(rsi) or rsi > 70:
                return False
            
            return True
        except Exception as e:
            return False
    
    def run_backtest(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """Run backtest"""
        trades = []
        position = None
        
        try:
            df = self.calculate_indicators(df)
            
            backtest_df = df[df.index >= self.config.backtest_start].copy()
            
            if len(backtest_df) < 100:
                return trades
            
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
                    # Look for entry
                    if self.is_broom_setup(df, idx) and self.is_entry_signal(df, idx):
                        entry_price = current_price
                        stop_loss = entry_price * (1 - self.config.stop_loss_pct)
                        take_profit = entry_price * (1 + self.config.take_profit_pct)
                        
                        position = {
                            'entry_date': date,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'atr': df.iloc[idx]['ATR'] if not pd.isna(df.iloc[idx]['ATR']) else 0
                        }
                        
                        logger.info(f"🚀 ENTRY: {ticker} @ ₹{entry_price:.2f}")
                
                else:
                    # Manage position
                    current_high = df.iloc[idx]['High']
                    current_low = df.iloc[idx]['Low']
                    days_held = (date - position['entry_date']).days
                    
                    # ATR trailing stop
                    if position['atr'] > 0:
                        atr_stop = current_price - (2 * position['atr'])
                        if atr_stop > position['stop_loss']:
                            position['stop_loss'] = atr_stop
                    
                    # Stop loss
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
                        
                        position = None
                        continue
                    
                    # Take profit
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
                        
                        position = None
                        continue
                    
                    # Time exit
                    if days_held >= self.config.max_holding_days:
                        exit_price = current_price
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'ticker': ticker,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
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
        logger.info(f"STARTING BACKTEST FOR {len(tickers)} STOCKS")
        logger.info("=" * 80)
        
        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"\n[{i+1}/{len(tickers)}] {ticker}")
                
                df = self.get_data(ticker)
                
                if df is None or len(df) < 250:
                    logger.warning(f"  ✗ Insufficient data")
                    # Still add to results
                    stock_results.append({
                        'ticker': ticker,
                        'total_trades': 0,
                        'win_rate': 0,
                        'total_return': 0,
                        'avg_return': 0,
                        'profit_factor': 0,
                        'max_drawdown': 0,
                        'avg_days_held': 0,
                        'processed': False,
                        'reason': 'insufficient_data'
                    })
                    continue
                
                trades = self.run_backtest(df, ticker)
                
                if trades:
                    all_trades.extend(trades)
                    metrics = self.calculate_metrics(trades)
                    metrics['ticker'] = ticker
                    metrics['processed'] = True
                    metrics['reason'] = 'success'
                    stock_results.append(metrics)
                    
                    logger.info(f"  ✓ {metrics['total_trades']} trades | "
                              f"Win: {metrics['win_rate']:.0f}% | "
                              f"Return: {metrics['total_return']:.1f}%")
                else:
                    logger.info(f"  - No trades")
                    stock_results.append({
                        'ticker': ticker,
                        'total_trades': 0,
                        'win_rate': 0,
                        'total_return': 0,
                        'avg_return': 0,
                        'profit_factor': 0,
                        'max_drawdown': 0,
                        'avg_days_held': 0,
                        'processed': True,
                        'reason': 'no_trades'
                    })
                
                del df, trades
                gc.collect()
                
                time.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                stock_results.append({
                    'ticker': ticker,
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'avg_return': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0,
                    'avg_days_held': 0,
                    'processed': False,
                    'reason': f'error: {str(e)}'
                })
        
        # Overall summary
        if all_trades:
            overall = self.calculate_metrics(all_trades)
            
            logger.info("\n" + "=" * 80)
            logger.info("OVERALL RESULTS")
            logger.info(f"Total trades: {overall['total_trades']}")
            logger.info(f"Win rate: {overall['win_rate']:.1f}%")
            logger.info(f"Total return: {overall['total_return']:.1f}%")
            logger.info(f"Profit factor: {overall['profit_factor']:.2f}")
            logger.info("=" * 80)
        
        # ALWAYS save results
        results_df = pd.DataFrame(stock_results)
        if results_df.empty:
            # Create empty DataFrame with columns
            results_df = pd.DataFrame(columns=[
                'ticker', 'total_trades', 'win_rate', 'total_return',
                'avg_return', 'profit_factor', 'max_drawdown',
                'avg_days_held', 'processed', 'reason'
            ])
        
        # Save to both locations
        results_df.to_csv('nifty500_broom_breakout_results.csv', index=False)
        results_df.to_csv(self.config.results_dir / 'results.csv', index=False)
        
        logger.info(f"\nResults saved to nifty500_broom_breakout_results.csv")
        logger.info(f"Total records: {len(results_df)}")
        
        return results_df

# ==================== UNIVERSE ====================

def get_universe() -> List[str]:
    """Get universe - Nifty 50 for testing"""
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
        'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS',
        'BAJAJFINSV.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
        'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
        'GRASIM.NS', 'HCLTECH.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
        'HINDALCO.NS', 'INDUSINDBK.NS', 'JSWSTEEL.NS', 'M&M.NS',
        'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'SBILIFE.NS',
        'SHRIRAMFIN.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
        'TECHM.NS', 'UPL.NS',
    ]

# ==================== MAIN ====================

def main():
    """Main execution - Always generates results file"""
    try:
        logger.info("=" * 80)
        logger.info("BROOM BREAKOUT STRATEGY")
        logger.info("=" * 80)
        
        tickers = get_universe()
        logger.info(f"Universe: {len(tickers)} stocks")
        
        engine = SimpleBroomBacktest(config)
        
        results = engine.run_universe(tickers)
        
        logger.info("\n✅ BACKTEST COMPLETED")
        
        # Print summary for GitHub Actions
        if config.is_github_actions:
            print("\n📊 Summary:")
            print(f"  - Stocks processed: {len(results)}")
            processed = results[results.get('processed', False) == True]
            if len(processed) > 0:
                print(f"  - Total trades: {processed['total_trades'].sum()}")
                stocks_with_trades = processed[processed['total_trades'] > 0]
                if len(stocks_with_trades) > 0:
                    print(f"  - Win rate: {stocks_with_trades['win_rate'].mean():.1f}%")
                    print(f"  - Total return: {stocks_with_trades['total_return'].sum():.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        # Create empty results file even on error
        empty_df = pd.DataFrame(columns=[
            'ticker', 'total_trades', 'win_rate', 'total_return',
            'avg_return', 'profit_factor', 'max_drawdown',
            'avg_days_held', 'processed', 'reason'
        ])
        empty_df.to_csv('nifty500_broom_breakout_results.csv', index=False)
        
        print(f"\n❌ Backtest failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
