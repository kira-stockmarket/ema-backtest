"""
Broom Breakout Strategy with Proper Trailing Stop-Loss
Let winners run with ATR-based trailing stops
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
    """Configuration with trailing stop focus"""
    
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
        self.consolidation_height = 0.15
        
        # Entry Conditions
        self.volume_surge = 1.5
        self.volume_ma_period = 20
        
        # ===== TRAILING STOP PARAMETERS =====
        self.initial_stop_atr_multiplier = 2.0  # Initial stop: 2x ATR
        self.trailing_stop_atr_multiplier = 3.0  # Trailing stop: 3x ATR
        self.atr_period = 14
        
        # Additional trailing mechanisms
        self.use_percentage_trailing = True
        self.trailing_percentage = 0.10  # 10% trailing from peak
        
        self.use_ema_trailing = True
        self.trailing_ema_period = 20  # Trail below 20 EMA
        
        # Time management
        self.max_holding_days = 120  # Extended to 4 months with trailing
        
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

# ==================== TRAILING STOP BACKTEST ====================

class TrailingStopBroomBacktest:
    """Broom Breakout with proper trailing stop management"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.results = []
        
        logger.info("=" * 80)
        logger.info("BROOM BREAKOUT WITH TRAILING STOP")
        logger.info(f"Initial Stop: {self.config.initial_stop_atr_multiplier}x ATR")
        logger.info(f"Trailing Stop: {self.config.trailing_stop_atr_multiplier}x ATR")
        logger.info(f"Percentage Trail: {self.config.trailing_percentage:.0%} from peak")
        logger.info(f"EMA Trail: {self.config.trailing_ema_period} EMA")
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
        df['ATR'] = true_range.rolling(self.config.atr_period).mean()
        
        return df
    
    def is_broom_setup(self, df: pd.DataFrame, idx: int) -> bool:
        """Check Broom setup"""
        if idx < 200:
            return False
        
        current_price = df.iloc[idx]['Close']
        
        # Price above 200 EMA
        if current_price <= df.iloc[idx]['EMA_200']:
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
    
    def is_entry_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """Check entry signal"""
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
        if pd.isna(rsi) or rsi > 75:
            return False
        
        return True
    
    def calculate_trailing_stop(self, position: Dict, df: pd.DataFrame, idx: int) -> float:
        """Calculate trailing stop using multiple methods - use the highest (tightest)"""
        stops = []
        
        current_price = df.iloc[idx]['Close']
        current_atr = df.iloc[idx]['ATR']
        
        # 1. ATR-based trailing stop
        if not pd.isna(current_atr) and current_atr > 0:
            atr_stop = position['highest_price'] - (self.config.trailing_stop_atr_multiplier * current_atr)
            stops.append(atr_stop)
        
        # 2. Percentage trailing stop from peak
        if self.config.use_percentage_trailing:
            pct_stop = position['highest_price'] * (1 - self.config.trailing_percentage)
            stops.append(pct_stop)
        
        # 3. EMA-based trailing stop
        if self.config.use_ema_trailing:
            ema_trail = df.iloc[idx][f'EMA_{self.config.trailing_ema_period}']
            if not pd.isna(ema_trail):
                stops.append(ema_trail)
        
        # 4. Previous stop (don't let stop decrease)
        stops.append(position['stop_loss'])
        
        # Return the highest stop (tightest)
        return max(stops)
    
    def run_backtest(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """Run backtest with trailing stops"""
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
                        atr = df.iloc[idx]['ATR']
                        
                        # Initial stop loss: 2x ATR below entry
                        if not pd.isna(atr) and atr > 0:
                            initial_stop = entry_price - (self.config.initial_stop_atr_multiplier * atr)
                        else:
                            # Fallback to percentage
                            initial_stop = entry_price * 0.95
                        
                        position = {
                            'entry_date': date,
                            'entry_price': entry_price,
                            'stop_loss': initial_stop,
                            'highest_price': entry_price,
                            'atr_at_entry': atr if not pd.isna(atr) else 0,
                            'trailing_activated': False
                        }
                        
                        logger.info(f"🚀 ENTRY: {ticker} @ ₹{entry_price:.2f} | "
                                  f"Initial Stop: ₹{initial_stop:.2f} | "
                                  f"ATR: ₹{atr:.2f}")
                
                else:
                    # Manage position with trailing stop
                    current_high = df.iloc[idx]['High']
                    current_low = df.iloc[idx]['Low']
                    days_held = (date - position['entry_date']).days
                    
                    # Update highest price
                    if current_high > position['highest_price']:
                        position['highest_price'] = current_high
                    
                    # Calculate new trailing stop
                    new_stop = self.calculate_trailing_stop(position, df, idx)
                    
                    # Only move stop up, never down
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                        position['trailing_activated'] = True
                    
                    # Check stop loss
                    if current_low <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        # Calculate profit from peak
                        peak_profit = (position['highest_price'] - position['entry_price']) / position['entry_price'] * 100
                        giveback = (position['highest_price'] - exit_price) / position['entry_price'] * 100
                        
                        trades.append({
                            'ticker': ticker,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'highest_price': position['highest_price'],
                            'return_pct': return_pct,
                            'peak_profit': peak_profit,
                            'giveback': giveback,
                            'exit_reason': 'trailing_stop',
                            'days_held': days_held
                        })
                        
                        logger.info(f"🛑 TRAILING STOP: {ticker} | "
                                  f"Return: {return_pct:.2f}% | "
                                  f"Peak: {peak_profit:.2f}% | "
                                  f"Giveback: {giveback:.2f}% | "
                                  f"{days_held}d")
                        
                        position = None
                        continue
                    
                    # Time exit (extended due to trailing)
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
                            'highest_price': position['highest_price'],
                            'return_pct': return_pct,
                            'peak_profit': peak_profit,
                            'giveback': peak_profit - return_pct,
                            'exit_reason': 'time_exit',
                            'days_held': days_held
                        })
                        
                        logger.info(f"⏰ TIME EXIT: {ticker} | {return_pct:.2f}% | {days_held}d")
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
                    'highest_price': position['highest_price'],
                    'return_pct': return_pct,
                    'peak_profit': peak_profit,
                    'giveback': peak_profit - return_pct,
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
                'avg_days_held': 0, 'avg_peak_profit': 0, 'avg_giveback': 0
            }
        
        try:
            df = pd.DataFrame(trades)
            
            total_trades = len(df)
            winners = df[df['return_pct'] > 0]
            losers = df[df['return_pct'] < 0]
            
            win_rate = len(winners) / total_trades * 100
            total_return = df['return_pct'].sum()
            avg_return = df['return_pct'].mean()
            avg_peak_profit = df['peak_profit'].mean()
            avg_giveback = df['giveback'].mean()
            
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
            
            # Best and worst trades
            best_trade = df['return_pct'].max()
            worst_trade = df['return_pct'].min()
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return': avg_return,
                'profit_factor': profit_factor,
                'max_drawdown': drawdown,
                'avg_days_held': avg_days,
                'avg_peak_profit': avg_peak_profit,
                'avg_giveback': avg_giveback,
                'best_trade': best_trade,
                'worst_trade': worst_trade
            }
        except Exception as e:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                'avg_return': 0, 'profit_factor': 0, 'max_drawdown': 0,
                'avg_days_held': 0, 'avg_peak_profit': 0, 'avg_giveback': 0,
                'best_trade': 0, 'worst_trade': 0
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
                    stock_results.append({
                        'ticker': ticker,
                        'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                        'avg_return': 0, 'profit_factor': 0, 'max_drawdown': 0,
                        'avg_days_held': 0, 'avg_peak_profit': 0, 'avg_giveback': 0,
                        'best_trade': 0, 'worst_trade': 0,
                        'processed': False, 'reason': 'insufficient_data'
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
                              f"Return: {metrics['total_return']:.1f}% | "
                              f"PF: {metrics['profit_factor']:.2f}")
                else:
                    logger.info(f"  - No trades")
                    stock_results.append({
                        'ticker': ticker,
                        'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                        'avg_return': 0, 'profit_factor': 0, 'max_drawdown': 0,
                        'avg_days_held': 0, 'avg_peak_profit': 0, 'avg_giveback': 0,
                        'best_trade': 0, 'worst_trade': 0,
                        'processed': True, 'reason': 'no_trades'
                    })
                
                del df, trades
                gc.collect()
                
                time.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                stock_results.append({
                    'ticker': ticker,
                    'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                    'avg_return': 0, 'profit_factor': 0, 'max_drawdown': 0,
                    'avg_days_held': 0, 'avg_peak_profit': 0, 'avg_giveback': 0,
                    'best_trade': 0, 'worst_trade': 0,
                    'processed': False, 'reason': f'error: {str(e)}'
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
            logger.info(f"Average peak profit: {overall['avg_peak_profit']:.1f}%")
            logger.info(f"Average giveback: {overall['avg_giveback']:.1f}%")
            logger.info(f"Best trade: {overall['best_trade']:.1f}%")
            logger.info(f"Worst trade: {overall['worst_trade']:.1f}%")
            logger.info(f"Max drawdown: {overall['max_drawdown']:.1f}%")
            logger.info("=" * 80)
        
        # Save results
        results_df = pd.DataFrame(stock_results)
        if results_df.empty:
            results_df = pd.DataFrame(columns=['ticker', 'processed', 'reason'])
        
        results_df.to_csv('nifty500_broom_breakout_results.csv', index=False)
        logger.info(f"\nResults saved to nifty500_broom_breakout_results.csv")
        
        return results_df

# ==================== UNIVERSE ====================

def get_universe() -> List[str]:
    """Get universe"""
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
    """Main execution"""
    try:
        logger.info("=" * 80)
        logger.info("BROOM BREAKOUT WITH TRAILING STOP-LOSS")
        logger.info("=" * 80)
        
        tickers = get_universe()
        
        engine = TrailingStopBroomBacktest(config)
        
        results = engine.run_universe(tickers)
        
        logger.info("\n✅ BACKTEST COMPLETED")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        # Create empty results file
        empty_df = pd.DataFrame(columns=['ticker', 'processed', 'reason'])
        empty_df.to_csv('nifty500_broom_breakout_results.csv', index=False)
        
        sys.exit(1)

if __name__ == "__main__":
    main()
