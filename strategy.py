"""
Time-Boxed Self-Improving Broom Breakout Strategy
Runs for 1 hour, continuously learning and improving
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
import json
import traceback
import random
from itertools import product

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    """Configuration for time-boxed self-improving strategy"""
    
    def __init__(self):
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # TIME BOX
        self.max_runtime_minutes = int(os.getenv('MAX_RUNTIME_MINUTES', '60'))  # 1 hour
        self.start_time = time.time()
        
        # Backtest Period
        self.backtest_start = '2018-01-01'
        self.data_start = '2000-01-01'
        
        # Universe
        self.universe_size = int(os.getenv('UNIVERSE_SIZE', '50'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        
        # Base Strategy Parameters
        self.ema_periods = [20, 50, 100, 200]
        self.weekly_ema_period = 200
        self.monthly_ema_period = 200
        
        # Parameters that will be optimized
        self.broom_compression_threshold = 0.08
        self.base_duration_min = 63
        self.base_duration_max = 147
        self.box_consolidation_height = 0.15
        self.volume_threshold_multiplier = 1.5
        self.stop_loss_buffer = 0.015
        self.measured_move_multiplier = 2.0
        
        # Parameter ranges for optimization
        self.param_ranges = {
            'broom_compression_threshold': [0.06, 0.08, 0.10, 0.12, 0.15],
            'base_duration_min': [40, 50, 63, 80],
            'base_duration_max': [120, 147, 180, 200],
            'box_consolidation_height': [0.10, 0.15, 0.20, 0.25],
            'volume_threshold_multiplier': [1.2, 1.5, 1.8, 2.0],
            'stop_loss_buffer': [0.01, 0.015, 0.02, 0.025],
            'measured_move_multiplier': [1.5, 2.0, 2.5, 3.0],
        }
        
        # Learning settings
        self.min_trades_per_iteration = 5
        self.iterations_completed = 0
        self.best_score = -float('inf')
        self.best_params = {}
        
        # Rate Limiting
        self.request_delay = 0.5  # Faster for optimization
        self.batch_delay = 2
        
        # Cache
        self.cache_enabled = True
        self.cache_expiry_days = 7
        
        # Directories
        self.data_dir = Path('data_cache')
        self.results_dir = Path('results')
        self.logs_dir = Path('logs')
        self.optimization_dir = Path('optimization')
        
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir, self.optimization_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def time_remaining(self) -> float:
        """Check remaining time in seconds"""
        elapsed = time.time() - self.start_time
        remaining = (self.max_runtime_minutes * 60) - elapsed
        return max(0, remaining)
    
    def should_continue(self) -> bool:
        """Check if we should continue running"""
        return self.time_remaining() > 0

# ==================== LOGGING ====================

def setup_logging(config: Config):
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logs_dir / 'optimization.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

config = Config()
logger = setup_logging(config)

# ==================== DATA FETCHER ====================

class DataFetcher:
    """Data fetcher with caching for speed"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.data_pool = {}  # Pre-loaded data pool
    
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
            return None
    
    def preload_data(self, tickers: List[str], start_date: str):
        """Preload all data into memory for faster iterations"""
        logger.info("📦 Preloading data for all stocks...")
        
        for ticker in tickers:
            if ticker not in self.data_pool:
                df = self.fetch_data(ticker, start_date)
                if df is not None and len(df) > 300:
                    self.data_pool[ticker] = df
                    logger.debug(f"  ✓ {ticker}: {len(df)} rows")
                else:
                    logger.warning(f"  ✗ {ticker}: insufficient data")
                
                time.sleep(0.3)  # Small delay
        
        logger.info(f"✓ Preloaded {len(self.data_pool)} stocks")
    
    def get_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get data from pool"""
        return self.data_pool.get(ticker)

# ==================== OPTIMIZATION ENGINE ====================

class OptimizationEngine:
    """Optimization engine that tests different parameter combinations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results_history = []
        self.best_result = None
    
    def generate_parameter_combinations(self) -> List[Dict]:
        """Generate parameter combinations to test"""
        # Start with current best
        combinations = [self.config.best_params] if self.config.best_params else []
        
        # Add random mutations
        for _ in range(5):
            params = {}
            for key, values in self.config.param_ranges.items():
                if self.config.best_params and key in self.config.best_params:
                    # Mutate around best
                    current = self.config.best_params[key]
                    idx = values.index(current) if current in values else len(values)//2
                    # Random walk
                    idx = max(0, min(len(values)-1, idx + random.randint(-1, 1)))
                    params[key] = values[idx]
                else:
                    # Random selection
                    params[key] = random.choice(values)
            combinations.append(params)
        
        # Add random combinations
        for _ in range(3):
            params = {}
            for key, values in self.config.param_ranges.items():
                params[key] = random.choice(values)
            combinations.append(params)
        
        return combinations
    
    def evaluate_result(self, metrics: Dict, params: Dict) -> float:
        """Score a result - higher is better"""
        if metrics['total_trades'] < self.config.min_trades_per_iteration:
            return -float('inf')
        
        # Score = weighted combination
        score = (
            metrics['win_rate'] * 0.3 +
            min(metrics['profit_factor'], 5.0) * 10 * 0.3 +
            min(metrics['total_return'], 100) * 0.2 +
            metrics['avg_return'] * 0.2
        )
        
        # Penalize too few trades
        if metrics['total_trades'] < 10:
            score *= 0.5
        
        return score
    
    def update_best(self, params: Dict, metrics: Dict, score: float):
        """Update best parameters if score is better"""
        if score > self.config.best_score:
            self.config.best_score = score
            self.config.best_params = params.copy()
            
            logger.info(f"\n🏆 NEW BEST PARAMETERS FOUND!")
            logger.info(f"  Score: {score:.2f}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.1f}%")
            logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            logger.info(f"  Total Return: {metrics['total_return']:.1f}%")
            logger.info(f"  Parameters:")
            for key, value in params.items():
                logger.info(f"    {key}: {value}")
            
            # Save best params
            self.save_best_params()
    
    def save_best_params(self):
        """Save best parameters to file"""
        try:
            best_file = self.config.optimization_dir / 'best_params.json'
            data = {
                'params': self.config.best_params,
                'score': self.config.best_score,
                'iterations': self.config.iterations_completed,
                'timestamp': datetime.now().isoformat()
            }
            with open(best_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save best params: {e}")
    
    def save_iteration_history(self):
        """Save all iteration results"""
        try:
            history_file = self.config.optimization_dir / 'iteration_history.json'
            with open(history_file, 'w') as f:
                json.dump(self.results_history, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

# ==================== BACKTEST ENGINE ====================

class FastBacktest:
    """Fast backtest engine for optimization"""
    
    def __init__(self, config: Config, data_fetcher: DataFetcher):
        self.config = config
        self.data_fetcher = data_fetcher
    
    def calculate_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs"""
        for period in self.config.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df
    
    def create_higher_timeframes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create higher timeframes"""
        try:
            weekly_df = df.resample('W-FRI').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            weekly_df[f'EMA_{self.config.weekly_ema_period}_Weekly'] = weekly_df['Close'].ewm(
                span=self.config.weekly_ema_period, adjust=False
            ).mean()
            
            monthly_df = df.resample('ME').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            monthly_df[f'EMA_{self.config.monthly_ema_period}_Monthly'] = monthly_df['Close'].ewm(
                span=self.config.monthly_ema_period, adjust=False
            ).mean()
            
            df['Weekly_200_EMA'] = weekly_df[f'EMA_{self.config.weekly_ema_period}_Weekly'].reindex(
                df.index, method='ffill'
            )
            df['Monthly_200_EMA'] = monthly_df[f'EMA_{self.config.monthly_ema_period}_Monthly'].reindex(
                df.index, method='ffill'
            )
            
        except Exception:
            df['Weekly_200_EMA'] = np.nan
            df['Monthly_200_EMA'] = np.nan
        
        return df
    
    def run_backtest_with_params(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """Run backtest with current parameters"""
        trades = []
        position = None
        
        try:
            df = self.calculate_emas(df.copy())
            df = self.create_higher_timeframes(df)
            
            backtest_df = df[df.index >= self.config.backtest_start].copy()
            
            if len(backtest_df) < 50:
                return trades
            
            for date in backtest_df.index:
                idx = df.index.get_loc(date)
                
                if position is None:
                    # Broom setup check
                    if idx < self.config.base_duration_max + 200:
                        continue
                    
                    current_price = df.iloc[idx]['Close']
                    
                    # Trend filter
                    weekly_ema = df.iloc[idx]['Weekly_200_EMA']
                    if pd.isna(weekly_ema) or current_price <= weekly_ema:
                        continue
                    
                    # EMA compression
                    emas = [df.iloc[idx][f'EMA_{p}'] for p in self.config.ema_periods]
                    if any(pd.isna(e) for e in emas):
                        continue
                    
                    ema_spread = (max(emas) - min(emas)) / current_price
                    if ema_spread >= self.config.broom_compression_threshold:
                        continue
                    
                    # Base duration
                    lookback = df.iloc[max(0, idx-200):idx]
                    if len(lookback) < 200:
                        continue
                    
                    peak_pos = df.index.get_loc(lookback['High'].idxmax())
                    days_since_peak = idx - peak_pos
                    
                    if days_since_peak < self.config.base_duration_min or \
                       days_since_peak > self.config.base_duration_max:
                        continue
                    
                    # Box consolidation
                    recent_20 = df.iloc[idx-19:idx+1]
                    box_height = (recent_20['High'].max() - recent_20['Low'].min()) / current_price
                    
                    if box_height >= self.config.box_consolidation_height:
                        continue
                    
                    # Volume
                    volume_ma = df.iloc[max(0, idx-50):idx]['Volume'].mean()
                    current_volume = df.iloc[idx]['Volume']
                    
                    if current_volume <= self.config.volume_threshold_multiplier * volume_ma:
                        continue
                    
                    # Entry
                    entry_price = current_price
                    stop_loss = df.iloc[idx]['EMA_200'] * (1 - self.config.stop_loss_buffer)
                    
                    # Take profit
                    base_depth = lookback['High'].max() - df.iloc[peak_pos:idx+1]['Low'].min()
                    take_profit = entry_price + (self.config.measured_move_multiplier * base_depth)
                    
                    position = {
                        'entry_date': date,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'highest_price': entry_price
                    }
                    
                else:
                    current_price = df.iloc[idx]['Close']
                    current_high = df.iloc[idx]['High']
                    current_low = df.iloc[idx]['Low']
                    days_held = (date - position['entry_date']).days
                    
                    if current_high > position['highest_price']:
                        position['highest_price'] = current_high
                    
                    # Trailing stop
                    ema_200 = df.iloc[idx]['EMA_200']
                    new_stop = ema_200 * (1 - self.config.stop_loss_buffer)
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                    
                    # Exit conditions
                    if current_low <= position['stop_loss'] or \
                       current_high >= position['take_profit'] or \
                       days_held >= 90:
                        
                        if current_low <= position['stop_loss']:
                            exit_price = position['stop_loss']
                            exit_reason = 'stop_loss'
                        elif current_high >= position['take_profit']:
                            exit_price = position['take_profit']
                            exit_reason = 'take_profit'
                        else:
                            exit_price = current_price
                            exit_reason = 'time_exit'
                        
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'ticker': ticker,
                            'return_pct': return_pct,
                            'exit_reason': exit_reason,
                            'days_held': days_held
                        })
                        
                        position = None
            
            return trades
            
        except Exception as e:
            return trades
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate metrics"""
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                    'avg_return': 0, 'profit_factor': 0}
        
        df = pd.DataFrame(trades)
        total_trades = len(df)
        winners = df[df['return_pct'] > 0]
        losers = df[df['return_pct'] <= 0]
        
        return {
            'total_trades': total_trades,
            'win_rate': len(winners) / total_trades * 100,
            'total_return': df['return_pct'].sum(),
            'avg_return': df['return_pct'].mean(),
            'profit_factor': winners['return_pct'].sum() / abs(losers['return_pct'].sum()) if len(losers) > 0 else float('inf')
        }

# ==================== MAIN OPTIMIZATION LOOP ====================

def main():
    """Main optimization loop"""
    try:
        logger.info("=" * 80)
        logger.info("TIME-BOXED SELF-IMPROVING BROOM BREAKOUT")
        logger.info(f"Runtime: {config.max_runtime_minutes} minutes")
        logger.info("=" * 80)
        
        # Universe
        tickers = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
            'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
            'DIVISLAB.NS', 'DRREDDY.NS', 'CIPLA.NS', 'BRITANNIA.NS', 'DABUR.NS',
            'PIDILITIND.NS', 'HAVELLS.NS', 'ASTRAL.NS', 'DIXON.NS', 'TRENT.NS',
            'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS',
            'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'MPHASIS.NS', 'COFORGE.NS',
        ]
        
        # Initialize
        data_fetcher = DataFetcher()
        optimizer = OptimizationEngine(config)
        backtest = FastBacktest(config, data_fetcher)
        
        # Preload all data
        data_fetcher.preload_data(tickers, config.data_start)
        
        logger.info(f"\n🚀 Starting optimization loop")
        logger.info(f"Total stocks: {len(data_fetcher.data_pool)}")
        
        iteration = 0
        
        while config.should_continue():
            iteration += 1
            config.iterations_completed = iteration
            
            # Generate parameter combinations
            param_combos = optimizer.generate_parameter_combinations()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 ITERATION {iteration}")
            logger.info(f"Time remaining: {config.time_remaining()/60:.1f} minutes")
            logger.info(f"Testing {len(param_combos)} parameter combinations")
            logger.info(f"{'='*60}")
            
            for combo_idx, params in enumerate(param_combos):
                if not config.should_continue():
                    break
                
                # Apply parameters
                for key, value in params.items():
                    setattr(config, key, value)
                
                logger.info(f"\n  Testing combo {combo_idx+1}: {params}")
                
                # Run backtest on all stocks
                all_trades = []
                
                for ticker, df in data_fetcher.data_pool.items():
                    trades = backtest.run_backtest_with_params(df, ticker)
                    all_trades.extend(trades)
                
                # Calculate metrics
                metrics = backtest.calculate_metrics(all_trades)
                
                # Evaluate
                score = optimizer.evaluate_result(metrics, params)
                
                logger.info(f"  Trades: {metrics['total_trades']} | "
                          f"Win: {metrics['win_rate']:.0f}% | "
                          f"PF: {metrics['profit_factor']:.2f} | "
                          f"Return: {metrics['total_return']:.1f}% | "
                          f"Score: {score:.2f}")
                
                # Update best
                optimizer.update_best(params, metrics, score)
                
                # Save history
                optimizer.results_history.append({
                    'iteration': iteration,
                    'params': params,
                    'metrics': metrics,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                })
                
                optimizer.save_iteration_history()
                
                # Cleanup
                del all_trades
                gc.collect()
        
        # Final summary
        elapsed = time.time() - config.start_time
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"Total iterations: {config.iterations_completed}")
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"Best score: {config.best_score:.2f}")
        logger.info(f"Best parameters:")
        for key, value in config.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
        
        # Save final results
        final_results = pd.DataFrame(optimizer.results_history)
        if not final_results.empty:
            final_results.to_csv('nifty500_broom_breakout_results.csv', index=False)
            logger.info(f"\n✓ Results saved to nifty500_broom_breakout_results.csv")
        
        # Save best params summary
        summary = {
            'best_params': config.best_params,
            'best_score': config.best_score,
            'total_iterations': config.iterations_completed,
            'runtime_minutes': elapsed/60,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config.optimization_dir / 'final_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return final_results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        pd.DataFrame(columns=['ticker', 'total_trades', 'win_rate']).to_csv(
            'nifty500_broom_breakout_results.csv', index=False
        )
        
        sys.exit(1)

if __name__ == "__main__":
    main()
