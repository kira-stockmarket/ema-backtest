"""
QUICK LOOP - 5 Minute Learning Cycle
Runs for 5 minutes, saves state, commits to GitHub
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
import subprocess

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    def __init__(self):
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # 5 MINUTE LOOP
        self.loop_duration_seconds = 300  # 5 minutes
        self.start_time = time.time()
        
        self.backtest_start = '2018-01-01'
        self.data_start = '2000-01-01'
        self.ema_periods = [20, 50, 100, 200]
        
        self.data_dir = Path('data_cache')
        self.state_dir = Path('state')
        self.logs_dir = Path('logs')
        
        for dir_path in [self.data_dir, self.state_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load previous state
        self.load_state()
        
        # Learning parameters
        self.learning_rate = 0.03
        self.min_trades_before_learning = 10
        self.learning_window = 25
        self.parameter_bounds = {
            'broom_compression_threshold': (0.03, 0.15),
            'volume_threshold_multiplier': (1.2, 3.0),
            'stop_loss_buffer': (0.008, 0.04),
            'measured_move_multiplier': (1.2, 4.0),
            'box_consolidation_height': (0.08, 0.25),
        }
        
        self.iteration = 0
        self.total_trades = 0
        self.performance_history = []
    
    def load_state(self):
        """Load previous state"""
        try:
            params_file = self.state_dir / 'current_params.json'
            if params_file.exists():
                with open(params_file, 'r') as f:
                    data = json.load(f)
                
                self.learned_params = data.get('params', {})
                self.version = data.get('version', 0)
                self.total_trades = data.get('total_trades', 0)
                
                self.broom_compression_threshold = self.learned_params.get('broom_compression_threshold', 0.08)
                self.volume_threshold_multiplier = self.learned_params.get('volume_threshold_multiplier', 1.5)
                self.stop_loss_buffer = self.learned_params.get('stop_loss_buffer', 0.015)
                self.measured_move_multiplier = self.learned_params.get('measured_move_multiplier', 2.0)
                self.box_consolidation_height = self.learned_params.get('box_consolidation_height', 0.15)
            else:
                self.learned_params = {
                    'broom_compression_threshold': 0.08,
                    'volume_threshold_multiplier': 1.5,
                    'stop_loss_buffer': 0.015,
                    'measured_move_multiplier': 2.0,
                    'box_consolidation_height': 0.15,
                }
                self.version = 0
                self.broom_compression_threshold = 0.08
                self.volume_threshold_multiplier = 1.5
                self.stop_loss_buffer = 0.015
                self.measured_move_multiplier = 2.0
                self.box_consolidation_height = 0.15
            
            # Load performance history
            perf_file = self.state_dir / 'learning_progress.json'
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    self.performance_history = json.load(f)
            else:
                self.performance_history = []
        except Exception as e:
            print(f"Error loading state: {e}")
            self.learned_params = {
                'broom_compression_threshold': 0.08,
                'volume_threshold_multiplier': 1.5,
                'stop_loss_buffer': 0.015,
                'measured_move_multiplier': 2.0,
                'box_consolidation_height': 0.15,
            }
            self.version = 0
            self.total_trades = 0
            self.performance_history = []
            self.broom_compression_threshold = 0.08
            self.volume_threshold_multiplier = 1.5
            self.stop_loss_buffer = 0.015
            self.measured_move_multiplier = 2.0
            self.box_consolidation_height = 0.15
    
    def time_remaining(self) -> float:
        """Check remaining time in this 5-minute loop"""
        elapsed = time.time() - self.start_time
        return max(0, self.loop_duration_seconds - elapsed)
    
    def should_continue(self) -> bool:
        """Check if should continue"""
        return self.time_remaining() > 0

# ==================== LOGGING ====================

def setup_logging(config: Config):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logs_dir / 'quick_loop.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

config = Config()
logger = setup_logging(config)

# ==================== GITHUB COMMITTER ====================

class GitHubCommitter:
    """Commits state to GitHub"""
    
    def commit_state(self, message: str):
        """Commit state files to GitHub"""
        try:
            subprocess.run(['git', 'config', '--local', 'user.email', 'action@github.com'], 
                         check=False, capture_output=True)
            subprocess.run(['git', 'config', '--local', 'user.name', 'GitHub Action'], 
                         check=False, capture_output=True)
            
            subprocess.run(['git', 'add', 'state/'], check=False, capture_output=True)
            
            result = subprocess.run(['git', 'commit', '-m', message], 
                                  check=False, capture_output=True, text=True)
            
            if result.returncode == 0:
                subprocess.run(['git', 'push'], check=False, capture_output=True)
                logger.info(f"✅ Committed: {message}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Commit failed: {e}")
            return False

# ==================== DATA FETCHER ====================

class DataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_data(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, auto_adjust=False, timeout=30)
            
            if df is not None and not df.empty:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    df = df.dropna()
                    if len(df) > 300:
                        return df
            return None
        except:
            return None

# ==================== LEARNING ENGINE ====================

class LearningEngine:
    def __init__(self, config: Config):
        self.config = config
        self.trade_history = []
    
    def learn_from_trades(self, trades: List[Dict]):
        """Learn from trades"""
        if len(trades) < self.config.min_trades_before_learning:
            return
        
        df = pd.DataFrame(trades)
        winners = df[df['return_pct'] > 0]
        losers = df[df['return_pct'] <= 0]
        
        if len(winners) == 0 or len(losers) == 0:
            return
        
        win_rate = len(winners) / len(df) * 100
        profit_factor = winners['return_pct'].sum() / abs(losers['return_pct'].sum()) if len(losers) > 0 else float('inf')
        
        changes = {}
        
        # Learn compression
        if 'compression' in winners.columns and len(winners) > 3:
            winner_comp = winners['compression'].mean()
            current = self.config.broom_compression_threshold
            new_value = current + (winner_comp - current) * self.config.learning_rate
            
            min_val, max_val = self.config.parameter_bounds['broom_compression_threshold']
            new_value = max(min_val, min(max_val, new_value))
            changes['broom_compression_threshold'] = new_value
        
        # Learn volume
        if 'volume_ratio' in winners.columns and len(winners) > 3:
            winner_vol = winners['volume_ratio'].mean()
            loser_vol = losers['volume_ratio'].mean()
            
            if winner_vol > loser_vol:
                new_value = self.config.volume_threshold_multiplier * (1 + self.config.learning_rate * 0.5)
            else:
                new_value = self.config.volume_threshold_multiplier * (1 - self.config.learning_rate * 0.3)
            
            min_val, max_val = self.config.parameter_bounds['volume_threshold_multiplier']
            new_value = max(min_val, min(max_val, new_value))
            changes['volume_threshold_multiplier'] = new_value
        
        # Learn stop loss
        if win_rate < 45:
            new_value = self.config.stop_loss_buffer * (1 - self.config.learning_rate)
        elif win_rate > 60:
            new_value = self.config.stop_loss_buffer * (1 + self.config.learning_rate * 0.5)
        else:
            new_value = self.config.stop_loss_buffer
        
        min_val, max_val = self.config.parameter_bounds['stop_loss_buffer']
        new_value = max(min_val, min(max_val, new_value))
        changes['stop_loss_buffer'] = new_value
        
        # Learn target
        avg_winner = winners['return_pct'].mean()
        
        if avg_winner < 15:
            new_value = self.config.measured_move_multiplier * (1 + self.config.learning_rate)
        elif avg_winner > 30:
            new_value = self.config.measured_move_multiplier * (1 - self.config.learning_rate * 0.3)
        else:
            new_value = self.config.measured_move_multiplier
        
        min_val, max_val = self.config.parameter_bounds['measured_move_multiplier']
        new_value = max(min_val, min(max_val, new_value))
        changes['measured_move_multiplier'] = new_value
        
        # Apply changes
        for param, value in changes.items():
            setattr(self.config, param, value)
            self.config.learned_params[param] = value
        
        logger.info(f"🧠 Learning: Win {win_rate:.1f}% | PF {profit_factor:.2f}")
        for param, value in changes.items():
            logger.info(f"   {param}: {value:.4f}")
        
        return {'win_rate': win_rate, 'profit_factor': profit_factor}

# ==================== BACKTEST ENGINE ====================

class BacktestEngine:
    def __init__(self, config: Config, learning_engine: LearningEngine):
        self.config = config
        self.learning_engine = learning_engine
        self.data_fetcher = DataFetcher()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.config.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
        return df
    
    def create_higher_timeframes(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            weekly_df = df.resample('W-FRI').agg({'Close': 'last'}).dropna()
            weekly_df['EMA_200_Weekly'] = weekly_df['Close'].ewm(span=200, adjust=False).mean()
            
            monthly_df = df.resample('ME').agg({'Close': 'last'}).dropna()
            monthly_df['EMA_200_Monthly'] = monthly_df['Close'].ewm(span=200, adjust=False).mean()
            
            df['Weekly_200_EMA'] = weekly_df['EMA_200_Weekly'].reindex(df.index, method='ffill')
            df['Monthly_200_EMA'] = monthly_df['EMA_200_Monthly'].reindex(df.index, method='ffill')
        except:
            df['Weekly_200_EMA'] = np.nan
            df['Monthly_200_EMA'] = np.nan
        return df
    
    def run_quick_iteration(self, data_pool: Dict) -> Dict:
        """Run one quick iteration"""
        all_trades = []
        
        prepared_data = {}
        for ticker, df in data_pool.items():
            df = self.calculate_indicators(df.copy())
            df = self.create_higher_timeframes(df)
            prepared_data[ticker] = df
        
        all_dates = []
        for df in prepared_data.values():
            dates = df[df.index >= self.config.backtest_start].index
            all_dates.extend(dates)
        
        all_dates = sorted(set(all_dates))
        open_positions = {}
        
        for current_date in all_dates:
            # Check exits
            for ticker in list(open_positions.keys()):
                position = open_positions[ticker]
                df = prepared_data[ticker]
                
                if current_date not in df.index:
                    continue
                
                idx = df.index.get_loc(current_date)
                current_price = df.iloc[idx]['Close']
                current_high = df.iloc[idx]['High']
                current_low = df.iloc[idx]['Low']
                
                ema_200 = df.iloc[idx]['EMA_200']
                trailing_stop = ema_200 * (1 - self.config.stop_loss_buffer)
                if trailing_stop > position['stop_loss']:
                    position['stop_loss'] = trailing_stop
                
                exit_price = None
                
                if current_low <= position['stop_loss']:
                    exit_price = position['stop_loss']
                elif current_high >= position['take_profit']:
                    exit_price = position['take_profit']
                elif (current_date - position['entry_date']).days >= 60:
                    exit_price = current_price
                
                if exit_price:
                    return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    
                    trade = {
                        'return_pct': return_pct,
                        'compression': position['compression'],
                        'volume_ratio': position['volume_ratio'],
                    }
                    
                    all_trades.append(trade)
                    del open_positions[ticker]
            
            # Check entries
            for ticker, df in prepared_data.items():
                if ticker in open_positions:
                    continue
                
                if current_date not in df.index:
                    continue
                
                idx = df.index.get_loc(current_date)
                
                if idx < 250:
                    continue
                
                current_price = df.iloc[idx]['Close']
                
                weekly_ema = df.iloc[idx]['Weekly_200_EMA']
                if pd.isna(weekly_ema) or current_price <= weekly_ema:
                    continue
                
                emas = [df.iloc[idx][f'EMA_{p}'] for p in self.config.ema_periods]
                if any(pd.isna(e) for e in emas):
                    continue
                
                compression = (max(emas) - min(emas)) / current_price
                
                if compression >= self.config.broom_compression_threshold:
                    continue
                
                volume = df.iloc[idx]['Volume']
                volume_ma = df.iloc[idx]['Volume_MA_50']
                
                if pd.isna(volume_ma) or volume_ma <= 0:
                    continue
                
                volume_ratio = volume / volume_ma
                
                if volume_ratio < self.config.volume_threshold_multiplier:
                    continue
                
                entry_price = current_price
                stop_loss = df.iloc[idx]['EMA_200'] * (1 - self.config.stop_loss_buffer)
                
                lookback = df.iloc[max(0, idx-200):idx]
                peak_price = lookback['High'].max()
                peak_idx = df.index.get_loc(lookback['High'].idxmax())
                base_low = df.iloc[peak_idx:idx+1]['Low'].min()
                base_depth = peak_price - base_low
                take_profit = entry_price + (self.config.measured_move_multiplier * base_depth)
                
                open_positions[ticker] = {
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'compression': compression,
                    'volume_ratio': volume_ratio,
                }
        
        # Learn from trades
        if all_trades:
            perf = self.learning_engine.learn_from_trades(all_trades)
            self.config.total_trades += len(all_trades)
        
        # Calculate metrics
        if all_trades:
            df = pd.DataFrame(all_trades)
            winners = df[df['return_pct'] > 0]
            losers = df[df['return_pct'] <= 0]
            
            return {
                'total_trades': len(df),
                'win_rate': len(winners) / len(df) * 100,
                'profit_factor': winners['return_pct'].sum() / abs(losers['return_pct'].sum()) if len(losers) > 0 else float('inf')
            }
        
        return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0}

# ==================== MAIN ====================

def main():
    """Main 5-minute loop"""
    try:
        logger.info("=" * 80)
        logger.info("QUICK 5-MINUTE LEARNING LOOP")
        logger.info(f"Loop duration: {config.loop_duration_seconds} seconds")
        logger.info(f"Version: {config.version}")
        logger.info(f"Params: {config.learned_params}")
        logger.info("=" * 80)
        
        committer = GitHubCommitter()
        
        # Universe (20 stocks for speed)
        tickers = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
            'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
        ]
        
        # Load data
        data_fetcher = DataFetcher()
        data_pool = {}
        
        logger.info("📦 Loading data...")
        for ticker in tickers:
            df = data_fetcher.fetch_data(ticker, config.data_start)
            if df is not None and len(df) > 300:
                data_pool[ticker] = df
            time.sleep(0.2)
        
        logger.info(f"✓ Loaded {len(data_pool)} stocks")
        
        learning_engine = LearningEngine(config)
        backtest = BacktestEngine(config, learning_engine)
        
        # Run iterations until 5 minutes is up
        while config.should_continue():
            config.iteration += 1
            
            logger.info(f"\n🔄 Iteration {config.iteration} | Time left: {config.time_remaining():.0f}s")
            
            metrics = backtest.run_quick_iteration(data_pool)
            
            logger.info(f"   Trades: {metrics['total_trades']} | Win: {metrics['win_rate']:.1f}% | PF: {metrics['profit_factor']:.2f}")
            
            # Save state every iteration
            state = {
                'params': config.learned_params,
                'version': config.version + 1,
                'updated_at': datetime.now().isoformat(),
                'total_trades': config.total_trades,
                'iteration': config.iteration,
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
            }
            
            # Save to files
            with open(config.state_dir / 'current_params.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save progress history
            config.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'iteration': config.iteration,
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'params': config.learned_params.copy(),
            })
            
            with open(config.state_dir / 'learning_progress.json', 'w') as f:
                json.dump(config.performance_history[-50:], f, indent=2, default=str)
            
            config.version += 1
            
            time.sleep(1)
        
        # Final commit
        logger.info("\n" + "=" * 80)
        logger.info("5-MINUTE LOOP COMPLETE")
        logger.info(f"Iterations: {config.iteration}")
        logger.info(f"Total trades: {config.total_trades}")
        logger.info(f"Params: {config.learned_params}")
        logger.info("=" * 80)
        
        # Final commit to GitHub
        committer.commit_state(f"Learning update - Iteration {config.iteration}")
        
        # Save CSV
        pd.DataFrame([config.learned_params]).to_csv('nifty500_broom_breakout_results.csv', index=False)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
