"""
FIXED: Continuous Learning Loop
Continues from where it left off
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
        
        # Loop duration
        self.loop_duration_seconds = int(os.getenv('LOOP_DURATION', '280'))  # 4.5 minutes
        self.start_time = time.time()
        
        self.backtest_start = '2018-01-01'
        self.data_start = '2000-01-01'
        self.ema_periods = [20, 50, 100, 200]
        
        # Directories
        self.data_dir = Path('data_cache')
        self.state_dir = Path('state')
        self.logs_dir = Path('logs')
        
        for dir_path in [self.data_dir, self.state_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load state
        self.load_state()
        
        # Learning parameters
        self.learning_rate = 0.05
        self.min_trades_to_learn = 5
        self.parameter_bounds = {
            'broom_compression_threshold': (0.03, 0.15),
            'volume_threshold_multiplier': (1.2, 3.0),
            'stop_loss_buffer': (0.008, 0.04),
            'measured_move_multiplier': (1.2, 4.0),
        }
    
    def load_state(self):
        """Load state from previous runs"""
        try:
            # Load params
            params_file = self.state_dir / 'current_params.json'
            if params_file.exists():
                with open(params_file, 'r') as f:
                    data = json.load(f)
                self.learned_params = data.get('params', self.get_default_params())
                self.version = data.get('version', 0)
            else:
                self.learned_params = self.get_default_params()
                self.version = 0
            
            # Load ALL trades
            trades_file = self.state_dir / 'all_trades.json'
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    self.all_trades = json.load(f)
            else:
                self.all_trades = []
            
            # Load learning progress
            progress_file = self.state_dir / 'learning_progress.json'
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    self.learning_progress = json.load(f)
            else:
                self.learning_progress = []
            
            # Apply params
            self.broom_compression_threshold = self.learned_params.get('broom_compression_threshold', 0.08)
            self.volume_threshold_multiplier = self.learned_params.get('volume_threshold_multiplier', 1.5)
            self.stop_loss_buffer = self.learned_params.get('stop_loss_buffer', 0.015)
            self.measured_move_multiplier = self.learned_params.get('measured_move_multiplier', 2.0)
            
            print(f"✓ State loaded: v{self.version}, {len(self.all_trades)} historical trades")
            print(f"✓ Params: {self.learned_params}")
            
        except Exception as e:
            print(f"Error loading state: {e}")
            self.learned_params = self.get_default_params()
            self.version = 0
            self.all_trades = []
            self.learning_progress = []
            self.broom_compression_threshold = 0.08
            self.volume_threshold_multiplier = 1.5
            self.stop_loss_buffer = 0.015
            self.measured_move_multiplier = 2.0
    
    def get_default_params(self):
        return {
            'broom_compression_threshold': 0.08,
            'volume_threshold_multiplier': 1.5,
            'stop_loss_buffer': 0.015,
            'measured_move_multiplier': 2.0,
        }
    
    def save_state(self, metrics):
        """Save state for next run"""
        try:
            state = {
                'params': self.learned_params,
                'version': self.version + 1,
                'updated_at': datetime.now().isoformat(),
                'total_trades': len(self.all_trades),
                'last_metrics': metrics,
            }
            
            with open(self.state_dir / 'current_params.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            with open(self.state_dir / 'all_trades.json', 'w') as f:
                json.dump(self.all_trades[-500:], f, indent=2, default=str)
            
            self.learning_progress.append({
                'timestamp': datetime.now().isoformat(),
                'version': self.version,
                'total_trades': len(self.all_trades),
                'metrics': metrics,
                'params': self.learned_params.copy(),
            })
            
            with open(self.state_dir / 'learning_progress.json', 'w') as f:
                json.dump(self.learning_progress[-100:], f, indent=2, default=str)
            
            self.version += 1
            print(f"✓ State saved: v{self.version}")
            
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def time_remaining(self):
        return max(0, self.loop_duration_seconds - (time.time() - self.start_time))

# ==================== LOGGING ====================

def setup_logging(config):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logs_dir / 'loop.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

config = Config()
logger = setup_logging(config)

# ==================== LEARNER ====================

class Learner:
    def __init__(self, config):
        self.config = config
    
    def learn(self):
        """Learn from all historical trades"""
        if len(self.config.all_trades) < self.config.min_trades_to_learn:
            return False
        
        df = pd.DataFrame(self.config.all_trades)
        recent = df.tail(50)
        
        winners = recent[recent['return_pct'] > 0]
        losers = recent[recent['return_pct'] <= 0]
        
        if len(winners) < 3 or len(losers) < 3:
            return False
        
        win_rate = len(winners) / len(recent) * 100
        
        changes = {}
        
        # Learn compression
        if 'compression' in winners.columns:
            winner_comp = winners['compression'].median()
            current = self.config.broom_compression_threshold
            new_value = current + (winner_comp - current) * self.config.learning_rate
            
            min_val, max_val = self.config.parameter_bounds['broom_compression_threshold']
            new_value = max(min_val, min(max_val, new_value))
            
            if abs(new_value - current) > 0.001:
                changes['broom_compression_threshold'] = new_value
        
        # Learn stop loss
        if win_rate < 40:
            changes['stop_loss_buffer'] = max(0.008, self.config.stop_loss_buffer * 0.9)
        elif win_rate > 55:
            changes['stop_loss_buffer'] = min(0.04, self.config.stop_loss_buffer * 1.1)
        
        # Learn target
        avg_winner = winners['return_pct'].mean()
        if avg_winner > 20:
            changes['measured_move_multiplier'] = min(4.0, self.config.measured_move_multiplier * 1.1)
        elif avg_winner < 8:
            changes['measured_move_multiplier'] = max(1.2, self.config.measured_move_multiplier * 0.9)
        
        # Apply changes
        for param, value in changes.items():
            setattr(self.config, param, value)
            self.config.learned_params[param] = value
        
        if changes:
            logger.info(f"🧠 Learned: Win={win_rate:.1f}% | Changes: {changes}")
            return True
        
        return False

# ==================== DATA FETCHER ====================

class DataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })
    
    def fetch_data(self, ticker, start_date):
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, auto_adjust=False, timeout=30)
            
            if df is not None and not df.empty:
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required):
                    df = df[required].dropna()
                    if len(df) > 300:
                        return df
            return None
        except:
            return None

# ==================== BACKTEST ====================

class Backtest:
    def __init__(self, config):
        self.config = config
        self.data_fetcher = DataFetcher()
    
    def calculate_indicators(self, df):
        for period in self.config.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
        return df
    
    def generate_trades(self, data_pool):
        """Generate trades with current params"""
        new_trades = []
        open_positions = {}
        
        for ticker, df in data_pool.items():
            df = self.calculate_indicators(df.copy())
            
            all_dates = df[df.index >= self.config.backtest_start].index
            
            for current_date in all_dates:
                if current_date not in df.index:
                    continue
                
                idx = df.index.get_loc(current_date)
                
                if idx < 250:
                    continue
                
                current_price = df.iloc[idx]['Close']
                
                # Simple setup check
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
                
                # Simulate trade
                entry_price = current_price
                ema_200 = df.iloc[idx]['EMA_200']
                stop_loss = ema_200 * (1 - self.config.stop_loss_buffer)
                
                # Find future exit (simplified)
                future_data = df.iloc[idx+1:min(idx+60, len(df))]
                
                if len(future_data) < 5:
                    continue
                
                exit_price = future_data['Close'].iloc[-1]
                return_pct = (exit_price - entry_price) / entry_price * 100
                
                new_trades.append({
                    'ticker': ticker,
                    'return_pct': return_pct,
                    'compression': compression,
                    'volume_ratio': volume_ratio,
                })
        
        return new_trades

# ==================== MAIN ====================

def main():
    try:
        logger.info("=" * 60)
        logger.info(f"LOOP START - v{config.version}")
        logger.info(f"Historical trades: {len(config.all_trades)}")
        logger.info(f"Current params: {config.learned_params}")
        logger.info("=" * 60)
        
        # Universe
        tickers = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
            'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
        ]
        
        # Load data
        data_fetcher = DataFetcher()
        data_pool = {}
        
        for ticker in tickers:
            df = data_fetcher.fetch_data(ticker, config.data_start)
            if df is not None and len(df) > 300:
                data_pool[ticker] = df
            time.sleep(0.2)
        
        logger.info(f"✓ {len(data_pool)} stocks loaded")
        
        # Learn from previous trades
        learner = Learner(config)
        learner.learn()
        
        # Generate new trades
        backtest = Backtest(config)
        new_trades = backtest.generate_trades(data_pool)
        
        logger.info(f"✓ Generated {len(new_trades)} new trades")
        
        # Add to history
        config.all_trades.extend(new_trades)
        
        # Save state
        metrics = {
            'new_trades': len(new_trades),
            'total_trades': len(config.all_trades),
            'params': config.learned_params,
        }
        config.save_state(metrics)
        
        # Save CSV (not ignored)
        pd.DataFrame([config.learned_params]).to_csv('nifty500_broom_breakout_results.csv', index=False)
        
        # Commit
        try:
            subprocess.run(['git', 'config', '--local', 'user.email', 'action@github.com'], check=False)
            subprocess.run(['git', 'config', '--local', 'user.name', 'GitHub Action'], check=False)
            subprocess.run(['git', 'add', '-f', 'state/', 'nifty500_broom_breakout_results.csv'], check=False)
            subprocess.run(['git', 'commit', '-m', f'Update v{config.version}: {len(config.all_trades)} trades'], check=False)
            subprocess.run(['git', 'push'], check=False)
            logger.info("✅ Committed")
        except Exception as e:
            logger.warning(f"Commit: {e}")
        
        logger.info("✅ LOOP COMPLETE")
        
    except Exception as e:
        logger.error(f"Fatal: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
