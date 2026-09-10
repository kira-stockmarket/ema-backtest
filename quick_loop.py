"""
FIXED: 5.8-Hour Continuous Learning Loop with True "Apples-to-Apples" Evaluation.
Runs continuously, optimizes parameters by proving they beat the baseline, and commits periodically.
"""

import pandas as pd
import numpy as np
import warnings
import time
import logging
import os
import sys
from datetime import datetime
import requests
from pathlib import Path
import json
import subprocess
import random

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    def __init__(self):
        # 5.8 hours = 20880 seconds
        self.loop_duration_seconds = int(os.getenv('LOOP_DURATION', '20880'))  
        self.start_time = time.time()
        
        self.data_start = '2015-01-01' # Fetch enough data for robust random slices
        self.ema_periods = [20, 50, 100, 200]
        
        self.data_dir = Path('data_cache')
        self.state_dir = Path('state')
        self.logs_dir = Path('logs')
        
        for dir_path in [self.data_dir, self.state_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.load_state()
        
        self.parameter_bounds = {
            'broom_compression_threshold': (0.03, 0.15),
            'volume_threshold_multiplier': (1.2, 3.5),
            'stop_loss_buffer': (0.008, 0.05),
            'measured_move_multiplier': (1.2, 5.0),
        }
    
    def load_state(self):
        try:
            params_file = self.state_dir / 'current_params.json'
            if params_file.exists():
                with open(params_file, 'r') as f:
                    data = json.load(f)
                self.learned_params = data.get('params', self.get_default_params())
                self.version = data.get('version', 0)
            else:
                self.learned_params = self.get_default_params()
                self.version = 0
            
            # Apply params
            self.broom_compression_threshold = self.learned_params.get('broom_compression_threshold', 0.08)
            self.volume_threshold_multiplier = self.learned_params.get('volume_threshold_multiplier', 1.5)
            self.stop_loss_buffer = self.learned_params.get('stop_loss_buffer', 0.015)
            self.measured_move_multiplier = self.learned_params.get('measured_move_multiplier', 2.0)
            
        except Exception:
            self.learned_params = self.get_default_params()
            self.version = 0

    def get_default_params(self):
        return {
            'broom_compression_threshold': 0.08,
            'volume_threshold_multiplier': 1.5,
            'stop_loss_buffer': 0.015,
            'measured_move_multiplier': 2.0,
        }
    
    def save_state(self, metrics):
        try:
            state = {
                'params': self.learned_params,
                'version': self.version + 1,
                'updated_at': datetime.now().isoformat(),
                'last_metrics': metrics,
            }
            with open(self.state_dir / 'current_params.json', 'w') as f:
                json.dump(state, f, indent=2)
            self.version += 1
        except Exception:
            pass
    
    def time_remaining(self):
        return max(0, self.loop_duration_seconds - (time.time() - self.start_time))

# ==================== LOGGING ====================
def setup_logging(config):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger(__name__)

config = Config()
logger = setup_logging(config)

# ==================== CORE LOGIC ====================
class DataFetcher:
    def fetch_data(self, ticker, start_date):
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(start=start_date, auto_adjust=False, timeout=30)
            if df is not None and not df.empty and len(df) > 300:
                return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        except:
            pass
        return None

def backtest_and_learn(data_pool, config):
    """Simulates trades on a fixed subset, comparing NEW vs OLD parameters (Apples-to-Apples)"""
    
    # 1. Mutate parameters slightly
    param_keys = list(config.learned_params.keys())
    key_to_mutate = random.choice(param_keys)
    bounds = config.parameter_bounds[key_to_mutate]
    
    original_val = config.learned_params[key_to_mutate]
    mutation = random.uniform(-0.05, 0.05) * original_val
    test_val = max(bounds[0], min(bounds[1], original_val + mutation))
    
    # Generate a random seed for this specific iteration slice
    slice_seed = random.randint(1, 1000000)
    
    # Helper function to run the strategy on a specific set of parameters
    def evaluate_strategy(test_config_value):
        trades = []
        setattr(config, key_to_mutate, test_config_value)
        
        for ticker, df in data_pool.items():
            for period in config.ema_periods:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
            
            if len(df) < 500: continue
            
            # Use the slice_seed so OLD and NEW are tested on the exact same days
            random.seed(slice_seed) 
            start_idx = random.randint(250, len(df) - 100)
            
            for idx in range(start_idx, start_idx + 60):
                current_price = df.iloc[idx]['Close']
                emas = [df.iloc[idx][f'EMA_{p}'] for p in config.ema_periods]
                compression = (max(emas) - min(emas)) / current_price
                
                if compression >= config.broom_compression_threshold: continue
                
                volume_ma = df.iloc[idx]['Volume_MA_50']
                if volume_ma <= 0: continue
                volume_ratio = df.iloc[idx]['Volume'] / volume_ma
                
                if volume_ratio < config.volume_threshold_multiplier: continue
                
                future_data = df.iloc[idx+1:idx+20]
                if not future_data.empty:
                    return_pct = (future_data['Close'].iloc[-1] - current_price) / current_price * 100
                    trades.append(return_pct)
        return trades

    # 2. Run Apples-to-Apples Comparison
    old_trades = evaluate_strategy(original_val)
    new_trades = evaluate_strategy(test_val)
    
    # Reset random seed back to system time for future iterations
    random.seed()

    # 3. Calculate Win Rates
    def calc_win_rate(trades_list):
        if not trades_list: return 0
        return len([t for t in trades_list if t > 0]) / len(trades_list) * 100

    old_win_rate = calc_win_rate(old_trades)
    new_win_rate = calc_win_rate(new_trades)
    
    # 4. STRICT EVALUATION: Does the mutation actually beat the baseline?
    if new_win_rate > old_win_rate and len(new_trades) >= 10:
        # It genuinely improved! Keep it.
        config.learned_params[key_to_mutate] = test_val
        setattr(config, key_to_mutate, test_val)
        logger.info(f"🏆 GENUINE UPGRADE! Old WR: {old_win_rate:.1f}% ({len(old_trades)} trades) -> New WR: {new_win_rate:.1f}% ({len(new_trades)} trades). Kept {key_to_mutate}: {test_val:.4f}")
    else:
        # Failed to beat the baseline, or not enough trades. Revert it.
        config.learned_params[key_to_mutate] = original_val
        setattr(config, key_to_mutate, original_val)

# ==================== MAIN LOOP ====================
def main():
    logger.info(f"STARTING 5.8 HOUR TRUE LEARNING LOOP - v{config.version}")
    
    # Expand universe slightly to ensure we generate enough trades per slice
    tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    ]
    
    data_pool = {}
    fetcher = DataFetcher()
    for ticker in tickers:
        df = fetcher.fetch_data(ticker, config.data_start)
        if df is not None: data_pool[ticker] = df
    
    logger.info(f"Loaded {len(data_pool)} stocks into memory. Beginning mutations...")
    
    last_commit_time = time.time()
    iterations = 0
    
    while config.time_remaining() > 0:
        backtest_and_learn(data_pool, config)
        iterations += 1
        
        # Commit every 30 minutes
        if time.time() - last_commit_time > 1800:
            config.save_state({'iterations': iterations})
            try:
                subprocess.run(['git', 'config', '--local', 'user.email', 'action@github.com'], check=False)
                subprocess.run(['git', 'config', '--local', 'user.name', 'GitHub Action'], check=False)
                subprocess.run(['git', 'add', '-f', 'state/current_params.json'], check=False)
                subprocess.run(['git', 'commit', '-m', f'Auto-learn upgrade v{config.version}'], check=False)
                subprocess.run(['git', 'push'], check=False)
                logger.info("Intermediate state pushed to GitHub.")
                last_commit_time = time.time()
            except Exception as e:
                logger.warning(f"Commit failed: {e}")
                
        # Sleep briefly to prevent maxing out CPU needlessly
        time.sleep(1)
        
    # Final save and commit
    config.save_state({'iterations': iterations})
    try:
        subprocess.run(['git', 'add', '-f', 'state/current_params.json'], check=False)
        subprocess.run(['git', 'commit', '-m', f'Final loop update v{config.version}'], check=False)
        subprocess.run(['git', 'push'], check=False)
    except:
        pass
    logger.info("LOOP COMPLETE.")

if __name__ == "__main__":
    main()
