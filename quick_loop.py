"""
7-Dimension Continuous Learning Loop (Indian Market Optimizer)
Optimizes EMA, Volume, SL, TP, and new Structural Base parameters via Head-to-Head combat.
"""

import pandas as pd
import numpy as np
import warnings
import time
import logging
import os
import sys
from datetime import datetime
import json
import subprocess
import random
from pathlib import Path

warnings.filterwarnings('ignore')

class Config:
    def __init__(self):
        self.loop_duration_seconds = int(os.getenv('LOOP_DURATION', '20880'))  
        self.start_time = time.time()
        self.data_start = '2015-01-01' 
        self.ema_periods = [20, 50, 100, 200]
        
        self.data_dir = Path('data_cache')
        self.state_dir = Path('state')
        for dir_path in [self.data_dir, self.state_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.load_state()
        
        # New 7-Dimension Bounds for Indian Markets
        self.parameter_bounds = {
            'broom_compression_threshold': (0.05, 0.20),
            'volume_threshold_multiplier': (1.0, 3.0),
            'stop_loss_buffer': (0.005, 0.05),
            'measured_move_multiplier': (1.0, 5.0),
            'base_duration_min': (20, 100),       # Int days
            'base_duration_max': (100, 300),      # Int days
            'box_consolidation_height': (0.10, 0.50)
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
            
            # Apply params dynamically
            for k, v in self.learned_params.items():
                setattr(self, k, v)
                
        except Exception:
            self.learned_params = self.get_default_params()
            self.version = 0

    def get_default_params(self):
        # Starting with your high-CAGR baseline
        return {
            'broom_compression_threshold': 0.15,
            'base_duration_min': 63,
            'base_duration_max': 200,
            'box_consolidation_height': 0.25,
            'volume_threshold_multiplier': 1.2,
            'stop_loss_buffer': 0.01,
            'measured_move_multiplier': 1.5
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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
    param_keys = list(config.learned_params.keys())
    key_to_mutate = random.choice(param_keys)
    bounds = config.parameter_bounds[key_to_mutate]
    
    original_val = config.learned_params[key_to_mutate]
    mutation_pct = random.uniform(-0.10, 0.10) # 10% swing for exploration
    
    if key_to_mutate in ['base_duration_min', 'base_duration_max']:
        # Integer mutation logic
        test_val = int(round(original_val * (1 + mutation_pct)))
        if test_val == original_val:
            test_val += random.choice([-1, 1])
        test_val = max(bounds[0], min(bounds[1], test_val))
    else:
        # Float mutation logic
        test_val = max(bounds[0], min(bounds[1], original_val + (mutation_pct * original_val)))
    
    slice_seed = random.randint(1, 1000000)
    
    def evaluate_strategy(test_config_value):
        trades = []
        setattr(config, key_to_mutate, test_config_value)
        
        for ticker, df in data_pool.items():
            for period in config.ema_periods:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
            
            if len(df) < 500: continue
            
            random.seed(slice_seed) 
            start_idx = random.randint(300, len(df) - 100)
            
            for idx in range(start_idx, start_idx + 80):
                current_price = df.iloc[idx]['Close']
                ema_200 = df.iloc[idx]['EMA_200']
                
                if current_price <= ema_200: continue
                
                emas = [df.iloc[idx][f'EMA_{p}'] for p in config.ema_periods]
                compression = (max(emas) - min(emas)) / current_price
                if compression >= config.broom_compression_threshold: continue
                
                volume_ma = df.iloc[idx]['Volume_MA_50']
                if volume_ma <= 0: continue
                volume_ratio = df.iloc[idx]['Volume'] / volume_ma
                if volume_ratio < config.volume_threshold_multiplier: continue
                
                # --- NEW STRUCTURAL FILTERS ---
                lookback = df['High'].iloc[max(0, idx - config.base_duration_max):idx]
                if lookback.empty: continue
                
                peak_price = lookback.max()
                peak_date = lookback.idxmax()
                peak_idx = df.index.get_loc(peak_date)
                
                base_duration = idx - peak_idx
                if not (config.base_duration_min <= base_duration <= config.base_duration_max):
                    continue
                
                base_low = df['Low'].iloc[peak_idx:idx+1].min()
                base_depth = peak_price - base_low
                if (base_depth / peak_price) > config.box_consolidation_height:
                    continue
                
                # Setup passed, simulate trade forward 40 days max
                future_data = df.iloc[idx+1:idx+41]
                if not future_data.empty:
                    tp = current_price + (config.measured_move_multiplier * base_depth)
                    sl = ema_200 * (1 - config.stop_loss_buffer)
                    
                    for f_idx, row in future_data.iterrows():
                        if row['High'] >= tp:
                            trades.append(((tp / current_price) - 1) * 100)
                            break
                        elif row['Low'] <= sl:
                            trades.append(((sl / current_price) - 1) * 100)
                            break
                    else:
                        # Trade didn't close in 40 days, log current PnL
                        trades.append(((future_data.iloc[-1]['Close'] / current_price) - 1) * 100)
        return trades

    old_trades = evaluate_strategy(original_val)
    new_trades = evaluate_strategy(test_val)
    random.seed()

    def calc_win_rate(trades_list):
        if not trades_list: return 0
        return len([t for t in trades_list if t > 0]) / len(trades_list) * 100

    old_win_rate = calc_win_rate(old_trades)
    new_win_rate = calc_win_rate(new_trades)
    
    if new_win_rate > old_win_rate and len(new_trades) >= 5:
        config.learned_params[key_to_mutate] = test_val
        setattr(config, key_to_mutate, test_val)
        logger.info(f"🏆 UPGRADE! Old WR: {old_win_rate:.1f}% ({len(old_trades)} tr) -> New WR: {new_win_rate:.1f}% ({len(new_trades)} tr). Kept {key_to_mutate}: {test_val}")
    else:
        config.learned_params[key_to_mutate] = original_val
        setattr(config, key_to_mutate, original_val)

def main():
    logger.info(f"STARTING 7-DIMENSION OPTIMIZER LOOP - v{config.version}")
    
    tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'TATAMOTORS.NS', 'M&M.NS', 'SUNPHARMA.NS', 'TATASTEEL.NS', 'NTPC.NS' # Added 5 more for better sample sizes
    ]
    
    data_pool = {}
    fetcher = DataFetcher()
    for ticker in tickers:
        df = fetcher.fetch_data(ticker, config.data_start)
        if df is not None: data_pool[ticker] = df
    
    logger.info(f"Loaded {len(data_pool)} stocks. Beginning mutations...")
    
    last_commit_time = time.time()
    iterations = 0
    
    while config.time_remaining() > 0:
        backtest_and_learn(data_pool, config)
        iterations += 1
        
        if time.time() - last_commit_time > 1800:
            config.save_state({'iterations': iterations})
            try:
                subprocess.run(['git', 'config', '--local', 'user.email', 'action@github.com'], check=False)
                subprocess.run(['git', 'config', '--local', 'user.name', 'GitHub Action'], check=False)
                subprocess.run(['git', 'add', '-f', 'state/current_params.json'], check=False)
                subprocess.run(['git', 'commit', '-m', f'Auto-learn upgrade v{config.version}'], check=False)
                subprocess.run(['git', 'push', 'origin', 'main'], check=False)
                last_commit_time = time.time()
            except Exception:
                pass
        time.sleep(1)
        
    config.save_state({'iterations': iterations})
    try:
        subprocess.run(['git', 'add', '-f', 'state/current_params.json'], check=False)
        subprocess.run(['git', 'commit', '-m', f'Final loop update v{config.version}'], check=False)
        subprocess.run(['git', 'push', 'origin', 'main'], check=False)
    except:
        pass

if __name__ == "__main__":
    main()
