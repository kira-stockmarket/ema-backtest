"""
TRUE INCREMENTAL LEARNING LOOP
Learns from trades across iterations
Saves state and improves each run
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
        self.loop_duration_seconds = int(os.getenv('LOOP_DURATION', '300'))  # 5 minutes default
        self.start_time = time.time()
        
        self.backtest_start = '2018-01-01'
        self.data_start = '2000-01-01'
        self.ema_periods = [20, 50, 100, 200]
        
        self.data_dir = Path('data_cache')
        self.state_dir = Path('state')
        self.logs_dir = Path('logs')
        
        for dir_path in [self.data_dir, self.state_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # ===== CRITICAL: Load ALL previous state =====
        self.load_full_state()
        
        # Learning parameters
        self.learning_rate = 0.05
        self.min_trades_to_learn = 5  # Learn after 5 trades
        self.parameter_bounds = {
            'broom_compression_threshold': (0.03, 0.15),
            'volume_threshold_multiplier': (1.2, 3.0),
            'stop_loss_buffer': (0.008, 0.04),
            'measured_move_multiplier': (1.2, 4.0),
        }
        
        self.iteration = 0
        self.new_trades_this_run = 0
    
    def load_full_state(self):
        """Load COMPLETE state from previous runs"""
        try:
            # Load params
            params_file = self.state_dir / 'current_params.json'
            if params_file.exists():
                with open(params_file, 'r') as f:
                    data = json.load(f)
                self.learned_params = data.get('params', {})
                self.version = data.get('version', 0)
                self.total_trades_historical = data.get('total_trades', 0)
            else:
                self.learned_params = self.get_default_params()
                self.version = 0
                self.total_trades_historical = 0
            
            # Load ALL trade history (CRITICAL for learning)
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
            
            # Apply loaded params
            self.broom_compression_threshold = self.learned_params.get('broom_compression_threshold', 0.08)
            self.volume_threshold_multiplier = self.learned_params.get('volume_threshold_multiplier', 1.5)
            self.stop_loss_buffer = self.learned_params.get('stop_loss_buffer', 0.015)
            self.measured_move_multiplier = self.learned_params.get('measured_move_multiplier', 2.0)
            
            print(f"✓ Loaded state: v{self.version}, {len(self.all_trades)} historical trades")
            print(f"✓ Current params: {self.learned_params}")
            
        except Exception as e:
            print(f"Error loading state: {e}")
            self.learned_params = self.get_default_params()
            self.version = 0
            self.total_trades_historical = 0
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
    
    def save_full_state(self, metrics):
        """Save COMPLETE state"""
        try:
            # Save params
            state = {
                'params': self.learned_params,
                'version': self.version + 1,
                'updated_at': datetime.now().isoformat(),
                'total_trades': len(self.all_trades),
                'total_trades_this_run': self.new_trades_this_run,
                'last_metrics': metrics,
            }
            
            with open(self.state_dir / 'current_params.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save ALL trades (CRITICAL)
            with open(self.state_dir / 'all_trades.json', 'w') as f:
                json.dump(self.all_trades[-500:], f, indent=2, default=str)  # Last 500 trades
            
            # Save learning progress
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
            
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def time_remaining(self):
        return max(0, self.loop_duration_seconds - (time.time() - self.start_time))
    
    def should_continue(self):
        return self.time_remaining() > 0

# ==================== LOGGING ====================

def setup_logging(config):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logs_dir / 'incremental_learning.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

config = Config()
logger = setup_logging(config)

# ==================== INCREMENTAL LEARNER ====================

class IncrementalLearner:
    """Learns from ALL historical trades across iterations"""
    
    def __init__(self, config):
        self.config = config
    
    def learn_from_all_trades(self):
        """Learn from all historical trades"""
        if len(self.config.all_trades) < self.config.min_trades_to_learn:
            logger.info(f"Not enough trades to learn ({len(self.config.all_trades)} < {self.config.min_trades_to_learn})")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(self.config.all_trades)
        
        # Use RECENT trades for learning (last 50)
        recent = df.tail(50)
        
        winners = recent[recent['return_pct'] > 0]
        losers = recent[recent['return_pct'] <= 0]
        
        if len(winners) < 3 or len(losers) < 3:
            logger.info("Not enough winners/losers to learn")
            return False
        
        win_rate = len(winners) / len(recent) * 100
        profit_factor = winners['return_pct'].sum() / abs(losers['return_pct'].sum()) if len(losers) > 0 else 0
        
        logger.info(f"\n🧠 LEARNING FROM {len(recent)} RECENT TRADES")
        logger.info(f"   Win rate: {win_rate:.1f}%")
        logger.info(f"   Profit factor: {profit_factor:.2f}")
        
        changes = {}
        
        # 1. Learn optimal compression from WINNERS
        if 'compression' in winners.columns:
            winner_comp = winners['compression'].median()
            current = self.config.broom_compression_threshold
            
            # Move threshold towards median of winners
            new_value = current + (winner_comp - current) * self.config.learning_rate
            
            min_val, max_val = self.config.parameter_bounds['broom_compression_threshold']
            new_value = max(min_val, min(max_val, new_value))
            
            if abs(new_value - current) > 0.001:
                changes['broom_compression_threshold'] = new_value
                logger.info(f"   Compression: {current:.4f} → {new_value:.4f} (winners median: {winner_comp:.4f})")
        
        # 2. Learn volume threshold
        if 'volume_ratio' in winners.columns and 'volume_ratio' in losers.columns:
            winner_vol = winners['volume_ratio'].median()
            loser_vol = losers['volume_ratio'].median()
            
            current = self.config.volume_threshold_multiplier
            
            if winner_vol > loser_vol * 1.1:
                # Winners have significantly higher volume
                new_value = min(current * 1.1, max_val)
                changes['volume_threshold_multiplier'] = new_value
                logger.info(f"   Volume: {current:.2f} → {new_value:.2f} (winners: {winner_vol:.2f}x, losers: {loser_vol:.2f}x)")
            elif winner_vol < loser_vol * 0.9:
                # Winners have lower volume - relax requirement
                new_value = max(current * 0.9, min_val)
                changes['volume_threshold_multiplier'] = new_value
                logger.info(f"   Volume: {current:.2f} → {new_value:.2f} (winners: {winner_vol:.2f}x, losers: {loser_vol:.2f}x)")
        
        # 3. Learn stop loss based on win rate
        current_stop = self.config.stop_loss_buffer
        
        if win_rate < 40:
            # Too many losers - tighten stop
            new_stop = max(current_stop * 0.9, 0.008)
            changes['stop_loss_buffer'] = new_stop
            logger.info(f"   Stop: {current_stop:.4f} → {new_stop:.4f} (tightening)")
        elif win_rate > 55:
            # Good win rate - can loosen stop
            new_stop = min(current_stop * 1.1, 0.04)
            changes['stop_loss_buffer'] = new_stop
            logger.info(f"   Stop: {current_stop:.4f} → {new_stop:.4f} (loosening)")
        
        # 4. Learn target based on avg winner
        avg_winner = winners['return_pct'].mean()
        current_target = self.config.measured_move_multiplier
        
        if avg_winner > 20:
            # Winners running well - increase target
            new_target = min(current_target * 1.1, 4.0)
            changes['measured_move_multiplier'] = new_target
            logger.info(f"   Target: {current_target:.2f} → {new_target:.2f} (avg winner: {avg_winner:.1f}%)")
        elif avg_winner < 8:
            # Winners too small - reduce target
            new_target = max(current_target * 0.9, 1.2)
            changes['measured_move_multiplier'] = new_target
            logger.info(f"   Target: {current_target:.2f} → {new_target:.2f} (avg winner: {avg_winner:.1f}%)")
        
        # Apply changes
        if changes:
            for param, value in changes.items():
                setattr(self.config, param, value)
                self.config.learned_params[param] = value
            
            logger.info(f"\n✅ PARAMETERS UPDATED: {changes}")
            return True
        else:
            logger.info("\nNo parameter changes needed")
            return False

# ==================== DATA FETCHER ====================

class DataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_data(self, ticker, start_date):
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

# ==================== BACKTEST ENGINE ====================

class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.data_fetcher = DataFetcher()
    
    def calculate_indicators(self, df):
        for period in self.config.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
        return df
    
    def create_higher_timeframes(self, df):
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
    
    def generate_trades(self, data_pool):
        """Generate trades using CURRENT learned parameters"""
        new_trades = []
        
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
                exit_reason = None
                
                if current_low <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                elif current_high >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'take_profit'
                elif (current_date - position['entry_date']).days >= 60:
                    exit_price = current_price
                    exit_reason = 'time_exit'
                
                if exit_price:
                    return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    
                    trade = {
                        'ticker': ticker,
                        'entry_date': str(position['entry_date']),
                        'exit_date': str(current_date),
                        'return_pct': return_pct,
                        'compression': position['compression'],
                        'volume_ratio': position['volume_ratio'],
                        'exit_reason': exit_reason,
                    }
                    
                    new_trades.append(trade)
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
        
        return new_trades
    
    def calculate_metrics(self, trades):
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'total_return': 0}
        
        df = pd.DataFrame(trades)
        winners = df[df['return_pct'] > 0]
        losers = df[df['return_pct'] <= 0]
        
        return {
            'total_trades': len(df),
            'win_rate': len(winners) / len(df) * 100,
            'profit_factor': winners['return_pct'].sum() / abs(losers['return_pct'].sum()) if len(losers) > 0 else float('inf'),
            'total_return': df['return_pct'].sum(),
        }

# ==================== MAIN ====================

def main():
    try:
        logger.info("=" * 80)
        logger.info("TRUE INCREMENTAL LEARNING LOOP")
        logger.info(f"Previous trades: {len(config.all_trades)}")
        logger.info(f"Current params: {config.learned_params}")
        logger.info("=" * 80)
        
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
        
        logger.info("📦 Loading data...")
        for ticker in tickers:
            df = data_fetcher.fetch_data(ticker, config.data_start)
            if df is not None and len(df) > 300:
                data_pool[ticker] = df
            time.sleep(0.2)
        
        logger.info(f"✓ Loaded {len(data_pool)} stocks")
        
        # Initialize
        learner = IncrementalLearner(config)
        backtest = BacktestEngine(config)
        
        # ===== STEP 1: LEARN FROM PREVIOUS TRADES =====
        logger.info("\n📚 STEP 1: Learning from previous trades...")
        learner.learn_from_all_trades()
        
        # ===== STEP 2: GENERATE NEW TRADES WITH UPDATED PARAMS =====
        logger.info("\n🔄 STEP 2: Generating trades with updated parameters...")
        new_trades = backtest.generate_trades(data_pool)
        
        logger.info(f"✓ Generated {len(new_trades)} new trades")
        
        # Add to historical trades
        config.all_trades.extend(new_trades)
        config.new_trades_this_run = len(new_trades)
        
        # ===== STEP 3: LEARN AGAIN WITH NEW TRADES =====
        if new_trades:
            logger.info("\n📚 STEP 3: Learning from combined trades...")
            learner.learn_from_all_trades()
        
        # ===== STEP 4: SAVE STATE =====
        metrics = backtest.calculate_metrics(new_trades)
        
        logger.info("\n" + "=" * 80)
        logger.info("ITERATION COMPLETE")
        logger.info(f"New trades: {len(new_trades)}")
        logger.info(f"Total historical trades: {len(config.all_trades)}")
        logger.info(f"Win rate (new): {metrics['win_rate']:.1f}%")
        logger.info(f"Profit factor (new): {metrics['profit_factor']:.2f}")
        logger.info(f"Updated params: {config.learned_params}")
        logger.info("=" * 80)
        
        config.save_full_state(metrics)
        
        # Save CSV
        pd.DataFrame([config.learned_params]).to_csv('nifty500_broom_breakout_results.csv', index=False)
        
        # Commit to GitHub
        try:
            subprocess.run(['git', 'config', '--local', 'user.email', 'action@github.com'], check=False)
            subprocess.run(['git', 'config', '--local', 'user.name', 'GitHub Action'], check=False)
            subprocess.run(['git', 'add', 'state/'], check=False)
            subprocess.run(['git', 'add', 'nifty500_broom_breakout_results.csv'], check=False)
            subprocess.run(['git', 'commit', '-m', f'Learning update: {len(config.all_trades)} trades, v{config.version}'], check=False)
            subprocess.run(['git', 'push'], check=False)
            logger.info("✅ Committed to GitHub")
        except Exception as e:
            logger.warning(f"Commit failed: {e}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
