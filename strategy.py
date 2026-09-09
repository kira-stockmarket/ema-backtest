"""
TRUE Online Learning Broom Breakout Strategy
Actually learns and adapts from each trade outcome in real-time
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
from collections import deque

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    """Configuration for true online learning"""
    
    def __init__(self):
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # Backtest Period
        self.backtest_start = '2018-01-01'
        self.data_start = '2000-01-01'
        
        # Universe
        self.universe_size = int(os.getenv('UNIVERSE_SIZE', '50'))
        
        # INITIAL Parameters (will be modified by learning)
        self.ema_periods = [20, 50, 100, 200]
        self.broom_compression_threshold = 0.08
        self.base_duration_min = 63
        self.base_duration_max = 147
        self.box_consolidation_height = 0.15
        self.volume_threshold_multiplier = 1.5
        self.stop_loss_buffer = 0.015
        self.measured_move_multiplier = 2.0
        
        # Online Learning Parameters
        self.learning_window = 20  # Look at last 20 trades
        self.adaptation_rate = 0.05  # How fast to adapt (5% per adjustment)
        self.min_trades_to_learn = 10  # Need at least 10 trades before learning
        
        # Directories
        self.data_dir = Path('data_cache')
        self.results_dir = Path('results')
        self.logs_dir = Path('logs')
        self.learning_dir = Path('learning')
        
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir, self.learning_dir]:
            dir_path.mkdir(exist_ok=True)

# ==================== LOGGING ====================

def setup_logging(config: Config):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logs_dir / 'online_learning.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

config = Config()
logger = setup_logging(config)

# ==================== ONLINE LEARNER ====================

class OnlineLearner:
    """True online learning - adapts after each trade"""
    
    def __init__(self, config: Config):
        self.config = config
        self.recent_trades = deque(maxlen=50)  # Last 50 trades
        self.parameter_history = []
        self.performance_history = []
        
        # Track what's working
        self.winning_features = {}
        self.losing_features = {}
        
    def add_trade(self, trade: Dict):
        """Learn from a completed trade"""
        self.recent_trades.append(trade)
        
        # Extract features from trade
        features = {
            'compression': trade.get('compression', 0),
            'volume_ratio': trade.get('volume_ratio', 0),
            'days_held': trade.get('days_held', 0),
            'return_pct': trade.get('return_pct', 0),
        }
        
        # Update winning/losing patterns
        if trade['return_pct'] > 0:
            for key, value in features.items():
                if key not in self.winning_features:
                    self.winning_features[key] = []
                self.winning_features[key].append(value)
        else:
            for key, value in features.items():
                if key not in self.losing_features:
                    self.losing_features[key] = []
                self.losing_features[key].append(value)
        
        # Adapt parameters if enough trades
        if len(self.recent_trades) >= self.config.min_trades_to_learn:
            self.adapt_parameters()
    
    def adapt_parameters(self):
        """Adapt parameters based on recent performance"""
        recent = list(self.recent_trades)[-self.config.learning_window:]
        
        if len(recent) < self.config.min_trades_to_learn:
            return
        
        recent_df = pd.DataFrame(recent)
        
        winners = recent_df[recent_df['return_pct'] > 0]
        losers = recent_df[recent_df['return_pct'] <= 0]
        
        if len(winners) == 0 or len(losers) == 0:
            return
        
        win_rate = len(winners) / len(recent_df) * 100
        
        logger.info(f"\n🧠 Learning from {len(recent)} recent trades")
        logger.info(f"   Recent win rate: {win_rate:.1f}%")
        
        changes = {}
        
        # 1. Adapt compression threshold based on winning trades
        if 'compression' in winners.columns and len(winners) > 3:
            winner_compression = winners['compression'].mean()
            loser_compression = losers['compression'].mean()
            
            # Move threshold towards winning compression
            optimal = winner_compression
            current = self.config.broom_compression_threshold
            
            # Adapt towards optimal
            new_value = current + (optimal - current) * self.config.adaptation_rate
            
            # Clamp to reasonable range
            new_value = max(0.04, min(0.15, new_value))
            
            changes['broom_compression_threshold'] = new_value
            
            logger.info(f"   Compression: {current:.3f} → {new_value:.3f}")
        
        # 2. Adapt volume threshold
        if 'volume_ratio' in winners.columns and len(winners) > 3:
            winner_volume = winners['volume_ratio'].mean()
            loser_volume = losers['volume_ratio'].mean()
            
            # If winners have higher volume, increase threshold
            if winner_volume > loser_volume:
                new_value = self.config.volume_threshold_multiplier * (1 + self.config.adaptation_rate)
            else:
                new_value = self.config.volume_threshold_multiplier * (1 - self.config.adaptation_rate)
            
            new_value = max(1.1, min(2.5, new_value))
            
            changes['volume_threshold_multiplier'] = new_value
            
            logger.info(f"   Volume: {self.config.volume_threshold_multiplier:.2f} → {new_value:.2f}")
        
        # 3. Adapt stop loss based on win rate
        if win_rate < 40:
            # Too many losers - tighten stop
            new_stop = self.config.stop_loss_buffer * (1 - self.config.adaptation_rate)
            changes['stop_loss_buffer'] = max(0.008, new_stop)
            logger.info(f"   Tightening stop: {self.config.stop_loss_buffer:.3f} → {changes['stop_loss_buffer']:.3f}")
        elif win_rate > 55:
            # Good win rate - can loosen stop
            new_stop = self.config.stop_loss_buffer * (1 + self.config.adaptation_rate)
            changes['stop_loss_buffer'] = min(0.03, new_stop)
            logger.info(f"   Loosening stop: {self.config.stop_loss_buffer:.3f} → {changes['stop_loss_buffer']:.3f}")
        
        # 4. Adapt take profit based on average winner
        avg_winner_return = winners['return_pct'].mean()
        
        if avg_winner_return < 10:
            # Winners too small - increase target
            new_target = self.config.measured_move_multiplier * (1 + self.config.adaptation_rate)
            changes['measured_move_multiplier'] = min(3.0, new_target)
            logger.info(f"   Increasing target: {self.config.measured_move_multiplier:.1f} → {changes['measured_move_multiplier']:.1f}")
        elif avg_winner_return > 30:
            # Winners good - can reduce target to take profits earlier
            new_target = self.config.measured_move_multiplier * (1 - self.config.adaptation_rate)
            changes['measured_move_multiplier'] = max(1.2, new_target)
            logger.info(f"   Reducing target: {self.config.measured_move_multiplier:.1f} → {changes['measured_move_multiplier']:.1f}")
        
        # Apply changes
        for param, value in changes.items():
            setattr(self.config, param, value)
        
        # Track history
        self.parameter_history.append({
            'timestamp': datetime.now().isoformat(),
            'win_rate': win_rate,
            'changes': changes
        })
        
        # Save learning state
        self.save_learning_state()
    
    def save_learning_state(self):
        """Save learning state for future runs"""
        state = {
            'parameter_history': self.parameter_history[-100:],  # Last 100 adaptations
            'current_params': {
                'broom_compression_threshold': self.config.broom_compression_threshold,
                'volume_threshold_multiplier': self.config.volume_threshold_multiplier,
                'stop_loss_buffer': self.config.stop_loss_buffer,
                'measured_move_multiplier': self.config.measured_move_multiplier,
            },
            'winning_features': self.winning_features,
            'losing_features': self.losing_features,
        }
        
        with open(self.config.learning_dir / 'learning_state.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def get_current_params(self) -> Dict:
        """Get current learned parameters"""
        return {
            'broom_compression_threshold': self.config.broom_compression_threshold,
            'volume_threshold_multiplier': self.config.volume_threshold_multiplier,
            'stop_loss_buffer': self.config.stop_loss_buffer,
            'measured_move_multiplier': self.config.measured_move_multiplier,
        }

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

# ==================== ONLINE LEARNING BACKTEST ====================

class OnlineLearningBacktest:
    """Backtest that learns online as trades complete"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.learner = OnlineLearner(config)
        
        logger.info("=" * 80)
        logger.info("ONLINE LEARNING BROOM BREAKOUT")
        logger.info("Adapts parameters after each trade")
        logger.info("=" * 80)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators"""
        for period in self.config.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df
    
    def create_higher_timeframes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create higher timeframes"""
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
    
    def check_setup(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Dict]:
        """Check setup and return features for learning"""
        if idx < 250:
            return False, {}
        
        try:
            current_price = df.iloc[idx]['Close']
            
            # Trend filter
            weekly_ema = df.iloc[idx]['Weekly_200_EMA']
            if pd.isna(weekly_ema) or current_price <= weekly_ema:
                return False, {}
            
            monthly_ema = df.iloc[idx]['Monthly_200_EMA']
            if not pd.isna(monthly_ema) and current_price <= monthly_ema:
                return False, {}
            
            # EMA compression (using CURRENT learned threshold)
            emas = [df.iloc[idx][f'EMA_{p}'] for p in self.config.ema_periods]
            if any(pd.isna(e) for e in emas):
                return False, {}
            
            compression = (max(emas) - min(emas)) / current_price
            
            if compression >= self.config.broom_compression_threshold:
                return False, {'compression': compression}
            
            # Base duration
            lookback = df.iloc[max(0, idx-200):idx]
            if len(lookback) < 200:
                return False, {'compression': compression}
            
            peak_pos = df.index.get_loc(lookback['High'].idxmax())
            days_since_peak = idx - peak_pos
            
            if days_since_peak < self.config.base_duration_min or \
               days_since_peak > self.config.base_duration_max:
                return False, {'compression': compression}
            
            # Box consolidation
            recent_20 = df.iloc[idx-19:idx+1]
            box_height = (recent_20['High'].max() - recent_20['Low'].min()) / current_price
            
            if box_height >= self.config.box_consolidation_height:
                return False, {'compression': compression}
            
            # Volume (using CURRENT learned threshold)
            volume = df.iloc[idx]['Volume']
            volume_ma = df.iloc[idx]['Volume_MA_50']
            
            if pd.isna(volume_ma):
                return False, {'compression': compression}
            
            volume_ratio = volume / volume_ma if volume_ma > 0 else 0
            
            if volume_ratio < self.config.volume_threshold_multiplier:
                return False, {
                    'compression': compression,
                    'volume_ratio': volume_ratio
                }
            
            # All conditions met
            return True, {
                'compression': compression,
                'volume_ratio': volume_ratio,
            }
            
        except:
            return False, {}
    
    def run_online_learning_backtest(self, data_pool: Dict):
        """Run backtest with online learning"""
        
        all_trades = []
        parameter_evolution = []
        
        # Prepare data
        prepared_data = {}
        for ticker, df in data_pool.items():
            df = self.calculate_indicators(df.copy())
            df = self.create_higher_timeframes(df)
            prepared_data[ticker] = df
        
        # Get all dates
        all_dates = []
        for df in prepared_data.values():
            dates = df[df.index >= self.config.backtest_start].index
            all_dates.extend(dates)
        
        all_dates = sorted(set(all_dates))
        
        open_positions = {}
        
        logger.info(f"Starting online learning backtest")
        logger.info(f"Dates: {all_dates[0].date()} to {all_dates[-1].date()}")
        logger.info(f"Initial params: {self.learner.get_current_params()}")
        
        trade_count = 0
        
        for current_date in all_dates:
            # 1. Check exits
            for ticker in list(open_positions.keys()):
                position = open_positions[ticker]
                df = prepared_data[ticker]
                
                if current_date not in df.index:
                    continue
                
                idx = df.index.get_loc(current_date)
                current_price = df.iloc[idx]['Close']
                current_high = df.iloc[idx]['High']
                current_low = df.iloc[idx]['Low']
                
                # Update trailing stop
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
                    days_held = (current_date - position['entry_date']).days
                    
                    trade = {
                        'ticker': ticker,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'return_pct': return_pct,
                        'days_held': days_held,
                        'exit_reason': exit_reason,
                        'compression': position['compression'],
                        'volume_ratio': position['volume_ratio'],
                    }
                    
                    all_trades.append(trade)
                    
                    # LEARN FROM THIS TRADE
                    self.learner.add_trade(trade)
                    
                    trade_count += 1
                    
                    if trade_count % 10 == 0:
                        logger.info(f"\n📊 After {trade_count} trades:")
                        logger.info(f"   Current params: {self.learner.get_current_params()}")
                        logger.info(f"   Recent win rate: {self.calculate_recent_win_rate(all_trades, 20):.1f}%")
                    
                    del open_positions[ticker]
            
            # 2. Check for entries
            for ticker, df in prepared_data.items():
                if ticker in open_positions:
                    continue
                
                if current_date not in df.index:
                    continue
                
                idx = df.index.get_loc(current_date)
                
                setup_ok, features = self.check_setup(df, idx)
                
                if setup_ok:
                    entry_price = df.iloc[idx]['Close']
                    stop_loss = df.iloc[idx]['EMA_200'] * (1 - self.config.stop_loss_buffer)
                    
                    # Calculate take profit
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
                        'compression': features.get('compression', 0),
                        'volume_ratio': features.get('volume_ratio', 0),
                    }
            
            # Track parameter evolution
            parameter_evolution.append({
                'date': current_date,
                'params': self.learner.get_current_params(),
                'trades_completed': trade_count
            })
        
        # Final results
        final_metrics = self.calculate_metrics(all_trades)
        
        logger.info("\n" + "=" * 80)
        logger.info("ONLINE LEARNING COMPLETE")
        logger.info(f"Total trades: {final_metrics['total_trades']}")
        logger.info(f"Win rate: {final_metrics['win_rate']:.1f}%")
        logger.info(f"Total return: {final_metrics['total_return']:.1f}%")
        logger.info(f"Profit factor: {final_metrics['profit_factor']:.2f}")
        logger.info(f"\nFinal learned params: {self.learner.get_current_params()}")
        logger.info("=" * 80)
        
        # Save results
        results = {
            'trades': all_trades,
            'metrics': final_metrics,
            'parameter_evolution': parameter_evolution,
            'final_params': self.learner.get_current_params(),
        }
        
        with open(self.config.learning_dir / 'online_learning_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create CSV
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv('nifty500_broom_breakout_results.csv', index=False)
        
        return results
    
    def calculate_recent_win_rate(self, trades: List[Dict], window: int) -> float:
        """Calculate win rate for recent trades"""
        if not trades:
            return 0
        
        recent = trades[-window:]
        winners = [t for t in recent if t['return_pct'] > 0]
        return len(winners) / len(recent) * 100
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate metrics"""
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_return': 0, 'profit_factor': 0}
        
        df = pd.DataFrame(trades)
        total_trades = len(df)
        winners = df[df['return_pct'] > 0]
        losers = df[df['return_pct'] <= 0]
        
        return {
            'total_trades': total_trades,
            'win_rate': len(winners) / total_trades * 100,
            'total_return': df['return_pct'].sum(),
            'profit_factor': winners['return_pct'].sum() / abs(losers['return_pct'].sum()) if len(losers) > 0 else float('inf')
        }

# ==================== MAIN ====================

def main():
    """Main execution"""
    try:
        logger.info("=" * 80)
        logger.info("ONLINE LEARNING BROOM BREAKOUT")
        logger.info("True self-improving system")
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
            time.sleep(0.3)
        
        logger.info(f"✓ Loaded {len(data_pool)} stocks")
        
        # Run online learning backtest
        backtest = OnlineLearningBacktest(config)
        results = backtest.run_online_learning_backtest(data_pool)
        
        logger.info("\n✅ ONLINE LEARNING COMPLETE")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        pd.DataFrame(columns=['error']).to_csv('nifty500_broom_breakout_results.csv', index=False)
        sys.exit(1)

if __name__ == "__main__":
    main()
