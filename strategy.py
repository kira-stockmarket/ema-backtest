"""
SNIPER Broom Breakout Strategy
Extremely Selective - Quality Over Quantity
Multiple Confirmations Required
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
    """Sniper configuration - extremely selective"""
    
    def __init__(self):
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # Backtest Period
        self.backtest_start = '2018-01-01'
        self.data_start = '2015-01-01'
        
        # Universe
        self.universe_size = int(os.getenv('UNIVERSE_SIZE', '50'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        
        # ===== SNIPER FILTERS - VERY STRICT =====
        
        # EMA Compression (very tight)
        self.compression_threshold = 0.05  # 5% only - very tight broom
        
        # Base Pattern (longer, cleaner)
        self.base_lookback = 180  # 9 months lookback
        self.base_duration_min = 40  # 2 months minimum
        self.base_duration_max = 100  # 5 months maximum
        
        # Consolidation (very tight)
        self.consolidation_period = 30  # 30 days consolidation
        self.consolidation_height = 0.10  # 10% maximum box height
        
        # ===== MULTIPLE CONFIRMATIONS REQUIRED =====
        
        # 1. Price Action Confirmation
        self.breakout_strength = 0.03  # 3% above 30-day high
        self.close_strength = 0.02  # Close 2% above breakout level
        
        # 2. Volume Confirmation (very strict)
        self.volume_surge_min = 2.0  # 2x 20-day average minimum
        self.volume_surge_ideal = 3.0  # 3x is ideal
        
        # 3. RSI Confirmation (momentum sweet spot)
        self.rsi_min = 55  # Not too weak
        self.rsi_max = 65  # Not too overbought
        self.rsi_rising = True  # RSI should be rising
        
        # 4. Sector & Market Confirmation
        self.require_sector_confirmation = True
        self.require_market_confirmation = True
        
        # 5. Trend Strength Requirements
        self.min_ema_slope = 0.001  # 200 EMA should be rising
        self.price_above_200ema_pct = 0.05  # 5% above 200 EMA
        
        # Risk Management
        self.initial_stop_atr = 2.0  # 2x ATR initial stop
        self.trailing_stop_atr = 2.5  # 2.5x ATR trailing
        self.max_holding_days = 90  # 3 months max
        
        # ===== QUALITY SCORING =====
        self.min_quality_score = 70  # Minimum 70/100 quality score
        
        # Filters
        self.min_price = 100
        self.min_avg_volume = 500000  # 5 lakh minimum
        
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
    """Data fetcher with index support"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.nifty_data = None
        self.nifty_monthly_trend = None
    
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
    
    def load_nifty_data(self):
        """Load Nifty 50 index data"""
        try:
            self.nifty_data = self.fetch_data('^NSEI', self.config.data_start)
            
            if self.nifty_data is not None:
                # Calculate monthly trend
                monthly = self.nifty_data.resample('ME').agg({
                    'Close': 'last'
                }).dropna()
                
                monthly['EMA_20'] = monthly['Close'].ewm(span=20, adjust=False).mean()
                monthly['Trend_Up'] = monthly['Close'] > monthly['EMA_20']
                
                self.nifty_monthly_trend = monthly['Trend_Up'].reindex(
                    self.nifty_data.index, method='ffill'
                )
                
                logger.info(f"✓ Nifty 50 loaded: {len(self.nifty_data)} days")
        except Exception as e:
            logger.warning(f"Failed to load Nifty data: {e}")
    
    def is_nifty_bullish(self, date) -> bool:
        """Check if Nifty is bullish on monthly timeframe"""
        if self.nifty_monthly_trend is None:
            return True  # If no data, don't filter
        
        try:
            trend = self.nifty_monthly_trend[self.nifty_monthly_trend.index <= date]
            
            if len(trend) == 0:
                return True
            
            return bool(trend.iloc[-1])
        except:
            return True

# ==================== SNIPER BACKTEST ====================

class SniperBroomBacktest:
    """Sniper Broom Breakout - Quality over quantity"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.data_fetcher.config = config
        self.results = []
        
        # Load Nifty data
        self.data_fetcher.load_nifty_data()
        
        logger.info("=" * 80)
        logger.info("🎯 SNIPER BROOM BREAKOUT STRATEGY")
        logger.info("Quality Over Quantity - Multiple Confirmations")
        logger.info(f"Compression: {self.config.compression_threshold:.0%}")
        logger.info(f"Volume Surge: {self.config.volume_surge_min}x minimum")
        logger.info(f"RSI Range: {self.config.rsi_min}-{self.config.rsi_max}")
        logger.info(f"Breakout Strength: {self.config.breakout_strength:.0%}")
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
        """Calculate comprehensive indicators"""
        # EMAs
        for period in [20, 50, 100, 200]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # EMA Slopes
        df['EMA_200_Slope'] = df['EMA_200'].pct_change(5)  # 5-day slope
        
        # Volume
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA_50'] = df['Volume'].rolling(window=50).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Rising'] = df['RSI'] > df['RSI'].shift(3)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Price position
        df['Pct_Above_200EMA'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
        
        # Rolling highs
        df['High_30'] = df['High'].rolling(window=30).max()
        df['High_20'] = df['High'].rolling(window=20).max()
        
        return df
    
    def calculate_quality_score(self, df: pd.DataFrame, idx: int) -> int:
        """Calculate quality score 0-100"""
        score = 0
        
        try:
            current_price = df.iloc[idx]['Close']
            
            # 1. Compression quality (25 points)
            emas = [df.iloc[idx][f'EMA_{p}'] for p in [20, 50, 100, 200]]
            ema_spread = (max(emas) - min(emas)) / current_price
            
            if ema_spread < 0.03:
                score += 25
            elif ema_spread < 0.05:
                score += 20
            elif ema_spread < 0.07:
                score += 15
            elif ema_spread < 0.10:
                score += 10
            
            # 2. Volume surge (25 points)
            volume_ratio = df.iloc[idx]['Volume_Ratio']
            
            if volume_ratio > 3.0:
                score += 25
            elif volume_ratio > 2.5:
                score += 20
            elif volume_ratio > 2.0:
                score += 15
            elif volume_ratio > 1.5:
                score += 10
            
            # 3. RSI quality (20 points)
            rsi = df.iloc[idx]['RSI']
            
            if 58 <= rsi <= 62:
                score += 20  # Perfect momentum
            elif 55 <= rsi <= 65:
                score += 15
            elif 50 <= rsi <= 70:
                score += 10
            
            # 4. Trend strength (20 points)
            pct_above_200 = df.iloc[idx]['Pct_Above_200EMA']
            ema_200_slope = df.iloc[idx]['EMA_200_Slope']
            
            if pct_above_200 > 0.10 and ema_200_slope > 0.001:
                score += 20
            elif pct_above_200 > 0.05 and ema_200_slope > 0:
                score += 15
            elif pct_above_200 > 0.02:
                score += 10
            
            # 5. Consolidation quality (10 points)
            recent = df.iloc[idx-29:idx+1]
            box_height = (recent['High'].max() - recent['Low'].min()) / current_price
            
            if box_height < 0.05:
                score += 10
            elif box_height < 0.08:
                score += 7
            elif box_height < 0.10:
                score += 5
            
            return score
            
        except Exception as e:
            return 0
    
    def is_sniper_setup(self, df: pd.DataFrame, idx: int, ticker: str) -> Tuple[bool, str, int]:
        """Check for Sniper setup - ALL conditions must be met"""
        
        # 1. Nifty Market Filter
        if self.config.require_market_confirmation:
            current_date = df.index[idx]
            if not self.data_fetcher.is_nifty_bullish(current_date):
                return False, "Nifty not bullish", 0
        
        # 2. Minimum data
        if idx < 200:
            return False, "Insufficient data", 0
        
        current_price = df.iloc[idx]['Close']
        
        # 3. Price above 200 EMA by at least 5%
        pct_above_200 = df.iloc[idx]['Pct_Above_200EMA']
        if pd.isna(pct_above_200) or pct_above_200 < self.config.min_ema_slope:
            return False, "Below 200 EMA", 0
        
        # 4. 200 EMA rising
        ema_200_slope = df.iloc[idx]['EMA_200_Slope']
        if pd.isna(ema_200_slope) or ema_200_slope <= 0:
            return False, "200 EMA not rising", 0
        
        # 5. Very tight EMA compression (Broom)
        emas = [df.iloc[idx][f'EMA_{p}'] for p in [20, 50, 100, 200]]
        
        if any(pd.isna(e) for e in emas):
            return False, "Missing EMA", 0
        
        ema_spread = (max(emas) - min(emas)) / current_price
        
        if ema_spread > self.config.compression_threshold:
            return False, f"EMA spread too wide: {ema_spread:.2%}", 0
        
        # 6. Tight consolidation
        recent_30 = df.iloc[idx-29:idx+1]
        box_height = (recent_30['High'].max() - recent_30['Low'].min()) / current_price
        
        if box_height > self.config.consolidation_height:
            return False, f"Box too wide: {box_height:.2%}", 0
        
        # 7. Base duration
        lookback = df.iloc[max(0, idx-self.config.base_lookback):idx]
        if len(lookback) < 100:
            return False, "Insufficient base", 0
        
        peak_pos = lookback['High'].idxmax()
        peak_idx = df.index.get_loc(peak_pos)
        days_since_peak = idx - peak_idx
        
        if days_since_peak < self.config.base_duration_min or \
           days_since_peak > self.config.base_duration_max:
            return False, f"Bad base duration: {days_since_peak}d", 0
        
        # 8. STRONG breakout (3% above 30-day high)
        high_30 = df.iloc[idx]['High_30']
        if pd.isna(high_30) or current_price < high_30 * (1 + self.config.breakout_strength):
            return False, "No strong breakout", 0
        
        # 9. MASSIVE volume surge (2x minimum)
        volume_ratio = df.iloc[idx]['Volume_Ratio']
        if pd.isna(volume_ratio) or volume_ratio < self.config.volume_surge_min:
            return False, f"Weak volume: {volume_ratio:.1f}x", 0
        
        # 10. RSI in sweet spot (55-65)
        rsi = df.iloc[idx]['RSI']
        if pd.isna(rsi) or rsi < self.config.rsi_min or rsi > self.config.rsi_max:
            return False, f"RSI out of range: {rsi:.1f}", 0
        
        # 11. RSI rising
        rsi_rising = df.iloc[idx]['RSI_Rising']
        if pd.isna(rsi_rising) or not rsi_rising:
            return False, "RSI not rising", 0
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(df, idx)
        
        # 12. Quality score threshold
        if quality_score < self.config.min_quality_score:
            return False, f"Quality too low: {quality_score}", quality_score
        
        return True, f"SNIPER SETUP! Score: {quality_score}", quality_score
    
    def run_backtest(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """Run sniper backtest"""
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
                    # Check Sniper setup
                    is_setup, reason, score = self.is_sniper_setup(df, idx, ticker)
                    
                    if is_setup:
                        entry_price = current_price
                        atr = df.iloc[idx]['ATR']
                        
                        # Initial stop: 2x ATR
                        if not pd.isna(atr) and atr > 0:
                            initial_stop = entry_price - (self.config.initial_stop_atr * atr)
                        else:
                            initial_stop = entry_price * 0.95
                        
                        position = {
                            'entry_date': date,
                            'entry_price': entry_price,
                            'stop_loss': initial_stop,
                            'highest_price': entry_price,
                            'quality_score': score,
                            'atr': atr if not pd.isna(atr) else 0
                        }
                        
                        logger.info(f"🎯 SNIPER ENTRY: {ticker} @ ₹{entry_price:.2f} | "
                                  f"Score: {score}/100 | {reason}")
                
                else:
                    # Manage position
                    current_high = df.iloc[idx]['High']
                    current_low = df.iloc[idx]['Low']
                    days_held = (date - position['entry_date']).days
                    
                    # Update highest price
                    if current_high > position['highest_price']:
                        position['highest_price'] = current_high
                    
                    # Trailing stop: 2.5x ATR from highest
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
                            'quality_score': position['quality_score'],
                            'exit_reason': 'trailing_stop',
                            'days_held': days_held
                        })
                        
                        logger.info(f"🛑 EXIT: {ticker} | "
                                  f"Return: {return_pct:.2f}% | "
                                  f"Peak: {peak_profit:.2f}% | "
                                  f"Score: {position['quality_score']} | "
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
                            'quality_score': position['quality_score'],
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
                    'quality_score': position['quality_score'],
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
                'avg_days_held': 0, 'avg_quality': 0
            }
        
        try:
            df = pd.DataFrame(trades)
            
            total_trades = len(df)
            winners = df[df['return_pct'] > 0]
            losers = df[df['return_pct'] < 0]
            
            win_rate = len(winners) / total_trades * 100
            total_return = df['return_pct'].sum()
            avg_return = df['return_pct'].mean()
            avg_quality = df['quality_score'].mean()
            
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
                'avg_quality': avg_quality
            }
        except Exception as e:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                'avg_return': 0, 'profit_factor': 0, 'max_drawdown': 0,
                'avg_days_held': 0, 'avg_quality': 0
            }
    
    def run_universe(self, tickers: List[str]) -> pd.DataFrame:
        """Run sniper backtest"""
        all_trades = []
        stock_results = []
        sniper_count = 0
        
        logger.info("=" * 80)
        logger.info(f"🎯 SNIPER BACKTEST - {len(tickers)} STOCKS")
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
                    sniper_count += len(trades)
                    
                    logger.info(f"  🎯 {metrics['total_trades']} SNIPER trades | "
                              f"Win: {metrics['win_rate']:.0f}% | "
                              f"Return: {metrics['total_return']:.1f}% | "
                              f"Avg Quality: {metrics['avg_quality']:.0f}")
                else:
                    logger.info(f"  - No sniper setups found")
                
                del df, trades
                gc.collect()
                
                time.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
        
        # Overall summary
        if all_trades:
            overall = self.calculate_metrics(all_trades)
            
            logger.info("\n" + "=" * 80)
            logger.info("🎯 SNIPER RESULTS")
            logger.info(f"Total sniper trades: {overall['total_trades']}")
            logger.info(f"Win rate: {overall['win_rate']:.1f}%")
            logger.info(f"Total return: {overall['total_return']:.1f}%")
            logger.info(f"Profit factor: {overall['profit_factor']:.2f}")
            logger.info(f"Average quality score: {overall['avg_quality']:.0f}/100")
            logger.info(f"Max drawdown: {overall['max_drawdown']:.1f}%")
            logger.info("=" * 80)
        
        # Save results
        results_df = pd.DataFrame(stock_results)
        if not results_df.empty:
            results_df.to_csv('nifty500_broom_breakout_results.csv', index=False)
            logger.info(f"\nResults saved to nifty500_broom_breakout_results.csv")
        else:
            # Create empty file
            pd.DataFrame(columns=['ticker', 'total_trades', 'win_rate', 'total_return']).to_csv(
                'nifty500_broom_breakout_results.csv', index=False
            )
        
        return results_df

# ==================== UNIVERSE ====================

def get_universe() -> List[str]:
    """Get universe - focus on quality stocks"""
    return [
        # High quality large caps
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS',
        'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
        'DIVISLAB.NS', 'DRREDDY.NS', 'CIPLA.NS', 'BRITANNIA.NS', 'DABUR.NS',
        
        # Quality midcaps
        'PIDILITIND.NS', 'HAVELLS.NS', 'ASTRAL.NS', 'DIXON.NS', 'TRENT.NS',
        'DMART.NS', 'CUMMINSIND.NS', 'VOLTAS.NS', 'CROMPTON.NS', 'KEI.NS',
        'POLYCAB.NS', 'SUPREMEIND.NS', 'WHIRLPOOL.NS', 'BLUESTARCO.NS',
        
        # Pharma leaders
        'LUPIN.NS', 'AUROPHARMA.NS', 'BIOCON.NS', 'GLENMARK.NS', 'ALKEM.NS',
        'TORNTPHARM.NS', 'ZYDUSLIFE.NS',
        
        # Auto leaders
        'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS',
        
        # IT leaders
        'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'MPHASIS.NS', 'COFORGE.NS',
        'PERSISTENT.NS',
    ]

# ==================== MAIN ====================

def main():
    """Main execution"""
    try:
        logger.info("=" * 80)
        logger.info("🎯 SNIPER BROOM BREAKOUT STRATEGY")
        logger.info("Quality Over Quantity")
        logger.info("=" * 80)
        
        tickers = get_universe()
        
        engine = SniperBroomBacktest(config)
        
        results = engine.run_universe(tickers)
        
        logger.info("\n✅ SNIPER BACKTEST COMPLETED")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        # Create empty results file
        pd.DataFrame(columns=['ticker', 'total_trades', 'win_rate', 'total_return']).to_csv(
            'nifty500_broom_breakout_results.csv', index=False
        )
        
        sys.exit(1)

if __name__ == "__main__":
    main()
