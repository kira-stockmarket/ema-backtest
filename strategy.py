"""
Institutional Moving Average "Broom" Breakout Strategy Backtesting Engine
For Nifty 500 Universe - CORRECTED TREND FILTERS
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

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    """Central configuration management"""
    
    def __init__(self):
        # Environment detection
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # API Keys
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        
        # Backtest Period
        self.backtest_start = os.getenv('BACKTEST_START', '2018-01-01')
        self.data_start = os.getenv('DATA_START', '2000-01-01')
        
        # Universe Configuration
        self.universe_size = int(os.getenv('UNIVERSE_SIZE', '20'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '5'))
        self.full_universe = os.getenv('FULL_UNIVERSE', 'false').lower() == 'true'
        
        # Strategy Parameters
        self.ema_periods = [20, 50, 100, 200]
        self.weekly_ema_period = 200
        self.monthly_ema_period = 200
        
        # Higher Timeframe Trend Filters - NEW
        self.weekly_ema_short = 20  # Weekly 20 EMA for short-term trend
        self.weekly_ema_medium = 50  # Weekly 50 EMA for medium-term trend
        self.monthly_ema_short = 10  # Monthly 10 EMA for short-term trend
        self.monthly_ema_medium = 20  # Monthly 20 EMA for medium-term trend
        
        # Trend Strength Requirements
        self.weekly_trend_strength = float(os.getenv('WEEKLY_TREND_STRENGTH', '0.05'))  # 5% above weekly EMA
        self.monthly_trend_strength = float(os.getenv('MONTHLY_TREND_STRENGTH', '0.03'))  # 3% above monthly EMA
        
        # Broom Compression
        self.broom_compression_threshold = float(os.getenv('BROOM_COMPRESSION_THRESHOLD', '0.12'))
        self.base_lookback_period = 200
        self.base_duration_min = int(os.getenv('BASE_DURATION_MIN', '40'))
        self.base_duration_max = int(os.getenv('BASE_DURATION_MAX', '180'))
        self.box_consolidation_height = float(os.getenv('BOX_CONSOLIDATION_HEIGHT', '0.20'))
        self.prior_trend_exhaustion_limit = float(os.getenv('PRIOR_TREND_EXHAUSTION_LIMIT', '0.80'))
        
        # Execution Parameters
        self.volume_poc_bins = 10
        self.poc_lookback = 20
        self.volume_threshold_multiplier = float(os.getenv('VOLUME_THRESHOLD_MULTIPLIER', '1.2'))
        self.volume_ma_period = 50
        self.stop_loss_buffer = float(os.getenv('STOP_LOSS_BUFFER', '0.02'))
        self.measured_move_multiplier = float(os.getenv('MEASURED_MOVE_MULTIPLIER', '1.5'))
        
        # Rate Limiting
        self.request_delay = float(os.getenv('REQUEST_DELAY', '2'))
        self.batch_delay = float(os.getenv('BATCH_DELAY', '10'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        
        # Cache settings
        self.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.cache_expiry_days = int(os.getenv('CACHE_EXPIRY_DAYS', '7'))
        
        # Debug settings
        self.debug_mode = os.getenv('DEBUG_MODE', 'true').lower() == 'true'
        self.trade_log_enabled = os.getenv('TRADE_LOG_ENABLED', 'true').lower() == 'true'
        
        # Directories
        self.data_dir = Path('data_cache')
        self.results_dir = Path('results')
        self.logs_dir = Path('logs')
        
        # Create directories
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.universe_size <= 0:
            raise ValueError("UNIVERSE_SIZE must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")

# ==================== LOGGING SETUP ====================

def setup_logging(config: Config):
    """Setup logging configuration"""
    log_file = config.logs_dir / 'backtest.log'
    
    logging.basicConfig(
        level=logging.DEBUG if config.debug_mode else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

config = Config()
logger = setup_logging(config)

# ==================== DATA FETCHING ====================

class DataFetcher:
    """Multi-source data fetcher with fallback mechanisms"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = self._create_session()
        self.rate_limit_hits = 0
        self.consecutive_failures = 0
        
    def _create_session(self):
        """Create session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=2,
            pool_maxsize=2
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json,text/plain,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        return session
    
    def fetch_from_yfinance(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance"""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker, session=self.session)
            
            df = stock.history(
                start=start_date,
                end=datetime.now().strftime('%Y-%m-%d'),
                auto_adjust=False,
                timeout=30,
                actions=False
            )
            
            if df is not None and not df.empty:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    df = df.dropna()
                    
                    if len(df) > 100:
                        return df
            
            return None
            
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                self.rate_limit_hits += 1
            return None
    
    def fetch_from_stooq(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from Stooq directly via HTTP"""
        try:
            if ticker.endswith('.NS'):
                stooq_ticker = ticker.replace('.NS', '.NSE')
            elif ticker.endswith('.BO'):
                stooq_ticker = ticker.replace('.BO', '.BSE')
            else:
                stooq_ticker = ticker + '.NSE'
            
            url = f"https://stooq.com/q/d/l/?s={stooq_ticker.lower()}&d1={start_date.replace('-', '')}&d2={datetime.now().strftime('%Y%m%d')}&i=d"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                
                if not df.empty and 'Close' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required_cols):
                        df = df[required_cols].copy()
                        df['Volume'] = df['Volume'].astype(float)
                        df = df.dropna()
                        
                        if len(df) > 100:
                            return df
            
            return None
            
        except Exception as e:
            return None
    
    def fetch_with_fallback(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data with multiple source fallback"""
        
        if self.rate_limit_hits >= 3:
            logger.warning(f"Multiple rate limit hits. Waiting 60 seconds...")
            time.sleep(60)
            self.rate_limit_hits = 0
        
        data_sources = [
            ('yfinance', self.fetch_from_yfinance),
            ('stooq', self.fetch_from_stooq),
        ]
        
        for source_name, fetch_func in data_sources:
            try:
                df = fetch_func(ticker, start_date)
                
                if df is not None and not df.empty and len(df) > 100:
                    self.consecutive_failures = 0
                    return df
                    
            except Exception as e:
                logger.debug(f"Error with {source_name} for {ticker}: {str(e)}")
        
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= 5:
            logger.warning(f"{self.consecutive_failures} consecutive failures. Waiting 120 seconds...")
            time.sleep(120)
            self.consecutive_failures = 0
        
        return None

# ==================== BACKTEST ENGINE ====================

class BroomBreakoutBacktest:
    """Main backtesting engine with corrected higher timeframe trend filters"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = []
        self.data_fetcher = DataFetcher(config)
        
        self.start_time = time.time()
        self.stocks_processed = 0
        self.stocks_failed = 0
        self.total_trades = 0
        
        # Trade log
        self.trade_log_file = None
        if config.trade_log_enabled:
            self.trade_log_file = open(config.logs_dir / 'trades.log', 'w')
        
        logger.info("=" * 80)
        logger.info("BROOM BREAKOUT BACKTEST ENGINE - CORRECTED TREND FILTERS")
        logger.info(f"Compression threshold: {self.config.broom_compression_threshold:.2%}")
        logger.info(f"Box height limit: {self.config.box_consolidation_height:.2%}")
        logger.info(f"Weekly trend strength: {self.config.weekly_trend_strength:.2%}")
        logger.info(f"Monthly trend strength: {self.config.monthly_trend_strength:.2%}")
        logger.info("=" * 80)
    
    def __del__(self):
        if self.trade_log_file:
            self.trade_log_file.close()
    
    def save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save data to cache"""
        if not self.config.cache_enabled:
            return
        
        try:
            cache_file = self.config.data_dir / f"{ticker.replace('.', '_')}.csv"
            df.to_csv(cache_file)
        except Exception as e:
            logger.warning(f"Failed to cache {ticker}: {e}")
    
    def load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        if not self.config.cache_enabled:
            return None
        
        try:
            cache_file = self.config.data_dir / f"{ticker.replace('.', '_')}.csv"
            
            if cache_file.exists():
                file_age = time.time() - cache_file.stat().st_mtime
                
                if file_age < self.config.cache_expiry_days * 24 * 3600:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    return df
        except Exception as e:
            logger.warning(f"Cache load failed for {ticker}: {e}")
        
        return None
    
    def download_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download data with caching"""
        df = self.load_from_cache(ticker)
        if df is not None:
            return df
        
        df = self.data_fetcher.fetch_with_fallback(ticker, self.config.data_start)
        
        if df is not None and not df.empty:
            self.save_to_cache(ticker, df)
        
        return df
    
    def calculate_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs"""
        for period in self.config.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df
    
    def create_higher_timeframes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weekly and monthly timeframes with multiple EMAs"""
        try:
            # Weekly resampling
            weekly_df = df.resample('W-FRI').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Weekly EMAs
            weekly_df['Weekly_EMA_20'] = weekly_df['Close'].ewm(span=20, adjust=False).mean()
            weekly_df['Weekly_EMA_50'] = weekly_df['Close'].ewm(span=50, adjust=False).mean()
            weekly_df['Weekly_EMA_200'] = weekly_df['Close'].ewm(span=200, adjust=False).mean()
            
            # Weekly trend direction
            weekly_df['Weekly_Trend_Up'] = (
                (weekly_df['Close'] > weekly_df['Weekly_EMA_20']) &
                (weekly_df['Weekly_EMA_20'] > weekly_df['Weekly_EMA_50']) &
                (weekly_df['Weekly_EMA_50'] > weekly_df['Weekly_EMA_200'])
            )
            
            # Weekly trend strength (percentage above 20 EMA)
            weekly_df['Weekly_Trend_Strength'] = (
                (weekly_df['Close'] - weekly_df['Weekly_EMA_20']) / weekly_df['Weekly_EMA_20']
            )
            
            # Monthly resampling
            monthly_df = df.resample('ME').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Monthly EMAs
            monthly_df['Monthly_EMA_10'] = monthly_df['Close'].ewm(span=10, adjust=False).mean()
            monthly_df['Monthly_EMA_20'] = monthly_df['Close'].ewm(span=20, adjust=False).mean()
            monthly_df['Monthly_EMA_200'] = monthly_df['Close'].ewm(span=200, adjust=False).mean()
            
            # Monthly trend direction
            monthly_df['Monthly_Trend_Up'] = (
                (monthly_df['Close'] > monthly_df['Monthly_EMA_10']) &
                (monthly_df['Monthly_EMA_10'] > monthly_df['Monthly_EMA_20']) &
                (monthly_df['Monthly_EMA_20'] > monthly_df['Monthly_EMA_200'])
            )
            
            # Monthly trend strength
            monthly_df['Monthly_Trend_Strength'] = (
                (monthly_df['Close'] - monthly_df['Monthly_EMA_10']) / monthly_df['Monthly_EMA_10']
            )
            
            # Map back to daily using forward fill
            df['Weekly_EMA_20'] = weekly_df['Weekly_EMA_20'].reindex(df.index, method='ffill')
            df['Weekly_EMA_50'] = weekly_df['Weekly_EMA_50'].reindex(df.index, method='ffill')
            df['Weekly_EMA_200'] = weekly_df['Weekly_EMA_200'].reindex(df.index, method='ffill')
            df['Weekly_Trend_Up'] = weekly_df['Weekly_Trend_Up'].reindex(df.index, method='ffill')
            df['Weekly_Trend_Strength'] = weekly_df['Weekly_Trend_Strength'].reindex(df.index, method='ffill')
            
            df['Monthly_EMA_10'] = monthly_df['Monthly_EMA_10'].reindex(df.index, method='ffill')
            df['Monthly_EMA_20'] = monthly_df['Monthly_EMA_20'].reindex(df.index, method='ffill')
            df['Monthly_EMA_200'] = monthly_df['Monthly_EMA_200'].reindex(df.index, method='ffill')
            df['Monthly_Trend_Up'] = monthly_df['Monthly_Trend_Up'].reindex(df.index, method='ffill')
            df['Monthly_Trend_Strength'] = monthly_df['Monthly_Trend_Strength'].reindex(df.index, method='ffill')
            
        except Exception as e:
            logger.debug(f"Error creating higher timeframes: {e}")
            # Set defaults
            df['Weekly_Trend_Up'] = False
            df['Weekly_Trend_Strength'] = 0
            df['Monthly_Trend_Up'] = False
            df['Monthly_Trend_Strength'] = 0
        
        return df
    
    def check_higher_timeframe_trend(self, df: pd.DataFrame, position: int) -> Tuple[bool, str]:
        """Check if higher timeframe trend is bullish - CRITICAL FILTER"""
        try:
            # Check Weekly trend
            weekly_trend_up = df.iloc[position]['Weekly_Trend_Up']
            weekly_strength = df.iloc[position]['Weekly_Trend_Strength']
            
            if pd.isna(weekly_trend_up) or not weekly_trend_up:
                return False, "Weekly trend not bullish"
            
            # Check if weekly trend is strong enough
            if weekly_strength < self.config.weekly_trend_strength:
                return False, f"Weekly trend too weak: {weekly_strength:.2%}"
            
            # Check Monthly trend
            monthly_trend_up = df.iloc[position]['Monthly_Trend_Up']
            monthly_strength = df.iloc[position]['Monthly_Trend_Strength']
            
            if pd.isna(monthly_trend_up):
                # If monthly data not available, only use weekly
                return True, "Weekly trend bullish"
            
            if not monthly_trend_up:
                return False, "Monthly trend not bullish"
            
            # Check if monthly trend is strong enough
            if monthly_strength < self.config.monthly_trend_strength:
                return False, f"Monthly trend too weak: {monthly_strength:.2%}"
            
            return True, "Weekly and Monthly trend bullish"
            
        except Exception as e:
            logger.debug(f"Error in higher timeframe check: {e}")
            return False, f"Error: {str(e)}"
    
    def check_broom_setup(self, df: pd.DataFrame, position: int) -> bool:
        """Check broom setup conditions with higher timeframe filter"""
        if position < self.config.base_lookback_period:
            return False
        
        try:
            current_price = df.iloc[position]['Close']
            if pd.isna(current_price) or current_price <= 0:
                return False
            
            # CRITICAL: Check higher timeframe trend first
            trend_ok, trend_reason = self.check_higher_timeframe_trend(df, position)
            if not trend_ok:
                logger.debug(f"Failed trend filter: {trend_reason}")
                return False
            
            # 1. EMA Broom Compression
            ema_values = []
            for period in self.config.ema_periods:
                ema_val = df.iloc[position][f'EMA_{period}']
                if pd.isna(ema_val):
                    return False
                ema_values.append(ema_val)
            
            ema_high = max(ema_values)
            ema_low = min(ema_values)
            ema_spread = (ema_high - ema_low) / current_price
            
            if ema_spread >= self.config.broom_compression_threshold:
                logger.debug(f"Failed compression: {ema_spread:.2%}")
                return False
            
            # 2. Base Duration Check
            lookback_start = max(0, position - self.config.base_lookback_period)
            lookback_data = df.iloc[lookback_start:position]
            
            if len(lookback_data) < self.config.base_lookback_period:
                return False
            
            peak_position = lookback_data['High'].idxmax()
            peak_pos = df.index.get_loc(peak_position)
            days_since_peak = position - peak_pos
            
            if days_since_peak < self.config.base_duration_min or \
               days_since_peak > self.config.base_duration_max:
                logger.debug(f"Failed base duration: {days_since_peak} days")
                return False
            
            # 3. Flat Box Consolidation
            recent_20 = df.iloc[position-19:position+1]
            box_height = (recent_20['High'].max() - recent_20['Low'].min()) / current_price
            
            if box_height >= self.config.box_consolidation_height:
                logger.debug(f"Failed box height: {box_height:.2%}")
                return False
            
            # 4. Prior Trend Exhaustion
            pre_peak_start = max(0, peak_pos - self.config.base_lookback_period)
            pre_peak_data = df.iloc[pre_peak_start:peak_pos+1]
            
            if len(pre_peak_data) > 0:
                lowest_low = pre_peak_data['Low'].min()
                peak_price = df.iloc[peak_pos]['High']
                
                if lowest_low > 0:
                    run_up = (peak_price - lowest_low) / lowest_low
                    
                    if run_up > self.config.prior_trend_exhaustion_limit:
                        logger.debug(f"Failed trend exhaustion: {run_up:.2%}")
                        return False
            
            logger.debug(f"✓ Broom setup found with {trend_reason}")
            return True
            
        except Exception as e:
            logger.debug(f"Error in broom setup: {e}")
            return False
    
    def calculate_volume_profile_poc(self, df: pd.DataFrame, position: int) -> Optional[float]:
        """Calculate Volume Profile POC"""
        if position < self.config.poc_lookback:
            return None
        
        try:
            recent_data = df.iloc[position - self.config.poc_lookback + 1:position + 1]
            
            price_range = recent_data['High'].max() - recent_data['Low'].min()
            if price_range == 0:
                return None
            
            bins = np.linspace(
                recent_data['Low'].min(),
                recent_data['High'].max(),
                self.config.volume_poc_bins + 1
            )
            
            volume_by_bin = np.zeros(self.config.volume_poc_bins)
            
            for i in range(len(recent_data)):
                row = recent_data.iloc[i]
                price_low = row['Low']
                price_high = row['High']
                volume = row['Volume']
                
                if price_high > price_low:
                    for j in range(self.config.volume_poc_bins):
                        bin_low = bins[j]
                        bin_high = bins[j + 1]
                        
                        overlap_low = max(price_low, bin_low)
                        overlap_high = min(price_high, bin_high)
                        
                        if overlap_high > overlap_low:
                            overlap_percentage = (overlap_high - overlap_low) / (price_high - price_low)
                            volume_by_bin[j] += volume * overlap_percentage
            
            poc_bin_idx = np.argmax(volume_by_bin)
            poc_price = (bins[poc_bin_idx] + bins[poc_bin_idx + 1]) / 2
            
            return poc_price
            
        except Exception as e:
            return None
    
    def check_entry_signal(self, df: pd.DataFrame, position: int) -> bool:
        """Check entry trigger conditions"""
        try:
            current_price = df.iloc[position]['Close']
            
            # Get highest EMA
            ema_values = [df.iloc[position][f'EMA_{period}'] for period in self.config.ema_periods]
            highest_ema = max(ema_values)
            
            # Check breakout above highest EMA
            if current_price <= highest_ema:
                logger.debug(f"Failed EMA breakout: {current_price:.2f} <= {highest_ema:.2f}")
                return False
            
            # Calculate Volume Profile POC
            poc_price = self.calculate_volume_profile_poc(df, position)
            if poc_price is None:
                return False
            
            # Check breakout above POC
            poc_threshold = poc_price * 0.98
            if current_price <= poc_threshold:
                logger.debug(f"Failed POC breakout: {current_price:.2f} <= {poc_threshold:.2f}")
                return False
            
            # Volume confirmation
            if position < self.config.volume_ma_period:
                return False
            
            volume_ma = df.iloc[position - self.config.volume_ma_period:position]['Volume'].mean()
            current_volume = df.iloc[position]['Volume']
            
            if current_volume <= self.config.volume_threshold_multiplier * volume_ma:
                logger.debug(f"Failed volume: {current_volume:.0f} <= {self.config.volume_threshold_multiplier * volume_ma:.0f}")
                return False
            
            logger.debug("✓ Entry signal triggered")
            return True
            
        except Exception as e:
            logger.debug(f"Error in entry signal: {e}")
            return False
    
    def run_backtest(self, df: pd.DataFrame, ticker: str = "") -> List[Dict]:
        """Run backtest on single stock"""
        trades = []
        position = None
        
        try:
            df = self.calculate_emas(df)
            df = self.create_higher_timeframes(df)
            
            backtest_mask = df.index >= self.config.backtest_start
            backtest_df = df[backtest_mask].copy()
            
            if len(backtest_df) < 50:
                return trades
            
            setup_count = 0
            trend_filter_count = 0
            entry_count = 0
            
            for date_idx in backtest_df.index:
                position_idx = df.index.get_loc(date_idx)
                
                if position is None:
                    # Check higher timeframe trend
                    trend_ok, trend_reason = self.check_higher_timeframe_trend(df, position_idx)
                    if not trend_ok:
                        trend_filter_count += 1
                        continue
                    
                    # Check broom setup
                    if self.check_broom_setup(df, position_idx):
                        setup_count += 1
                        if self.check_entry_signal(df, position_idx):
                            entry_count += 1
                            entry_price = df.iloc[position_idx]['Close']
                            ema_200 = df.iloc[position_idx]['EMA_200']
                            stop_loss = ema_200 * (1 - self.config.stop_loss_buffer)
                            
                            lookback_start = max(0, position_idx - self.config.base_lookback_period)
                            lookback_data = df.iloc[lookback_start:position_idx]
                            
                            peak_price = lookback_data['High'].max()
                            peak_idx = lookback_data['High'].idxmax()
                            peak_pos = df.index.get_loc(peak_idx)
                            
                            consolidation_data = df.iloc[peak_pos:position_idx+1]
                            lowest_price = consolidation_data['Low'].min()
                            base_depth = peak_price - lowest_price
                            take_profit = entry_price + (self.config.measured_move_multiplier * base_depth)
                            
                            position = {
                                'entry_date': date_idx,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'base_depth': base_depth,
                                'entry_position': position_idx,
                                'trend_reason': trend_reason
                            }
                            
                            logger.info(f"🚀 ENTRY: {ticker} at {date_idx.date()} | "
                                      f"Price: {entry_price:.2f} | Target: {take_profit:.2f} | {trend_reason}")
                            
                            if self.trade_log_file:
                                self.trade_log_file.write(
                                    f"ENTRY,{ticker},{date_idx.date()},{entry_price:.2f},{stop_loss:.2f},{take_profit:.2f},{trend_reason}\n"
                                )
                                self.trade_log_file.flush()
                else:
                    # Manage existing position
                    current_price = df.iloc[position_idx]['Close']
                    current_high = df.iloc[position_idx]['High']
                    current_low = df.iloc[position_idx]['Low']
                    ema_200 = df.iloc[position_idx]['EMA_200']
                    
                    new_stop = ema_200 * (1 - self.config.stop_loss_buffer)
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                    
                    if current_low <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        days_held = (date_idx - position['entry_date']).days
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': date_idx,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': 'stop_loss',
                            'base_depth': position['base_depth'],
                            'days_held': days_held
                        })
                        
                        logger.info(f"🛑 EXIT (Stop): {ticker} | Return: {return_pct:.2f}% | Days: {days_held}")
                        
                        if self.trade_log_file:
                            self.trade_log_file.write(
                                f"EXIT_STOP,{ticker},{date_idx.date()},{exit_price:.2f},{return_pct:.2f}%\n"
                            )
                            self.trade_log_file.flush()
                        
                        position = None
                        continue
                    
                    if current_high >= position['take_profit']:
                        exit_price = position['take_profit']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        days_held = (date_idx - position['entry_date']).days
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': date_idx,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': 'take_profit',
                            'base_depth': position['base_depth'],
                            'days_held': days_held
                        })
                        
                        logger.info(f"🎯 EXIT (Target): {ticker} | Return: {return_pct:.2f}% | Days: {days_held}")
                        
                        if self.trade_log_file:
                            self.trade_log_file.write(
                                f"EXIT_TP,{ticker},{date_idx.date()},{exit_price:.2f},{return_pct:.2f}%\n"
                            )
                            self.trade_log_file.flush()
                        
                        position = None
            
            if position is not None:
                last_date = backtest_df.index[-1]
                last_position = df.index.get_loc(last_date)
                last_price = df.iloc[last_position]['Close']
                return_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
                days_held = (last_date - position['entry_date']).days
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': last_date,
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'return_pct': return_pct,
                    'exit_reason': 'end_of_period',
                    'base_depth': position['base_depth'],
                    'days_held': days_held
                })
            
            if setup_count > 0 or trend_filter_count > 0:
                logger.info(f"📊 {ticker}: {trend_filter_count} trend-filtered, {setup_count} setups, {entry_count} entries, {len(trades)} trades")
            
            return trades
            
        except Exception as e:
            logger.debug(f"Error in backtest for {ticker}: {e}")
            return trades
    
    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'avg_return_per_trade': 0,
                'profit_factor': 0,
                'avg_days_held': 0
            }
        
        try:
            trades_df = pd.DataFrame(trades)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['return_pct'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            
            cumulative_return = 100.0
            equity_curve = [100.0]
            
            for _, trade in trades_df.iterrows():
                trade_return = trade['return_pct'] / 100
                cumulative_return *= (1 + trade_return)
                equity_curve.append(cumulative_return)
            
            total_return = cumulative_return - 100
            
            equity_series = pd.Series(equity_curve)
            max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
            
            avg_return_per_trade = trades_df['return_pct'].mean()
            
            gross_profit = trades_df[trades_df['return_pct'] > 0]['return_pct'].sum()
            gross_loss = abs(trades_df[trades_df['return_pct'] < 0]['return_pct'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_days_held = trades_df['days_held'].mean() if 'days_held' in trades_df.columns else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'avg_return_per_trade': avg_return_per_trade,
                'profit_factor': profit_factor,
                'avg_days_held': avg_days_held
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'avg_return_per_trade': 0,
                'profit_factor': 0,
                'avg_days_held': 0
            }
    
    def run_universe_backtest(self, ticker_list: List[str]) -> List[Dict]:
        """Run backtest for universe"""
        logger.info("=" * 80)
        logger.info("STARTING UNIVERSE BACKTEST")
        logger.info(f"Universe size: {len(ticker_list)} stocks")
        logger.info(f"Backtest period: {self.config.backtest_start} to present")
        logger.info("=" * 80)
        
        for batch_start in range(0, len(ticker_list), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(ticker_list))
            batch = ticker_list[batch_start:batch_end]
            
            logger.info(f"\nProcessing batch {batch_start//self.config.batch_size + 1}: "
                       f"stocks {batch_start+1}-{batch_end}")
            
            for i, ticker in enumerate(batch):
                global_idx = batch_start + i
                
                try:
                    logger.info(f"\n[{global_idx+1}/{len(ticker_list)}] Processing {ticker}...")
                    
                    df = self.download_data(ticker)
                    
                    if df is None or len(df) < 300:
                        logger.warning(f"  ✗ Insufficient data for {ticker}")
                        self.results.append({
                            'ticker': ticker,
                            'total_trades': 0,
                            'win_rate': 0,
                            'total_return': 0,
                            'max_drawdown': 0,
                            'avg_return_per_trade': 0,
                            'profit_factor': 0,
                            'avg_days_held': 0,
                            'processed': False,
                            'reason': 'insufficient_data'
                        })
                        self.stocks_failed += 1
                        continue
                    
                    trades = self.run_backtest(df, ticker)
                    
                    metrics = self.calculate_performance_metrics(trades)
                    metrics['ticker'] = ticker
                    metrics['processed'] = True
                    metrics['reason'] = 'success'
                    
                    self.results.append(metrics)
                    self.stocks_processed += 1
                    self.total_trades += len(trades)
                    
                    if trades:
                        logger.info(f"  ✓ {metrics['total_trades']} trades, "
                                  f"Win Rate: {metrics['win_rate']:.1f}%, "
                                  f"Return: {metrics['total_return']:.1f}%")
                    else:
                        logger.info(f"  ✓ No qualifying trades")
                    
                    del df, trades
                    gc.collect()
                    
                    time.sleep(self.config.request_delay)
                    
                except Exception as e:
                    logger.error(f"  ✗ Error processing {ticker}: {str(e)}")
                    self.results.append({
                        'ticker': ticker,
                        'total_trades': 0,
                        'win_rate': 0,
                        'total_return': 0,
                        'max_drawdown': 0,
                        'avg_return_per_trade': 0,
                        'profit_factor': 0,
                        'avg_days_held': 0,
                        'processed': False,
                        'reason': f'error: {str(e)}'
                    })
                    self.stocks_failed += 1
            
            if batch_end < len(ticker_list):
                logger.info(f"\nBatch complete. Waiting {self.config.batch_delay} seconds...")
                time.sleep(self.config.batch_delay)
        
        elapsed_time = time.time() - self.start_time
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETED")
        logger.info(f"Stocks processed: {self.stocks_processed}/{len(ticker_list)}")
        logger.info(f"Stocks failed: {self.stocks_failed}")
        logger.info(f"Total trades: {self.total_trades}")
        logger.info(f"Time elapsed: {elapsed_time/60:.2f} minutes")
        logger.info("=" * 80)
        
        return self.results
    
    def export_results(self, filename: str = 'nifty500_broom_breakout_results.csv') -> pd.DataFrame:
        """Export results to CSV"""
        results_df = pd.DataFrame(self.results)
        
        if results_df.empty:
            logger.error("No results to export")
            return pd.DataFrame()
        
        output_path = self.config.results_dir / filename
        results_df.to_csv(output_path, index=False)
        results_df.to_csv(filename, index=False)
        
        logger.info(f"\nResults exported to {output_path}")
        logger.info(f"Total records: {len(results_df)}")
        
        processed_df = results_df[results_df['processed'] == True]
        if len(processed_df) > 0:
            logger.info("\n=== SUMMARY STATISTICS ===")
            logger.info(f"Stocks processed: {len(processed_df)}")
            logger.info(f"Stocks with trades: {len(processed_df[processed_df['total_trades'] > 0])}")
            logger.info(f"Total trades: {processed_df['total_trades'].sum()}")
            logger.info(f"Average win rate: {processed_df['win_rate'].mean():.2f}%")
            logger.info(f"Average return: {processed_df['total_return'].mean():.2f}%")
            
            stocks_with_trades = processed_df[processed_df['total_trades'] > 0]
            if len(stocks_with_trades) > 0:
                top_performers = stocks_with_trades.nlargest(10, 'total_return')[
                    ['ticker', 'total_return', 'win_rate', 'total_trades']
                ]
                logger.info("\n=== TOP 10 PERFORMERS ===")
                for _, row in top_performers.iterrows():
                    logger.info(f"  {row['ticker']}: {row['total_return']:.1f}% return, "
                              f"{row['win_rate']:.1f}% win rate, {row['total_trades']} trades")
        
        return results_df

# ==================== NIFTY 500 UNIVERSE ====================

def get_nifty500_tickers() -> List[str]:
    """Get list of Nifty 500 tickers - testing with 20 stocks"""
    
    nifty_500 = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
    ]
    
    if config.universe_size < len(nifty_500):
        logger.info(f"Using subset of {config.universe_size} stocks")
        nifty_500 = nifty_500[:config.universe_size]
    
    return nifty_500

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    try:
        logger.info("=" * 80)
        logger.info("INSTITUTIONAL MOVING AVERAGE 'BROOM' BREAKOUT STRATEGY")
        logger.info("Nifty 500 Universe - Corrected Trend Filters")
        logger.info("=" * 80)
        
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Pandas version: {pd.__version__}")
        logger.info(f"Numpy version: {np.__version__}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        tickers = get_nifty500_tickers()
        logger.info(f"\nUniverse size: {len(tickers)} stocks")
        
        engine = BroomBreakoutBacktest(config)
        
        results = engine.run_universe_backtest(tickers)
        
        results_df = engine.export_results()
        
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return results_df
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
