"""
Institutional Moving Average "Broom" Breakout Strategy Backtesting Engine
For Nifty 500 Universe - GitHub Actions Optimized Version
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
        
        # API Keys from environment (GitHub Secrets)
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
        
        self.broom_compression_threshold = float(os.getenv('BROOM_COMPRESSION_THRESHOLD', '0.08'))
        self.base_lookback_period = 200
        self.base_duration_min = int(os.getenv('BASE_DURATION_MIN', '63'))
        self.base_duration_max = int(os.getenv('BASE_DURATION_MAX', '147'))
        self.box_consolidation_height = float(os.getenv('BOX_CONSOLIDATION_HEIGHT', '0.15'))
        self.prior_trend_exhaustion_limit = float(os.getenv('PRIOR_TREND_EXHAUSTION_LIMIT', '0.60'))
        
        # Execution Parameters
        self.volume_poc_bins = 10
        self.poc_lookback = 20
        self.volume_threshold_multiplier = float(os.getenv('VOLUME_THRESHOLD_MULTIPLIER', '1.5'))
        self.volume_ma_period = 50
        self.stop_loss_buffer = float(os.getenv('STOP_LOSS_BUFFER', '0.015'))
        self.measured_move_multiplier = float(os.getenv('MEASURED_MOVE_MULTIPLIER', '2'))
        
        # Rate Limiting
        self.request_delay = float(os.getenv('REQUEST_DELAY', '2'))
        self.batch_delay = float(os.getenv('BATCH_DELAY', '10'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        
        # Cache settings
        self.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.cache_expiry_days = int(os.getenv('CACHE_EXPIRY_DAYS', '7'))
        
        # Directories
        self.data_dir = Path('data_cache')
        self.results_dir = Path('results')
        self.logs_dir = Path('logs')
        
        # Create directories
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.universe_size <= 0:
            raise ValueError("UNIVERSE_SIZE must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")
        
        if self.broom_compression_threshold <= 0 or self.broom_compression_threshold >= 1:
            raise ValueError("BROOM_COMPRESSION_THRESHOLD must be between 0 and 1")
        
        if self.base_duration_min >= self.base_duration_max:
            raise ValueError("BASE_DURATION_MIN must be less than BASE_DURATION_MAX")

# ==================== LOGGING SETUP ====================

def setup_logging(config: Config):
    """Setup logging configuration"""
    log_file = config.logs_dir / 'backtest.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize configuration and logging
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
        
        # Set headers to mimic browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json,text/plain,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        return session
    
    def fetch_from_yfinance(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance"""
        try:
            import yfinance as yf
            
            logger.debug(f"Fetching {ticker} from yfinance")
            
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
                        logger.info(f"✓ yfinance: {len(df)} rows for {ticker}")
                        return df
            
            return None
            
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(f"Rate limited by yfinance for {ticker}")
                self.rate_limit_hits += 1
            else:
                logger.debug(f"yfinance failed for {ticker}: {str(e)}")
            return None
    
    def fetch_from_stooq(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from Stooq directly via HTTP"""
        try:
            # Convert ticker format for Stooq
            if ticker.endswith('.NS'):
                stooq_ticker = ticker.replace('.NS', '.NSE')
            elif ticker.endswith('.BO'):
                stooq_ticker = ticker.replace('.BO', '.BSE')
            else:
                stooq_ticker = ticker + '.NSE'
            
            logger.debug(f"Fetching {ticker} from Stooq as {stooq_ticker}")
            
            # Stooq URL
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
                            logger.info(f"✓ Stooq: {len(df)} rows for {ticker}")
                            return df
            
            return None
            
        except Exception as e:
            logger.debug(f"Stooq failed for {ticker}: {str(e)}")
            return None
    
    def fetch_from_alpha_vantage(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch from Alpha Vantage using API key"""
        if not self.config.alpha_vantage_api_key:
            logger.debug("Alpha Vantage API key not configured")
            return None
        
        try:
            # Convert ticker format for Alpha Vantage
            if ticker.endswith('.NS'):
                av_ticker = ticker.replace('.NS', '.BSE')
            elif ticker.endswith('.BO'):
                av_ticker = ticker.replace('.BO', '.BSE')
            else:
                av_ticker = ticker
            
            logger.debug(f"Fetching {ticker} from Alpha Vantage as {av_ticker}")
            
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_ticker}&outputsize=full&apikey={self.config.alpha_vantage_api_key}"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)
                    
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df = df.astype(float)
                    
                    df = df[df.index >= start_date]
                    
                    if len(df) > 100:
                        logger.info(f"✓ Alpha Vantage: {len(df)} rows for {ticker}")
                        return df
                elif 'Note' in data:
                    logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                    self.rate_limit_hits += 1
            
            return None
            
        except Exception as e:
            logger.debug(f"Alpha Vantage failed for {ticker}: {str(e)}")
            return None
    
    def fetch_with_fallback(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data with multiple source fallback"""
        
        if self.rate_limit_hits >= 3:
            logger.warning(f"Multiple rate limit hits ({self.rate_limit_hits}). Waiting 60 seconds...")
            time.sleep(60)
            self.rate_limit_hits = 0
        
        data_sources = [
            ('yfinance', self.fetch_from_yfinance),
            ('stooq', self.fetch_from_stooq),
            ('alpha_vantage', self.fetch_from_alpha_vantage),
        ]
        
        for source_name, fetch_func in data_sources:
            try:
                df = fetch_func(ticker, start_date)
                
                if df is not None and not df.empty and len(df) > 100:
                    self.consecutive_failures = 0
                    return df
                    
            except Exception as e:
                logger.error(f"Error with {source_name} for {ticker}: {str(e)}")
                
                if "429" in str(e):
                    self.rate_limit_hits += 1
        
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= 5:
            logger.warning(f"{self.consecutive_failures} consecutive failures. Waiting 120 seconds...")
            time.sleep(120)
            self.consecutive_failures = 0
        
        return None

# ==================== BACKTEST ENGINE ====================

class BroomBreakoutBacktest:
    """Main backtesting engine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = []
        self.data_fetcher = DataFetcher(config)
        
        self.start_time = time.time()
        self.stocks_processed = 0
        self.stocks_failed = 0
        self.total_trades = 0
        
        logger.info("=" * 80)
        logger.info("BROOM BREAKOUT BACKTEST ENGINE INITIALIZED")
        logger.info("=" * 80)
    
    def save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save data to cache"""
        if not self.config.cache_enabled:
            return
        
        try:
            cache_file = self.config.data_dir / f"{ticker.replace('.', '_')}.csv"
            df.to_csv(cache_file)
            logger.debug(f"Cached data for {ticker}")
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
                    logger.info(f"✓ Cache: {len(df)} rows for {ticker}")
                    return df
                else:
                    logger.debug(f"Cache expired for {ticker}")
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
        """Create weekly and monthly timeframes"""
        try:
            weekly_df = df.resample('W-FRI').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            weekly_df[f'EMA_{self.config.weekly_ema_period}_Weekly'] = weekly_df['Close'].ewm(
                span=self.config.weekly_ema_period, adjust=False
            ).mean()
            
            monthly_df = df.resample('ME').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
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
            
        except Exception as e:
            logger.error(f"Error creating higher timeframes: {e}")
            df['Weekly_200_EMA'] = np.nan
            df['Monthly_200_EMA'] = np.nan
        
        return df
    
    def check_broom_setup(self, df: pd.DataFrame, idx: int) -> bool:
        """Check broom setup conditions"""
        if idx < self.config.base_lookback_period:
            return False
        
        try:
            current_price = df.loc[idx, 'Close']
            if pd.isna(current_price) or current_price <= 0:
                return False
            
            weekly_ema = df.loc[idx, 'Weekly_200_EMA']
            if pd.isna(weekly_ema) or current_price <= weekly_ema:
                return False
            
            monthly_ema = df.loc[idx, 'Monthly_200_EMA']
            if not pd.isna(monthly_ema) and current_price <= monthly_ema:
                return False
            
            ema_values = []
            for period in self.config.ema_periods:
                ema_val = df.loc[idx, f'EMA_{period}']
                if pd.isna(ema_val):
                    return False
                ema_values.append(ema_val)
            
            ema_high = max(ema_values)
            ema_low = min(ema_values)
            ema_spread = (ema_high - ema_low) / current_price
            
            if ema_spread >= self.config.broom_compression_threshold:
                return False
            
            lookback_data = df.iloc[max(0, idx - self.config.base_lookback_period):idx]
            if len(lookback_data) < self.config.base_lookback_period:
                return False
            
            peak_idx = lookback_data['High'].idxmax()
            peak_price = lookback_data['High'].max()
            peak_position = df.index.get_loc(peak_idx)
            days_since_peak = idx - peak_position
            
            if days_since_peak < self.config.base_duration_min or \
               days_since_peak > self.config.base_duration_max:
                return False
            
            recent_20 = df.iloc[idx-19:idx+1]
            box_height = (recent_20['High'].max() - recent_20['Low'].min()) / current_price
            
            if box_height >= self.config.box_consolidation_height:
                return False
            
            pre_peak_data = df.iloc[max(0, peak_position - self.config.base_lookback_period):peak_position+1]
            if len(pre_peak_data) > 0:
                lowest_low = pre_peak_data['Low'].min()
                if lowest_low > 0:
                    run_up = (peak_price - lowest_low) / lowest_low
                    
                    if run_up > self.config.prior_trend_exhaustion_limit:
                        return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in broom setup check: {e}")
            return False
    
    def calculate_volume_profile_poc(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """Calculate Volume Profile POC"""
        if idx < self.config.poc_lookback:
            return None
        
        try:
            recent_data = df.iloc[idx - self.config.poc_lookback + 1:idx + 1]
            
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
            logger.debug(f"Error in POC calculation: {e}")
            return None
    
    def check_entry_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """Check entry trigger conditions"""
        try:
            current_price = df.loc[idx, 'Close']
            
            ema_values = [df.loc[idx, f'EMA_{period}'] for period in self.config.ema_periods]
            highest_ema = max(ema_values)
            
            poc_price = self.calculate_volume_profile_poc(df, idx)
            if poc_price is None:
                return False
            
            if current_price <= highest_ema:
                return False
            if current_price <= poc_price:
                return False
            
            if idx < self.config.volume_ma_period:
                return False
            
            volume_ma = df.iloc[idx - self.config.volume_ma_period:idx]['Volume'].mean()
            current_volume = df.loc[idx, 'Volume']
            
            if current_volume <= self.config.volume_threshold_multiplier * volume_ma:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in entry signal: {e}")
            return False
    
    def run_backtest(self, df: pd.DataFrame) -> List[Dict]:
        """Run backtest on single stock"""
        trades = []
        position = None
        
        try:
            df = self.calculate_emas(df)
            df = self.create_higher_timeframes(df)
            
            backtest_df = df[df.index >= self.config.backtest_start].copy()
            
            if len(backtest_df) < 50:
                return trades
            
            logger.debug(f"Running backtest on {len(backtest_df)} days")
            
            for idx in backtest_df.index:
                position_idx = df.index.get_loc(idx)
                
                if position is None:
                    if self.check_broom_setup(df, position_idx):
                        if self.check_entry_signal(df, position_idx):
                            entry_price = df.loc[idx, 'Close']
                            stop_loss = df.loc[idx, 'EMA_200'] * (1 - self.config.stop_loss_buffer)
                            
                            lookback_data = df.iloc[max(0, position_idx - self.config.base_lookback_period):position_idx]
                            peak_price = lookback_data['High'].max()
                            peak_pos = df.index.get_loc(lookback_data['High'].idxmax())
                            consolidation_data = df.iloc[peak_pos:position_idx+1]
                            lowest_price = consolidation_data['Low'].min()
                            base_depth = peak_price - lowest_price
                            take_profit = entry_price + (self.config.measured_move_multiplier * base_depth)
                            
                            position = {
                                'entry_date': idx,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'base_depth': base_depth
                            }
                            
                            logger.debug(f"Entry: {idx.date()} @ {entry_price:.2f}")
                else:
                    current_price = df.loc[idx, 'Close']
                    current_high = df.loc[idx, 'High']
                    current_low = df.loc[idx, 'Low']
                    ema_200 = df.loc[idx, 'EMA_200']
                    
                    new_stop = ema_200 * (1 - self.config.stop_loss_buffer)
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                    
                    if current_low <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': idx,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': 'stop_loss',
                            'base_depth': position['base_depth']
                        })
                        
                        logger.debug(f"Exit (Stop Loss): {idx.date()} Return: {return_pct:.2f}%")
                        position = None
                        continue
                    
                    if current_high >= position['take_profit']:
                        exit_price = position['take_profit']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': idx,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': 'take_profit',
                            'base_depth': position['base_depth']
                        })
                        
                        logger.debug(f"Exit (Take Profit): {idx.date()} Return: {return_pct:.2f}%")
                        position = None
            
            if position is not None:
                last_idx = backtest_df.index[-1]
                last_price = backtest_df.loc[last_idx, 'Close']
                return_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': last_idx,
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'return_pct': return_pct,
                    'exit_reason': 'end_of_period',
                    'base_depth': position['base_depth']
                })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
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
                'profit_factor': 0
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
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'avg_return_per_trade': avg_return_per_trade,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'avg_return_per_trade': 0,
                'profit_factor': 0
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
                            'processed': False,
                            'reason': 'insufficient_data'
                        })
                        self.stocks_failed += 1
                        continue
                    
                    trades = self.run_backtest(df)
                    
                    metrics = self.calculate_performance_metrics(trades)
                    metrics['ticker'] = ticker
                    metrics['processed'] = True
                    metrics['reason'] = 'success'
                    
                    self.results.append(metrics)
                    self.stocks_processed += 1
                    self.total_trades += len(trades)
                    
                    if trades:
                        logger.info(f"  ✓ {metrics['total_trades']} trades, "
                                  f"Win Rate: {
