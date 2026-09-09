"""
Enhanced Institutional Moving Average "Broom" Breakout Strategy
For Nifty 500 High Volume Stocks - Multi-Layered Selection & Position Sizing
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
    """Enhanced configuration with multi-layered filters"""
    
    def __init__(self):
        # Environment detection
        self.is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # API Keys
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        
        # Backtest Period
        self.backtest_start = os.getenv('BACKTEST_START', '2018-01-01')
        self.data_start = os.getenv('DATA_START', '2000-01-01')
        
        # Universe Configuration
        self.universe_size = int(os.getenv('UNIVERSE_SIZE', '500'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        
        # Enhanced Strategy Parameters
        self.ema_periods = [20, 50, 100, 200]
        
        # Multi-Layer Selection Filters
        self.min_win_rate = float(os.getenv('MIN_WIN_RATE', '0.40'))  # 40% minimum win rate
        self.min_profit_factor = float(os.getenv('MIN_PROFIT_FACTOR', '2.5'))  # 2.5 minimum PF
        self.min_return_drawdown = float(os.getenv('MIN_RETURN_DRAWDOWN', '2.0'))  # 2.0 minimum RDD
        
        # Sector Classification & Filters
        self.excluded_sectors = ['PSU', 'Commodities', 'Real Estate']
        self.high_conviction_sectors = ['Pharma', 'Banking', 'Auto', 'IT', 'FMCG']
        self.sector_weights = {
            'Pharma': 1.5,
            'Banking': 1.3,
            'Auto': 1.2,
            'IT': 1.1,
            'FMCG': 1.1,
            'Others': 1.0,
            'Metals': 0.7,
            'Energy': 0.8,
            'Infrastructure': 0.7
        }
        
        # High Conviction Stocks
        self.high_conviction_stocks = {
            'DIVISLAB.NS': 2.0,  # 2x position size
            'TRENT.NS': 1.8,
            'CUMMINSIND.NS': 1.7,
            'TITAN.NS': 1.6,
            'ASIANPAINT.NS': 1.5,
            'PIDILITIND.NS': 1.5,
            'HAVELLS.NS': 1.4,
            'BAJFINANCE.NS': 1.4,
            'DIXON.NS': 1.3,
            'ASTRAL.NS': 1.3
        }
        
        # Dynamic Position Sizing
        self.base_position_size = float(os.getenv('BASE_POSITION_SIZE', '0.05'))  # 5% base
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.10'))  # 10% max
        self.min_position_size = float(os.getenv('MIN_POSITION_SIZE', '0.02'))  # 2% min
        
        # ATR Trailing Stop
        self.atr_period = 14
        self.atr_multiplier = float(os.getenv('ATR_MULTIPLIER', '1.5'))
        
        # Time-Based Exit
        self.max_holding_days = int(os.getenv('MAX_HOLDING_DAYS', '90'))
        
        # Volatility-Adjusted Entry
        self.min_volume_surge = float(os.getenv('MIN_VOLUME_SURGE', '0.50'))  # 50% above 20-day average
        self.rsi_period = 14
        self.rsi_min = float(os.getenv('RSI_MIN', '55'))
        self.rsi_max = float(os.getenv('RSI_MAX', '75'))
        
        # Broom Compression
        self.broom_compression_threshold = float(os.getenv('BROOM_COMPRESSION_THRESHOLD', '0.15'))
        self.base_lookback_period = 150
        self.base_duration_min = int(os.getenv('BASE_DURATION_MIN', '30'))
        self.base_duration_max = int(os.getenv('BASE_DURATION_MAX', '200'))
        self.box_consolidation_height = float(os.getenv('BOX_CONSOLIDATION_HEIGHT', '0.25'))
        
        # Liquidity Filters
        self.min_price = float(os.getenv('MIN_PRICE', '100'))
        self.max_price = float(os.getenv('MAX_PRICE', '5000'))
        self.min_volume = float(os.getenv('MIN_VOLUME', '500000'))
        self.min_value_traded = float(os.getenv('MIN_VALUE_TRADED', '50000000'))
        
        # Rate Limiting
        self.request_delay = float(os.getenv('REQUEST_DELAY', '1'))
        self.batch_delay = float(os.getenv('BATCH_DELAY', '5'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        
        # Cache & Checkpoint
        self.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.cache_expiry_days = int(os.getenv('CACHE_EXPIRY_DAYS', '7'))
        self.checkpoint_file = 'checkpoint.json'
        self.save_checkpoint = os.getenv('SAVE_CHECKPOINT', 'true').lower() == 'true'
        
        # Directories
        self.data_dir = Path('data_cache')
        self.results_dir = Path('results')
        self.logs_dir = Path('logs')
        
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

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

config = Config()
logger = setup_logging(config)

# ==================== SECTOR CLASSIFICATION ====================

class SectorClassifier:
    """Classify stocks into sectors"""
    
    def __init__(self):
        self.sector_map = {
            # Pharma & Healthcare
            'SUNPHARMA.NS': 'Pharma', 'DIVISLAB.NS': 'Pharma', 'CIPLA.NS': 'Pharma',
            'DRREDDY.NS': 'Pharma', 'LUPIN.NS': 'Pharma', 'AUROPHARMA.NS': 'Pharma',
            'BIOCON.NS': 'Pharma', 'GLENMARK.NS': 'Pharma', 'ALKEM.NS': 'Pharma',
            'TORNTPHARM.NS': 'Pharma', 'ZYDUSLIFE.NS': 'Pharma', 'GRANULES.NS': 'Pharma',
            'LAURUSLABS.NS': 'Pharma', 'NATCOPHARM.NS': 'Pharma', 'PPLPHARMA.NS': 'Pharma',
            'SYNGENE.NS': 'Pharma', 'APOLLOHOSP.NS': 'Pharma',
            
            # Banking & Financial Services
            'HDFCBANK.NS': 'Banking', 'ICICIBANK.NS': 'Banking', 'SBIN.NS': 'Banking',
            'KOTAKBANK.NS': 'Banking', 'AXISBANK.NS': 'Banking', 'INDUSINDBK.NS': 'Banking',
            'BANKBARODA.NS': 'Banking', 'PNB.NS': 'Banking', 'FEDERALBNK.NS': 'Banking',
            'AUBANK.NS': 'Banking', 'BANDHANBNK.NS': 'Banking', 'CANBK.NS': 'Banking',
            'BANKINDIA.NS': 'Banking', 'INDIANB.NS': 'Banking', 'KARURVYSYA.NS': 'Banking',
            'RBLBANK.NS': 'Banking', 'UCOBANK.NS': 'Banking', 'UNIONBANK.NS': 'Banking',
            'YESBANK.NS': 'Banking', 'IDFCFIRSTB.NS': 'Banking',
            'BAJFINANCE.NS': 'Banking', 'BAJAJFINSV.NS': 'Banking', 'CHOLAFIN.NS': 'Banking',
            'MUTHOOTFIN.NS': 'Banking', 'M&MFIN.NS': 'Banking', 'LICHSGFIN.NS': 'Banking',
            'SBICARD.NS': 'Banking', 'ICICIGI.NS': 'Banking', 'HDFCLIFE.NS': 'Banking',
            'SBILIFE.NS': 'Banking', 'ICICIPRULI.NS': 'Banking',
            
            # Auto & Auto Ancillaries
            'MARUTI.NS': 'Auto', 'TATAMOTORS.NS': 'Auto', 'M&M.NS': 'Auto',
            'BAJAJ-AUTO.NS': 'Auto', 'EICHERMOT.NS': 'Auto', 'HEROMOTOCO.NS': 'Auto',
            'TVSMOTOR.NS': 'Auto', 'ASHOKLEY.NS': 'Auto', 'BHARATFORG.NS': 'Auto',
            'BOSCHLTD.NS': 'Auto', 'ESCORTS.NS': 'Auto', 'EXIDEIND.NS': 'Auto',
            'MOTHERSON.NS': 'Auto', 'MRF.NS': 'Auto',
            
            # IT & Technology
            'TCS.NS': 'IT', 'INFY.NS': 'IT', 'WIPRO.NS': 'IT', 'HCLTECH.NS': 'IT',
            'TECHM.NS': 'IT', 'LTIM.NS': 'IT', 'MPHASIS.NS': 'IT', 'COFORGE.NS': 'IT',
            'PERSISTENT.NS': 'IT', 'CYIENT.NS': 'IT', 'LTTS.NS': 'IT',
            'TATAELXSI.NS': 'IT', 'KPITTECH.NS': 'IT', 'ZENSARTECH.NS': 'IT',
            'INTELLECT.NS': 'IT', 'HAPPSTMNDS.NS': 'IT',
            
            # FMCG & Consumer
            'HINDUNILVR.NS': 'FMCG', 'ITC.NS': 'FMCG', 'NESTLEIND.NS': 'FMCG',
            'BRITANNIA.NS': 'FMCG', 'DABUR.NS': 'FMCG', 'MARICO.NS': 'FMCG',
            'GODREJCP.NS': 'FMCG', 'GODREJIND.NS': 'FMCG', 'EMAMI.NS': 'FMCG',
            'TATACONSUM.NS': 'FMCG', 'UBL.NS': 'FMCG', 'VBL.NS': 'FMCG',
            'RADICO.NS': 'FMCG',
            
            # High Conviction Multi-baggers
            'TRENT.NS': 'FMCG', 'DMART.NS': 'FMCG', 'TITAN.NS': 'FMCG',
            'ASIANPAINT.NS': 'FMCG', 'PIDILITIND.NS': 'FMCG', 'HAVELLS.NS': 'FMCG',
            'DIXON.NS': 'FMCG', 'ASTRAL.NS': 'FMCG', 'CUMMINSIND.NS': 'FMCG',
            
            # Metals & Mining (Excluded - Commodities)
            'TATASTEEL.NS': 'Commodities', 'JSWSTEEL.NS': 'Commodities', 'HINDALCO.NS': 'Commodities',
            'VEDL.NS': 'Commodities', 'JINDALSTEL.NS': 'Commodities', 'SAIL.NS': 'Commodities',
            'NATIONALUM.NS': 'Commodities', 'HINDCOPPER.NS': 'Commodities', 'COALINDIA.NS': 'Commodities',
            'HINDZINC.NS': 'Commodities',
            
            # Energy & Power
            'RELIANCE.NS': 'Energy', 'ONGC.NS': 'Energy', 'BPCL.NS': 'Energy',
            'IOC.NS': 'Energy', 'GAIL.NS': 'Energy', 'NTPC.NS': 'Energy',
            'POWERGRID.NS': 'Energy', 'TATAPOWER.NS': 'Energy', 'ADANIGREEN.NS': 'Energy',
            'ADANIPOWER.NS': 'Energy', 'JSWENERGY.NS': 'Energy', 'NHPC.NS': 'Energy',
            
            # Infrastructure & Construction
            'LT.NS': 'Infrastructure', 'ADANIPORTS.NS': 'Infrastructure', 'DLF.NS': 'Infrastructure',
            'GODREJPROP.NS': 'Infrastructure', 'OBEROIRLTY.NS': 'Infrastructure', 'PRESTIGE.NS': 'Infrastructure',
            'NBCC.NS': 'Infrastructure', 'NCC.NS': 'Infrastructure', 'IRB.NS': 'Infrastructure',
            'BEL.NS': 'Infrastructure', 'BHEL.NS': 'Infrastructure',
            
            # Real Estate (Excluded)
            'BRIGADE.NS': 'Real Estate', 'SUNTECK.NS': 'Real Estate', 'PHOENIXLTD.NS': 'Real Estate',
            
            # PSU (Excluded)
            'IOC.NS': 'PSU', 'BPCL.NS': 'PSU', 'ONGC.NS': 'PSU', 'GAIL.NS': 'PSU',
            'NTPC.NS': 'PSU', 'POWERGRID.NS': 'PSU', 'COALINDIA.NS': 'PSU',
            'SBIN.NS': 'PSU', 'PNB.NS': 'PSU', 'BANKBARODA.NS': 'PSU',
            
            # Others
            'SIEMENS.NS': 'Others', 'ABB.NS': 'Others', 'CUMMINSIND.NS': 'Others',
            'VOLTAS.NS': 'Others', 'BLUESTARCO.NS': 'Others', 'CROMPTON.NS': 'Others',
            'KAJARIACER.NS': 'Others', 'KANSAINER.NS': 'Others', 'KEI.NS': 'Others',
            'POLYCAB.NS': 'Others', 'SUPREMEIND.NS': 'Others', 'WHIRLPOOL.NS': 'Others',
        }
    
    def get_sector(self, ticker: str) -> str:
        """Get sector for a ticker"""
        return self.sector_map.get(ticker, 'Others')
    
    def is_excluded(self, ticker: str) -> bool:
        """Check if stock is in excluded sector"""
        sector = self.get_sector(ticker)
        return sector in ['PSU', 'Commodities', 'Real Estate']
    
    def get_sector_weight(self, ticker: str) -> float:
        """Get sector weight for position sizing"""
        sector = self.get_sector(ticker)
        return config.sector_weights.get(sector, 1.0)

# ==================== DATA FETCHING ====================

class DataFetcher:
    """Multi-source data fetcher"""
    
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
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=5, pool_maxsize=5)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json,text/plain,*/*',
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
            if "429" in str(e):
                self.rate_limit_hits += 1
            return None
    
    def fetch_with_fallback(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data with fallback"""
        
        if self.rate_limit_hits >= 5:
            logger.warning(f"Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            self.rate_limit_hits = 0
        
        df = self.fetch_from_yfinance(ticker, start_date)
        
        if df is not None and not df.empty and len(df) > 100:
            return df
        
        return None

# ==================== ENHANCED BACKTEST ENGINE ====================

class EnhancedBroomBreakoutBacktest:
    """Enhanced backtesting engine with multi-layered filters"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = []
        self.data_fetcher = DataFetcher(config)
        self.sector_classifier = SectorClassifier()
        
        self.start_time = time.time()
        self.stocks_processed = 0
        self.stocks_failed = 0
        self.total_trades = 0
        
        # Checkpoint
        self.checkpoint = self.load_checkpoint()
        
        # Trade log
        self.trade_log_file = open(config.logs_dir / 'trades.log', 'a')
        
        logger.info("=" * 80)
        logger.info("ENHANCED BROOM BREAKOUT ENGINE - MULTI-LAYERED SELECTION")
        logger.info(f"Min Win Rate: {self.config.min_win_rate:.0%}")
        logger.info(f"Min Profit Factor: {self.config.min_profit_factor:.1f}")
        logger.info(f"Min Return/Drawdown: {self.config.min_return_drawdown:.1f}")
        logger.info(f"ATR Multiplier: {self.config.atr_multiplier}x")
        logger.info(f"Max Holding Days: {self.config.max_holding_days}")
        logger.info(f"RSI Range: {self.config.rsi_min}-{self.config.rsi_max}")
        logger.info(f"Min Volume Surge: {self.config.min_volume_surge:.0%}")
        logger.info("=" * 80)
    
    def __del__(self):
        if self.trade_log_file:
            self.trade_log_file.close()
    
    def load_checkpoint(self) -> Dict:
        """Load checkpoint"""
        if not self.config.save_checkpoint:
            return {}
        
        try:
            if Path(self.config.checkpoint_file).exists():
                with open(self.config.checkpoint_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
        
        return {}
    
    def save_checkpoint_state(self, current_index: int):
        """Save checkpoint"""
        if not self.config.save_checkpoint:
            return
        
        try:
            checkpoint = {
                'last_processed_index': current_index,
                'processed_stocks': self.stocks_processed,
                'total_trades': self.total_trades,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.config.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save to cache"""
        if not self.config.cache_enabled:
            return
        
        try:
            cache_file = self.config.data_dir / f"{ticker.replace('.', '_')}.csv"
            df.to_csv(cache_file)
        except Exception as e:
            logger.warning(f"Failed to cache {ticker}: {e}")
    
    def load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load from cache"""
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
        """Download data"""
        df = self.load_from_cache(ticker)
        if df is not None:
            return df
        
        df = self.data_fetcher.fetch_with_fallback(ticker, self.config.data_start)
        
        if df is not None and not df.empty:
            self.save_to_cache(ticker, df)
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # EMAs
        for period in self.config.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(self.config.atr_period).mean()
        
        # Volume indicators
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA_50'] = df['Volume'].rolling(window=50).mean()
        df['Volume_Surge'] = (df['Volume'] - df['Volume_MA_20']) / df['Volume_MA_20']
        
        return df
    
    def create_higher_timeframes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weekly and monthly timeframes"""
        try:
            # Weekly
            weekly_df = df.resample('W-FRI').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            weekly_df['Weekly_EMA_20'] = weekly_df['Close'].ewm(span=20, adjust=False).mean()
            weekly_df['Weekly_EMA_50'] = weekly_df['Close'].ewm(span=50, adjust=False).mean()
            
            weekly_df['Weekly_Trend_Up'] = (
                (weekly_df['Close'] > weekly_df['Weekly_EMA_20']) &
                (weekly_df['Weekly_EMA_20'] > weekly_df['Weekly_EMA_50'])
            )
            
            # Monthly
            monthly_df = df.resample('ME').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            monthly_df['Monthly_EMA_10'] = monthly_df['Close'].ewm(span=10, adjust=False).mean()
            monthly_df['Monthly_EMA_20'] = monthly_df['Close'].ewm(span=20, adjust=False).mean()
            
            monthly_df['Monthly_Trend_Up'] = (
                (monthly_df['Close'] > monthly_df['Monthly_EMA_10']) &
                (monthly_df['Monthly_EMA_10'] > monthly_df['Monthly_EMA_20'])
            )
            
            # Map to daily
            df['Weekly_EMA_20'] = weekly_df['Weekly_EMA_20'].reindex(df.index, method='ffill')
            df['Weekly_EMA_50'] = weekly_df['Weekly_EMA_50'].reindex(df.index, method='ffill')
            df['Weekly_Trend_Up'] = weekly_df['Weekly_Trend_Up'].reindex(df.index, method='ffill')
            df['Monthly_EMA_10'] = monthly_df['Monthly_EMA_10'].reindex(df.index, method='ffill')
            df['Monthly_EMA_20'] = monthly_df['Monthly_EMA_20'].reindex(df.index, method='ffill')
            df['Monthly_Trend_Up'] = monthly_df['Monthly_Trend_Up'].reindex(df.index, method='ffill')
            
        except Exception as e:
            logger.debug(f"Error creating higher timeframes: {e}")
            df['Weekly_Trend_Up'] = False
            df['Monthly_Trend_Up'] = False
        
        return df
    
    def calculate_position_size(self, ticker: str, df: pd.DataFrame, position: int) -> float:
        """Dynamic position sizing based on conviction and sector"""
        try:
            base_size = self.config.base_position_size
            
            # High conviction stocks get larger position
            conviction_multiplier = self.config.high_conviction_stocks.get(ticker, 1.0)
            
            # Sector weight
            sector_weight = self.sector_classifier.get_sector_weight(ticker)
            
            # Volatility adjustment
            atr = df.iloc[position]['ATR']
            close = df.iloc[position]['Close']
            if not pd.isna(atr) and close > 0:
                volatility = atr / close
                if volatility > 0.05:  # High volatility
                    volatility_multiplier = 0.7
                elif volatility < 0.02:  # Low volatility
                    volatility_multiplier = 1.3
                else:
                    volatility_multiplier = 1.0
            else:
                volatility_multiplier = 1.0
            
            # Calculate final position size
            position_size = base_size * conviction_multiplier * sector_weight * volatility_multiplier
            
            # Clamp to min/max
            position_size = max(self.config.min_position_size, 
                              min(self.config.max_position_size, position_size))
            
            return position_size
            
        except Exception as e:
            return self.config.base_position_size
    
    def check_multi_layer_filter(self, ticker: str, df: pd.DataFrame, position: int) -> Tuple[bool, str, float]:
        """Multi-layered stock selection filter"""
        try:
            # 1. Sector filter
            if self.sector_classifier.is_excluded(ticker):
                return False, f"Excluded sector: {self.sector_classifier.get_sector(ticker)}", 0
            
            # 2. Liquidity filter
            current_price = df.iloc[position]['Close']
            avg_volume = df.iloc[max(0, position-20):position]['Volume'].mean()
            avg_value = (df.iloc[max(0, position-20):position]['Close'] * 
                        df.iloc[max(0, position-20):position]['Volume']).mean()
            
            if current_price < self.config.min_price:
                return False, f"Price too low: ₹{current_price:.2f}", 0
            
            if avg_volume < self.config.min_volume:
                return False, f"Volume too low: {avg_volume:,.0f}", 0
            
            if avg_value < self.config.min_value_traded:
                return False, f"Value too low: ₹{avg_value:,.0f}", 0
            
            # 3. Trend filter
            weekly_trend = df.iloc[position]['Weekly_Trend_Up']
            monthly_trend = df.iloc[position]['Monthly_Trend_Up']
            
            if pd.isna(weekly_trend) or not weekly_trend:
                return False, "Weekly trend not bullish", 0
            
            if not pd.isna(monthly_trend) and not monthly_trend:
                return False, "Monthly trend not bullish", 0
            
            # 4. Calculate position size
            position_size = self.calculate_position_size(ticker, df, position)
            
            return True, "All filters passed", position_size
            
        except Exception as e:
            return False, f"Error: {str(e)}", 0
    
    def check_volatility_adjusted_entry(self, df: pd.DataFrame, position: int) -> bool:
        """Volatility-adjusted entry with volume surge and RSI"""
        try:
            # Volume surge check (50% above 20-day average)
            volume_surge = df.iloc[position]['Volume_Surge']
            if pd.isna(volume_surge) or volume_surge < self.config.min_volume_surge:
                logger.debug(f"Failed volume surge: {volume_surge:.2%}")
                return False
            
            # RSI check (55-75 range)
            rsi = df.iloc[position]['RSI']
            if pd.isna(rsi) or rsi < self.config.rsi_min or rsi > self.config.rsi_max:
                logger.debug(f"Failed RSI: {rsi:.2f}")
                return False
            
            # Price above highest EMA
            current_price = df.iloc[position]['Close']
            ema_values = [df.iloc[position][f'EMA_{period}'] for period in self.config.ema_periods]
            highest_ema = max(ema_values)
            
            if current_price <= highest_ema:
                logger.debug(f"Failed EMA breakout: {current_price:.2f} <= {highest_ema:.2f}")
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def check_broom_setup(self, df: pd.DataFrame, position: int) -> bool:
        """Check broom setup conditions"""
        if position < self.config.base_lookback_period:
            return False
        
        try:
            current_price = df.iloc[position]['Close']
            if pd.isna(current_price) or current_price <= 0:
                return False
            
            # EMA Broom Compression
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
                return False
            
            # Base Duration
            lookback_start = max(0, position - self.config.base_lookback_period)
            lookback_data = df.iloc[lookback_start:position]
            
            if len(lookback_data) < self.config.base_lookback_period:
                return False
            
            peak_position = lookback_data['High'].idxmax()
            peak_pos = df.index.get_loc(peak_position)
            days_since_peak = position - peak_pos
            
            if days_since_peak < self.config.base_duration_min or \
               days_since_peak > self.config.base_duration_max:
                return False
            
            # Box Consolidation
            recent_20 = df.iloc[position-19:position+1]
            box_height = (recent_20['High'].max() - recent_20['Low'].min()) / current_price
            
            if box_height >= self.config.box_consolidation_height:
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def calculate_atr_stop(self, df: pd.DataFrame, position: int, entry_price: float) -> float:
        """Calculate ATR-based trailing stop"""
        atr = df.iloc[position]['ATR']
        
        if pd.isna(atr) or atr <= 0:
            # Fallback to EMA-based stop
            ema_200 = df.iloc[position]['EMA_200']
            return ema_200 * 0.98
        
        # ATR trailing stop
        stop = entry_price - (self.config.atr_multiplier * atr)
        
        # Ensure stop is not too far
        ema_200 = df.iloc[position]['EMA_200']
        min_stop = ema_200 * 0.95
        
        return max(stop, min_stop)
    
    def run_backtest(self, df: pd.DataFrame, ticker: str = "") -> List[Dict]:
        """Run enhanced backtest"""
        trades = []
        position = None
        
        try:
            df = self.calculate_indicators(df)
            df = self.create_higher_timeframes(df)
            
            backtest_mask = df.index >= self.config.backtest_start
            backtest_df = df[backtest_mask].copy()
            
            if len(backtest_df) < 50:
                return trades
            
            for date_idx in backtest_df.index:
                position_idx = df.index.get_loc(date_idx)
                
                if position is None:
                    # Multi-layer filter
                    filter_ok, filter_reason, position_size = self.check_multi_layer_filter(
                        ticker, df, position_idx
                    )
                    
                    if not filter_ok:
                        continue
                    
                    # Broom setup
                    if not self.check_broom_setup(df, position_idx):
                        continue
                    
                    # Volatility-adjusted entry
                    if not self.check_volatility_adjusted_entry(df, position_idx):
                        continue
                    
                    # Enter position
                    entry_price = df.iloc[position_idx]['Close']
                    atr_stop = self.calculate_atr_stop(df, position_idx, entry_price)
                    
                    # Calculate take profit (measured move)
                    lookback_start = max(0, position_idx - self.config.base_lookback_period)
                    lookback_data = df.iloc[lookback_start:position_idx]
                    peak_price = lookback_data['High'].max()
                    peak_idx = lookback_data['High'].idxmax()
                    peak_pos = df.index.get_loc(peak_idx)
                    consolidation_data = df.iloc[peak_pos:position_idx+1]
                    lowest_price = consolidation_data['Low'].min()
                    base_depth = peak_price - lowest_price
                    take_profit = entry_price + (1.5 * base_depth)  # 1.5x measured move
                    
                    position = {
                        'entry_date': date_idx,
                        'entry_price': entry_price,
                        'stop_loss': atr_stop,
                        'take_profit': take_profit,
                        'base_depth': base_depth,
                        'position_size': position_size,
                        'sector': self.sector_classifier.get_sector(ticker),
                        'highest_price': entry_price
                    }
                    
                    logger.info(f"🚀 ENTRY: {ticker} ({position['sector']}) at {date_idx.date()} | "
                              f"Price: ₹{entry_price:.2f} | Size: {position_size:.1%} | "
                              f"Stop: ₹{atr_stop:.2f} | Target: ₹{take_profit:.2f}")
                    
                    self.trade_log_file.write(
                        f"ENTRY,{ticker},{position['sector']},{date_idx.date()},{entry_price:.2f},"
                        f"{position_size:.1%},{atr_stop:.2f},{take_profit:.2f}\n"
                    )
                    self.trade_log_file.flush()
                    
                else:
                    # Manage position
                    current_price = df.iloc[position_idx]['Close']
                    current_high = df.iloc[position_idx]['High']
                    current_low = df.iloc[position_idx]['Low']
                    current_atr = df.iloc[position_idx]['ATR']
                    
                    # Update highest price
                    if current_high > position['highest_price']:
                        position['highest_price'] = current_high
                    
                    # Update trailing stop (ATR-based)
                    if not pd.isna(current_atr) and current_atr > 0:
                        new_stop = position['highest_price'] - (self.config.atr_multiplier * current_atr)
                        if new_stop > position['stop_loss']:
                            position['stop_loss'] = new_stop
                    
                    days_held = (date_idx - position['entry_date']).days
                    
                    # Check stop loss
                    if current_low <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': date_idx,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': 'atr_stop',
                            'position_size': position['position_size'],
                            'sector': position['sector'],
                            'days_held': days_held
                        })
                        
                        logger.info(f"🛑 EXIT (ATR Stop): {ticker} | Return: {return_pct:.2f}% | Days: {days_held}")
                        
                        self.trade_log_file.write(
                            f"EXIT_STOP,{ticker},{date_idx.date()},{exit_price:.2f},{return_pct:.2f}%,{days_held}\n"
                        )
                        self.trade_log_file.flush()
                        
                        position = None
                        continue
                    
                    # Check take profit
                    if current_high >= position['take_profit']:
                        exit_price = position['take_profit']
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': date_idx,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': 'take_profit',
                            'position_size': position['position_size'],
                            'sector': position['sector'],
                            'days_held': days_held
                        })
                        
                        logger.info(f"🎯 EXIT (Target): {ticker} | Return: {return_pct:.2f}% | Days: {days_held}")
                        
                        self.trade_log_file.write(
                            f"EXIT_TP,{ticker},{date_idx.date()},{exit_price:.2f},{return_pct:.2f}%,{days_held}\n"
                        )
                        self.trade_log_file.flush()
                        
                        position = None
                        continue
                    
                    # Time-based exit (90 days)
                    if days_held >= self.config.max_holding_days:
                        exit_price = current_price
                        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': date_idx,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': 'time_exit',
                            'position_size': position['position_size'],
                            'sector': position['sector'],
                            'days_held': days_held
                        })
                        
                        logger.info(f"⏰ EXIT (Time): {ticker} | Return: {return_pct:.2f}% | Days: {days_held}")
                        
                        self.trade_log_file.write(
                            f"EXIT_TIME,{ticker},{date_idx.date()},{exit_price:.2f},{return_pct:.2f}%,{days_held}\n"
                        )
                        self.trade_log_file.flush()
                        
                        position = None
            
            # Close open position
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
                    'position_size': position['position_size'],
                    'sector': position['sector'],
                    'days_held': days_held
                })
            
            return trades
            
        except Exception as e:
            logger.debug(f"Error in backtest for {ticker}: {e}")
            return trades
    
    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate enhanced performance metrics"""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                'max_drawdown': 0, 'avg_return_per_trade': 0,
                'profit_factor': 0, 'avg_days_held': 0,
                'return_drawdown_ratio': 0, 'weighted_return': 0
            }
        
        try:
            trades_df = pd.DataFrame(trades)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['return_pct'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            
            # Weighted returns (position size adjusted)
            trades_df['weighted_return'] = trades_df['return_pct'] * trades_df['position_size']
            total_return = trades_df['weighted_return'].sum()
            
            # Equity curve
            cumulative_return = 100.0
            equity_curve = [100.0]
            
            for _, trade in trades_df.iterrows():
                trade_return = trade['return_pct'] * trade['position_size'] / 100
                cumulative_return *= (1 + trade_return)
                equity_curve.append(cumulative_return)
            
            total_return_pct = (cumulative_return - 100)
            
            equity_series = pd.Series(equity_curve)
            max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
            
            avg_return_per_trade = trades_df['return_pct'].mean()
            
            # Profit factor (weighted)
            gross_profit = trades_df[trades_df['weighted_return'] > 0]['weighted_return'].sum()
            gross_loss = abs(trades_df[trades_df['weighted_return'] < 0]['weighted_return'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_days_held = trades_df['days_held'].mean()
            
            # Return to drawdown ratio
            return_drawdown_ratio = total_return_pct / max_drawdown if max_drawdown > 0 else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return_pct,
                'max_drawdown': max_drawdown,
                'avg_return_per_trade': avg_return_per_trade,
                'profit_factor': profit_factor,
                'avg_days_held': avg_days_held,
                'return_drawdown_ratio': return_drawdown_ratio,
                'weighted_return': total_return
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                'max_drawdown': 0, 'avg_return_per_trade': 0,
                'profit_factor': 0, 'avg_days_held': 0,
                'return_drawdown_ratio': 0, 'weighted_return': 0
            }
    
    def run_universe_backtest(self, ticker_list: List[str]) -> List[Dict]:
        """Run universe backtest"""
        logger.info("=" * 80)
        logger.info("STARTING ENHANCED UNIVERSE BACKTEST")
        logger.info(f"Universe size: {len(ticker_list)} stocks")
        logger.info("=" * 80)
        
        start_idx = self.checkpoint.get('last_processed_index', 0)
        
        for batch_start in range(start_idx, len(ticker_list), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(ticker_list))
            batch = ticker_list[batch_start:batch_end]
            
            logger.info(f"\nProcessing batch {batch_start//self.config.batch_size + 1}: "
                       f"stocks {batch_start+1}-{batch_end}")
            
            for i, ticker in enumerate(batch):
                global_idx = batch_start + i
                
                try:
                    # Skip excluded sectors early
                    if self.sector_classifier.is_excluded(ticker):
                        logger.info(f"[{global_idx+1}/{len(ticker_list)}] ⏭️ Skipping {ticker} "
                                  f"({self.sector_classifier.get_sector(ticker)})")
                        continue
                    
                    logger.info(f"[{global_idx+1}/{len(ticker_list)}] Processing {ticker}...")
                    
                    df = self.download_data(ticker)
                    
                    if df is None or len(df) < 300:
                        logger.warning(f"  ✗ Insufficient data")
                        continue
                    
                    trades = self.run_backtest(df, ticker)
                    
                    metrics = self.calculate_performance_metrics(trades)
                    metrics['ticker'] = ticker
                    metrics['sector'] = self.sector_classifier.get_sector(ticker)
                    metrics['processed'] = True
                    
                    # Apply historical performance filters
                    if metrics['total_trades'] > 0:
                        if metrics['win_rate'] >= self.config.min_win_rate * 100 and \
                           metrics['profit_factor'] >= self.config.min_profit_factor and \
                           metrics['return_drawdown_ratio'] >= self.config.min_return_drawdown:
                            metrics['qualifies'] = True
                        else:
                            metrics['qualifies'] = False
                    else:
                        metrics['qualifies'] = False
                    
                    self.results.append(metrics)
                    self.stocks_processed += 1
                    self.total_trades += len(trades)
                    
                    if trades:
                        logger.info(f"  ✓ {metrics['total_trades']} trades, "
                                  f"Win: {metrics['win_rate']:.1f}%, "
                                  f"PF: {metrics['profit_factor']:.2f}, "
                                  f"RDD: {metrics['return_drawdown_ratio']:.2f}")
                    
                    self.save_checkpoint_state(global_idx + 1)
                    
                    del df, trades
                    gc.collect()
                    
                    time.sleep(self.config.request_delay)
                    
                except Exception as e:
                    logger.error(f"  ✗ Error: {str(e)}")
                    self.stocks_failed += 1
            
            if batch_end < len(ticker_list):
                time.sleep(self.config.batch_delay)
        
        elapsed_time = time.time() - self.start_time
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETED")
        logger.info(f"Stocks processed: {self.stocks_processed}")
        logger.info(f"Total trades: {self.total_trades}")
        logger.info(f"Time elapsed: {elapsed_time/60:.2f} minutes")
        logger.info("=" * 80)
        
        return self.results
    
    def export_results(self, filename: str = 'enhanced_broom_breakout_results.csv') -> pd.DataFrame:
        """Export results"""
        results_df = pd.DataFrame(self.results)
        
        if results_df.empty:
            logger.error("No results")
            return pd.DataFrame()
        
        results_df.to_csv(filename, index=False)
        
        # Summary
        qualified = results_df[results_df.get('qualifies', False) == True]
        logger.info("\n=== QUALIFIED STOCKS ===")
        logger.info(f"Total qualified: {len(qualified)}")
        
        if len(qualified) > 0:
            logger.info(f"Average win rate: {qualified['win_rate'].mean():.1f}%")
            logger.info(f"Average profit factor: {qualified['profit_factor'].mean():.2f}")
            logger.info(f"Average RDD: {qualified['return_drawdown_ratio'].mean():.2f}")
            
            top = qualified.nlargest(10, 'total_return')[
                ['ticker', 'sector', 'total_return', 'win_rate', 'profit_factor']
            ]
            logger.info("\n=== TOP QUALIFIED PERFORMERS ===")
            for _, row in top.iterrows():
                logger.info(f"  {row['ticker']} ({row['sector']}): {row['total_return']:.1f}% return, "
                          f"{row['win_rate']:.1f}% win rate, PF: {row['profit_factor']:.2f}")
        
        return results_df

# ==================== UNIVERSE ====================

def get_enhanced_universe() -> List[str]:
    """Get enhanced universe of high-conviction stocks"""
    
    # Prioritized list - high conviction first
    enhanced_universe = [
        # High Conviction Multi-baggers
        'DIVISLAB.NS', 'TRENT.NS', 'CUMMINSIND.NS', 'TITAN.NS', 'ASIANPAINT.NS',
        'PIDILITIND.NS', 'HAVELLS.NS', 'BAJFINANCE.NS', 'DIXON.NS', 'ASTRAL.NS',
        
        # Pharma Leaders
        'SUNPHARMA.NS', 'CIPLA.NS', 'DRREDDY.NS', 'LUPIN.NS', 'AUROPHARMA.NS',
        'BIOCON.NS', 'GLENMARK.NS', 'ALKEM.NS', 'TORNTPHARM.NS', 'ZYDUSLIFE.NS',
        
        # Banking Leaders
        'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'INDUSINDBK.NS',
        'FEDERALBNK.NS', 'AUBANK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS',
        
        # Auto Leaders
        'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS',
        'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BHARATFORG.NS',
        
        # IT Leaders
        'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS',
        'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS',
        
        # FMCG Leaders
        'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS',
        'MARICO.NS', 'GODREJCP.NS', 'TATACONSUM.NS',
        
        # Consumer Discretionary
        'TRENT.NS', 'DMART.NS', 'TITAN.NS', 'VOLTAS.NS', 'CROMPTON.NS',
        
        # Others (Quality)
        'SIEMENS.NS', 'ABB.NS', 'KEI.NS', 'POLYCAB.NS', 'SUPREMEIND.NS',
    ]
    
    # Remove excluded sectors
    sector_classifier = SectorClassifier()
    enhanced_universe = [t for t in enhanced_universe if not sector_classifier.is_excluded(t)]
    
    # Remove duplicates
    seen = set()
    enhanced_universe = [x for x in enhanced_universe if not (x in seen or seen.add(x))]
    
    logger.info(f"Enhanced universe: {len(enhanced_universe)} stocks")
    return enhanced_universe

# ==================== MAIN ====================

def main():
    """Main execution"""
    try:
        logger.info("=" * 80)
        logger.info("ENHANCED BROOM BREAKOUT STRATEGY")
        logger.info("Multi-Layered Selection & Position Sizing")
        logger.info("=" * 80)
        
        tickers = get_enhanced_universe()
        
        engine = EnhancedBroomBreakoutBacktest(config)
        
        results = engine.run_universe_backtest(tickers)
        
        results_df = engine.export_results()
        
        logger.info("\n✅ BACKTEST COMPLETED SUCCESSFULLY")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
