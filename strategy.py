"""
Institutional Moving Average "Broom" Breakout Strategy Backtesting Engine
For Nifty 500 Universe - Version with Alternative Data Sources
"""

import pandas as pd
import numpy as np
import warnings
import gc
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataFetcher:
    """Multi-source data fetcher with fallback mechanisms"""
    
    def __init__(self):
        self.session = self._create_session()
        self.rate_limit_hits = 0
        self.consecutive_failures = 0
        
    def _create_session(self):
        """Create session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers to mimic browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json,text/plain,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://finance.yahoo.com',
            'Referer': 'https://finance.yahoo.com/',
        })
        return session
    
    def fetch_from_yfinance(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance with enhanced error handling"""
        try:
            import yfinance as yf
            
            # Create ticker object with custom session
            stock = yf.Ticker(ticker, session=self.session)
            
            # Try to get history
            df = stock.history(
                start=start_date,
                end=datetime.now().strftime('%Y-%m-%d'),
                auto_adjust=False,
                timeout=15
            )
            
            if df is not None and not df.empty:
                # Ensure required columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    return df[required_cols].copy()
            
            return None
            
        except Exception as e:
            logger.debug(f"yfinance failed for {ticker}: {str(e)}")
            return None
    
    def fetch_from_stooq(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from Stooq (free alternative)"""
        try:
            # Convert ticker format for Stooq
            # Stooq uses different format for Indian stocks
            if ticker.endswith('.NS'):
                stooq_ticker = ticker.replace('.NS', '.NSE')
            elif ticker.endswith('.BO'):
                stooq_ticker = ticker.replace('.BO', '.BSE')
            else:
                stooq_ticker = ticker + '.NSE'
            
            # Stooq URL format
            url = f"https://stooq.com/q/d/l/?s={stooq_ticker.lower()}&d1={start_date.replace('-', '')}&d2={datetime.now().strftime('%Y%m%d')}&i=d"
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                df = pd.read_csv(pd.StringIO(response.text))
                
                if not df.empty and 'Close' in df.columns:
                    # Convert to standard format
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    # Ensure required columns
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required_cols):
                        # Convert volume to int
                        df['Volume'] = df['Volume'].astype(float)
                        return df[required_cols]
            
            return None
            
        except Exception as e:
            logger.debug(f"Stooq failed for {ticker}: {str(e)}")
            return None
    
    def fetch_from_alpha_vantage(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch from Alpha Vantage (requires API key)"""
        # Note: Alpha Vantage requires API key
        # You can get free key from https://www.alphavantage.co/support/#api-key
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        
        try:
            # Convert ticker format
            if ticker.endswith('.NS'):
                av_ticker = ticker.replace('.NS', '.BSE')  # Alpha Vantage uses BSE for Indian stocks
            else:
                av_ticker = ticker
            
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_ticker}&outputsize=full&apikey={api_key}"
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)
                    
                    # Convert columns
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df = df.astype(float)
                    
                    # Filter to start date
                    df = df[df.index >= start_date]
                    
                    return df
            
            return None
            
        except Exception as e:
            logger.debug(f"Alpha Vantage failed for {ticker}: {str(e)}")
            return None
    
    def fetch_from_nse_india(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch from NSE India directly"""
        try:
            # NSE symbol (remove .NS suffix)
            nse_symbol = ticker.replace('.NS', '')
            
            # NSE API endpoint
            url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={nse_symbol}&series=[%22EQ%22]&from={start_date}&to={datetime.now().strftime('%d-%m-%Y')}"
            
            # NSE requires cookies and specific headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json,text/plain,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nseindia.com/',
            }
            
            # First get cookies
            session = requests.Session()
            session.headers.update(headers)
            session.get('https://www.nseindia.com/', timeout=15)
            
            # Then fetch data
            response = session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    df['Date'] = pd.to_datetime(df['CH_TIMESTAMP'])
                    df.set_index('Date', inplace=True)
                    
                    # Map columns
                    column_mapping = {
                        'CH_OPENING_PRICE': 'Open',
                        'CH_TRADE_HIGH_PRICE': 'High',
                        'CH_TRADE_LOW_PRICE': 'Low',
                        'CH_CLOSING_PRICE': 'Close',
                        'CH_TOT_TRADED_QTY': 'Volume'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                    df.sort_index(inplace=True)
                    
                    return df
            
            return None
            
        except Exception as e:
            logger.debug(f"NSE India failed for {ticker}: {str(e)}")
            return None
    
    def fetch_with_fallback(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Fetch data with multiple source fallback"""
        
        # Check if we're being rate limited
        if self.rate_limit_hits >= 3:
            logger.warning("Multiple rate limit hits detected. Waiting 60 seconds...")
            time.sleep(60)
            self.rate_limit_hits = 0
        
        # Try each data source in order
        data_sources = [
            ('yfinance', self.fetch_from_yfinance),
            ('stooq', self.fetch_from_stooq),
            ('nse_india', self.fetch_from_nse_india),
        ]
        
        for source_name, fetch_func in data_sources:
            try:
                logger.info(f"Trying {source_name} for {ticker}")
                df = fetch_func(ticker, start_date)
                
                if df is not None and not df.empty and len(df) > 100:
                    logger.info(f"Successfully fetched {len(df)} rows from {source_name}")
                    self.consecutive_failures = 0
                    return df
                else:
                    logger.warning(f"{source_name} returned insufficient data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error with {source_name} for {ticker}: {str(e)}")
                
                if "429" in str(e):
                    self.rate_limit_hits += 1
        
        # All sources failed
        self.consecutive_failures += 1
        
        # If too many consecutive failures, wait longer
        if self.consecutive_failures >= 5:
            logger.warning(f"{self.consecutive_failures} consecutive failures. Waiting 120 seconds...")
            time.sleep(120)
            self.consecutive_failures = 0
        
        return None


class BroomBreakoutBacktest:
    """Main backtesting engine"""
    
    def __init__(self, data_dir='data_cache'):
        # Strategy Parameters
        self.ema_periods = [20, 50, 100, 200]
        self.weekly_ema_period = 200
        self.monthly_ema_period = 200
        
        # Broom Setup Parameters
        self.broom_compression_threshold = 0.08
        self.base_lookback_period = 200
        self.base_duration_min = 63
        self.base_duration_max = 147
        self.box_consolidation_height = 0.15
        self.prior_trend_exhaustion_limit = 0.60
        
        # Execution Parameters
        self.volume_poc_bins = 10
        self.poc_lookback = 20
        self.volume_threshold_multiplier = 1.5
        self.volume_ma_period = 50
        self.stop_loss_buffer = 0.015
        self.measured_move_multiplier = 2
        
        # Backtest Period
        self.backtest_start = '2018-01-01'
        self.data_start = '2000-01-01'
        
        # Data cache
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Results Storage
        self.results = []
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher()
    
    def save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save data to CSV cache"""
        cache_file = os.path.join(self.data_dir, f"{ticker.replace('.', '_')}.csv")
        try:
            df.to_csv(cache_file)
            logger.info(f"Cached data for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to cache data for {ticker}: {e}")
    
    def load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        cache_file = os.path.join(self.data_dir, f"{ticker.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Check if cache is recent (less than 3 days old)
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 3 * 24 * 3600:
                    logger.info(f"Loaded cached data for {ticker}")
                    return df
                else:
                    logger.info(f"Cache expired for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")
        return None
    
    def download_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download historical data with caching"""
        # Try cache first
        df = self.load_from_cache(ticker)
        if df is not None:
            return df
        
        # Fetch from data sources
        df = self.data_fetcher.fetch_with_fallback(ticker, self.data_start)
        
        # Cache if successful
        if df is not None and not df.empty:
            self.save_to_cache(ticker, df)
        
        return df
    
    def calculate_emas(self, df):
        """Calculate EMAs"""
        for period in self.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df
    
    def create_higher_timeframes(self, df):
        """Create weekly and monthly timeframes"""
        # Weekly
        weekly_df = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        weekly_df[f'EMA_{self.weekly_ema_period}_Weekly'] = weekly_df['Close'].ewm(
            span=self.weekly_ema_period, adjust=False
        ).mean()
        
        # Monthly
        monthly_df = df.resample('ME').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        monthly_df[f'EMA_{self.monthly_ema_period}_Monthly'] = monthly_df['Close'].ewm(
            span=self.monthly_ema_period, adjust=False
        ).mean()
        
        # Map back to daily
        df['Weekly_200_EMA'] = weekly_df[f'EMA_{self.weekly_ema_period}_Weekly'].reindex(
            df.index, method='ffill'
        )
        df['Monthly_200_EMA'] = monthly_df[f'EMA_{self.monthly_ema_period}_Monthly'].reindex(
            df.index, method='ffill'
        )
        
        return df
    
    def check_broom_setup(self, df, idx):
        """Check broom setup conditions"""
        if idx < self.base_lookback_period:
            return False
        
        current_price = df.loc[idx, 'Close']
        if pd.isna(current_price) or current_price <= 0:
            return False
        
        # 1. Macro Trend Filter
        if pd.isna(df.loc[idx, 'Weekly_200_EMA']):
            return False
        if current_price <= df.loc[idx, 'Weekly_200_EMA']:
            return False
        
        if not pd.isna(df.loc[idx, 'Monthly_200_EMA']):
            if current_price <= df.loc[idx, 'Monthly_200_EMA']:
                return False
        
        # 2. EMA Broom Compression
        ema_values = []
        for period in self.ema_periods:
            ema_val = df.loc[idx, f'EMA_{period}']
            if pd.isna(ema_val):
                return False
            ema_values.append(ema_val)
        
        ema_high = max(ema_values)
        ema_low = min(ema_values)
        ema_spread = (ema_high - ema_low) / current_price
        
        if ema_spread >= self.broom_compression_threshold:
            return False
        
        # 3. Base Duration Check
        lookback_data = df.iloc[max(0, idx - self.base_lookback_period):idx]
        if len(lookback_data) < self.base_lookback_period:
            return False
        
        peak_idx = lookback_data['High'].idxmax()
        peak_price = lookback_data['High'].max()
        peak_position = df.index.get_loc(peak_idx)
        days_since_peak = idx - peak_position
        
        if days_since_peak < self.base_duration_min or days_since_peak > self.base_duration_max:
            return False
        
        # 4. Flat Box Consolidation
        recent_20 = df.iloc[idx-19:idx+1]
        box_height = (recent_20['High'].max() - recent_20['Low'].min()) / current_price
        
        if box_height >= self.box_consolidation_height:
            return False
        
        # 5. Prior Trend Exhaustion
        pre_peak_data = df.iloc[max(0, peak_position - self.base_lookback_period):peak_position+1]
        if len(pre_peak_data) > 0:
            lowest_low = pre_peak_data['Low'].min()
            run_up = (peak_price - lowest_low) / lowest_low
            
            if run_up > self.prior_trend_exhaustion_limit:
                return False
        
        return True
    
    def calculate_volume_profile_poc(self, df, idx):
        """Calculate Volume Profile POC"""
        if idx < self.poc_lookback:
            return None
        
        recent_data = df.iloc[idx - self.poc_lookback + 1:idx + 1]
        price_range = recent_data['High'].max() - recent_data['Low'].min()
        if price_range == 0:
            return None
        
        bins = np.linspace(recent_data['Low'].min(), recent_data['High'].max(), self.volume_poc_bins + 1)
        volume_by_bin = np.zeros(self.volume_poc_bins)
        
        for i in range(len(recent_data)):
            row = recent_data.iloc[i]
            price_low = row['Low']
            price_high = row['High']
            volume = row['Volume']
            
            if price_high > price_low:
                for j in range(self.volume_poc_bins):
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
    
    def check_entry_signal(self, df, idx):
        """Check entry trigger conditions"""
        current_price = df.loc[idx, 'Close']
        
        ema_values = [df.loc[idx, f'EMA_{period}'] for period in self.ema_periods]
        highest_ema = max(ema_values)
        
        poc_price = self.calculate_volume_profile_poc(df, idx)
        if poc_price is None:
            return False
        
        if current_price <= highest_ema:
            return False
        if current_price <= poc_price:
            return False
        
        if idx < self.volume_ma_period:
            return False
        
        volume_ma = df.iloc[idx - self.volume_ma_period:idx]['Volume'].mean()
        current_volume = df.loc[idx, 'Volume']
        
        if current_volume <= self.volume_threshold_multiplier * volume_ma:
            return False
        
        return True
    
    def run_backtest(self, df):
        """Run backtest on single stock"""
        trades = []
        position = None
        
        df = self.calculate_emas(df)
        df = self.create_higher_timeframes(df)
        
        backtest_df = df[df.index >= self.backtest_start].copy()
        
        if len(backtest_df) < 50:
            return trades
        
        for idx in backtest_df.index:
            position_idx = df.index.get_loc(idx)
            
            if position is None:
                if self.check_broom_setup(df, position_idx):
                    if self.check_entry_signal(df, position_idx):
                        entry_price = df.loc[idx, 'Close']
                        stop_loss = df.loc[idx, 'EMA_200'] * (1 - self.stop_loss_buffer)
                        
                        # Calculate take profit
                        lookback_data = df.iloc[max(0, position_idx - self.base_lookback_period):position_idx]
                        peak_price = lookback_data['High'].max()
                        peak_pos = df.index.get_loc(lookback_data['High'].idxmax())
                        consolidation_data = df.iloc[peak_pos:position_idx+1]
                        lowest_price = consolidation_data['Low'].min()
                        base_depth = peak_price - lowest_price
                        take_profit = entry_price + (self.measured_move_multiplier * base_depth)
                        
                        position = {
                            'entry_date': idx,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'base_depth': base_depth
                        }
            else:
                current_price = df.loc[idx, 'Close']
                current_high = df.loc[idx, 'High']
                current_low = df.loc[idx, 'Low']
                ema_200 = df.loc[idx, 'EMA_200']
                
                # Update trailing stop
                new_stop = ema_200 * (1 - self.stop_loss_buffer)
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
                
                # Check stop loss
                if current_low <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': idx,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'return_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                        'exit_reason': 'stop_loss'
                    })
                    position = None
                    continue
                
                # Check take profit
                if current_high >= position['take_profit']:
                    exit_price = position['take_profit']
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': idx,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'return_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                        'exit_reason': 'take_profit'
                    })
                    position = None
        
        # Close open position
        if position is not None:
            last_idx = backtest_df.index[-1]
            last_price = backtest_df.loc[last_idx, 'Close']
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': last_idx,
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'return_pct': (last_price - position['entry_price']) / position['entry_price'] * 100,
                'exit_reason': 'end_of_period'
            })
        
        return trades
    
    def calculate_performance_metrics(self, trades):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                'max_drawdown': 0, 'avg_return_per_trade': 0, 'profit_factor': 0
            }
        
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        # Calculate returns
        cumulative_return = 100.0
        equity_curve = [100.0]
        
        for _, trade in trades_df.iterrows():
            trade_return = trade['return_pct'] / 100
            cumulative_return *= (1 + trade_return)
            equity_curve.append(cumulative_return)
        
        total_return = (cumulative_return - 100)
        
        # Calculate drawdown
        equity_series = pd.Series(equity_curve)
        drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
        
        avg_return_per_trade = trades_df['return_pct'].mean()
        
        # Profit factor
        gross_profit = trades_df[trades_df['return_pct'] > 0]['return_pct'].sum()
        gross_loss = abs(trades_df[trades_df['return_pct'] < 0]['return_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'avg_return_per_trade': avg_return_per_trade,
            'profit_factor': profit_factor
        }
    
    def run_universe_backtest(self, ticker_list, start_idx=0, end_idx=None):
        """Run backtest for universe"""
        if end_idx is None:
            end_idx = len(ticker_list)
        
        print(f"Starting backtest for {end_idx - start_idx} stocks...")
        print(f"Backtest period: {self.backtest_start} to present")
        print("-" * 80)
        
        start_time = time.time()
        processed_count = 0
        
        for i in range(start_idx, min(end_idx, len(ticker_list))):
            ticker = ticker_list[i]
            
            try:
                print(f"[{i+1}/{end_idx}] Processing {ticker}...")
                
                # Download data
                df = self.download_data(ticker)
                
                if df is None or len(df) < 300:
                    print(f"  Skipping {ticker}: insufficient data")
                    self.results.append({
                        'ticker': ticker,
                        'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                        'max_drawdown': 0, 'avg_return_per_trade': 0,
                        'profit_factor': 0, 'processed': False,
                        'reason': 'insufficient_data'
                    })
                    continue
                
                # Run backtest
                trades = self.run_backtest(df)
                
                # Calculate metrics
                metrics = self.calculate_performance_metrics(trades)
                metrics['ticker'] = ticker
                metrics['processed'] = True
                metrics['reason'] = 'success'
                
                self.results.append(metrics)
                processed_count += 1
                
                # Print results
                if trades:
                    print(f"  Trades: {metrics['total_trades']}, "
                          f"Win Rate: {metrics['win_rate']:.1f}%, "
                          f"Return: {metrics['total_return']:.1f}%")
                else:
                    print(f"  No trades generated")
                
                # Clear memory
                del df, trades
                gc.collect()
                
                # Add delay
                time.sleep(2)  # 2 seconds between stocks
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                self.results.append({
                    'ticker': ticker,
                    'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                    'max_drawdown': 0, 'avg_return_per_trade': 0,
                    'profit_factor': 0, 'processed': False,
                    'reason': f'error: {str(e)}'
                })
        
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"Backtest completed: {processed_count}/{end_idx - start_idx} stocks processed")
        print(f"Time elapsed: {elapsed_time/60:.2f} minutes")
        
        return self.results
    
    def export_results(self, filename='nifty500_broom_breakout_results.csv'):
        """Export results"""
        results_df = pd.DataFrame(self.results)
        
        if results_df.empty:
            logger.error("No results to export")
            return None
        
        results_df.to_csv(filename, index=False)
        print(f"\nResults exported to {filename}")
        
        # Summary statistics
        processed_df = results_df[results_df['processed'] == True]
        if len(processed_df) > 0:
            print("\n=== SUMMARY STATISTICS ===")
            print(f"Stocks processed: {len(processed_df)}")
            print(f"Stocks with trades: {len(processed_df[processed_df['total_trades'] > 0])}")
            print(f"Total trades: {processed_df['total_trades'].sum()}")
            print(f"Average win rate: {processed_df['win_rate'].mean():.2f}%")
            print(f"Average return: {processed_df['total_return'].mean():.2f}%")
            
            # Top performers
            stocks_with_trades = processed_df[processed_df['total_trades'] > 0]
            if len(stocks_with_trades) > 0:
                top_performers = stocks_with_trades.nlargest(10, 'total_return')[
                    ['ticker', 'total_return', 'win_rate', 'total_trades']
                ]
                print("\n=== TOP 10 PERFORMERS ===")
                print(top_performers.to_string(index=False))
        
        return results_df


def get_nifty500_tickers():
    """Get sample Nifty 500 tickers"""
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
    ]


def main():
    """Main execution"""
    print("=" * 80)
    print("INSTITUTIONAL MOVING AVERAGE 'BROOM' BREAKOUT STRATEGY")
    print("Nifty 500 Universe Backtesting Engine - Multi-Source Version")
    print("=" * 80)
    
    # Get universe
    tickers = get_nifty500_tickers()
    print(f"\nUniverse size: {len(tickers)} stocks")
    
    # Initialize backtest engine
    engine = BroomBreakoutBacktest()
    
    # Run backtest
    results = engine.run_universe_backtest(tickers)
    
    # Export results
    results_df = engine.export_results()
    
    print("\nBacktest completed!")
    return results_df


if __name__ == "__main__":
    main()
