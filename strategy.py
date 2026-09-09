"""
Institutional Moving Average "Broom" Breakout Strategy Backtesting Engine
For Nifty 500 Universe - Enhanced Version with Robust Error Handling
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import gc
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from typing import Optional, Dict, List
import json
import os

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

class RobustDataFetcher:
    """Enhanced data fetching with retry logic and error handling"""
    
    def __init__(self, max_retries=3, retry_delay=2):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = self._create_session()
        
    def _create_session(self):
        """Create a session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def fetch_with_retry(self, ticker: str, start_date: str, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch data with exponential backoff retry"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        for attempt in range(self.max_retries):
            try:
                # Add delay between attempts
                if attempt > 0:
                    wait_time = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(f"Retry {attempt} for {ticker} after {wait_time} seconds")
                    time.sleep(wait_time)
                
                # Use yfinance with custom session
                stock = yf.Ticker(ticker, session=self.session)
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    timeout=30
                )
                
                if df is None or df.empty:
                    logger.warning(f"No data returned for {ticker} on attempt {attempt + 1}")
                    continue
                
                # Validate data
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_columns):
                    logger.warning(f"Missing columns for {ticker}")
                    continue
                
                # Clean data
                df = df[required_columns].copy()
                df = df.dropna()
                
                if len(df) < 100:  # Minimum data requirement
                    logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
                    return None
                
                logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
                return df
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    logger.warning(f"Rate limited for {ticker} (attempt {attempt + 1})")
                elif "404" in error_str:
                    logger.error(f"Ticker {ticker} not found")
                    return None
                else:
                    logger.error(f"Error fetching {ticker} (attempt {attempt + 1}): {error_str}")
                
                if attempt == self.max_retries - 1:
                    return None
        
        return None
    
    def fetch_with_alternative_sources(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """Try alternative data sources if Yahoo Finance fails"""
        # Try different Yahoo Finance ticker formats
        alternative_formats = []
        
        if ticker.endswith('.NS'):
            # Try without suffix
            alternative_formats.append(ticker.replace('.NS', ''))
            # Try with .BO suffix
            alternative_formats.append(ticker.replace('.NS', '.BO'))
        elif ticker.endswith('.BO'):
            alternative_formats.append(ticker.replace('.BO', '.NS'))
            alternative_formats.append(ticker.replace('.BO', ''))
        else:
            alternative_formats.append(f"{ticker}.NS")
            alternative_formats.append(f"{ticker}.BO")
        
        for alt_ticker in alternative_formats:
            logger.info(f"Trying alternative ticker format: {alt_ticker}")
            df = self.fetch_with_retry(alt_ticker, start_date)
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched data using {alt_ticker}")
                return df
        
        return None


class BroomBreakoutBacktest:
    def __init__(self, data_dir='data_cache'):
        # Strategy Parameters
        self.ema_periods = [20, 50, 100, 200]
        self.weekly_ema_period = 200
        self.monthly_ema_period = 200
        
        # Broom Setup Parameters
        self.broom_compression_threshold = 0.08  # 8% spread
        self.base_lookback_period = 200  # Trading days
        self.base_duration_min = 63  # Minimum 3 months
        self.base_duration_max = 147  # Maximum 7 months
        self.box_consolidation_height = 0.15  # 15% height
        self.prior_trend_exhaustion_limit = 0.60  # 60% max run-up
        
        # Execution Parameters
        self.volume_poc_bins = 10
        self.poc_lookback = 20  # Trading days
        self.volume_threshold_multiplier = 1.5
        self.volume_ma_period = 50
        self.stop_loss_buffer = 0.015  # 1.5% below 200 EMA
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
        self.data_fetcher = RobustDataFetcher()
    
    def save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save data to local cache"""
        cache_file = os.path.join(self.data_dir, f"{ticker.replace('.', '_')}.parquet")
        try:
            df.to_parquet(cache_file)
            logger.info(f"Cached data for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to cache data for {ticker}: {e}")
    
    def load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data from local cache"""
        cache_file = os.path.join(self.data_dir, f"{ticker.replace('.', '_')}.parquet")
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                # Check if cache is recent (less than 7 days old)
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 7 * 24 * 3600:  # 7 days
                    logger.info(f"Loaded cached data for {ticker}")
                    return df
                else:
                    logger.info(f"Cache expired for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")
        return None
    
    def download_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download historical data with caching"""
        # Try loading from cache first
        df = self.load_from_cache(ticker)
        if df is not None:
            return df
        
        # Fetch from Yahoo Finance
        df = self.data_fetcher.fetch_with_retry(ticker, self.data_start)
        
        # Try alternative sources if primary fails
        if df is None:
            logger.warning(f"Primary fetch failed for {ticker}, trying alternatives")
            df = self.data_fetcher.fetch_with_alternative_sources(ticker, self.data_start)
        
        # Cache the data if successful
        if df is not None and not df.empty:
            self.save_to_cache(ticker, df)
        
        return df
    
    def calculate_emas(self, df):
        """Calculate EMAs at different timeframes"""
        # Daily EMAs
        for period in self.ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        return df
    
    def create_higher_timeframes(self, df):
        """Create weekly and monthly dataframes and calculate their EMAs"""
        # Weekly resampling
        weekly_df = df.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        weekly_df[f'EMA_{self.weekly_ema_period}_Weekly'] = weekly_df['Close'].ewm(
            span=self.weekly_ema_period, adjust=False
        ).mean()
        
        # Monthly resampling
        monthly_df = df.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        monthly_df[f'EMA_{self.monthly_ema_period}_Monthly'] = monthly_df['Close'].ewm(
            span=self.monthly_ema_period, adjust=False
        ).mean()
        
        # Map back to daily using forward fill
        df['Weekly_200_EMA'] = weekly_df[f'EMA_{self.weekly_ema_period}_Weekly'].reindex(
            df.index, method='ffill'
        )
        df['Monthly_200_EMA'] = monthly_df[f'EMA_{self.monthly_ema_period}_Monthly'].reindex(
            df.index, method='ffill'
        )
        
        return df
    
    def check_broom_setup(self, df, idx):
        """Check if all broom setup conditions are met"""
        if idx < self.base_lookback_period:
            return False
        
        current_price = df.loc[idx, 'Close']
        if pd.isna(current_price) or current_price <= 0:
            return False
        
        # 1. Macro Trend Filter
        # Daily Close above Weekly 200 EMA
        if pd.isna(df.loc[idx, 'Weekly_200_EMA']):
            return False
        if current_price <= df.loc[idx, 'Weekly_200_EMA']:
            return False
        
        # Daily Close above Monthly 200 EMA (if available)
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
        
        # 4. Flat Box Consolidation (last 20 days)
        recent_20 = df.iloc[idx-19:idx+1]
        box_height = (recent_20['High'].max() - recent_20['Low'].min()) / current_price
        
        if box_height >= self.box_consolidation_height:
            return False
        
        # 5. Prior Trend Exhaustion Limit
        # Find lowest low before peak
        pre_peak_data = df.iloc[max(0, peak_position - self.base_lookback_period):peak_position+1]
        if len(pre_peak_data) > 0:
            lowest_low = pre_peak_data['Low'].min()
            run_up = (peak_price - lowest_low) / lowest_low
            
            if run_up > self.prior_trend_exhaustion_limit:
                return False
        
        return True
    
    def calculate_volume_profile_poc(self, df, idx):
        """Calculate Volume Profile Point of Control (POC)"""
        if idx < self.poc_lookback:
            return None
        
        # Get last 20 days of data
        recent_data = df.iloc[idx - self.poc_lookback + 1:idx + 1]
        
        # Create price bins
        price_range = recent_data['High'].max() - recent_data['Low'].min()
        if price_range == 0:
            return None
            
        bin_size = price_range / self.volume_poc_bins
        bins = np.linspace(recent_data['Low'].min(), recent_data['High'].max(), self.volume_poc_bins + 1)
        
        # Calculate volume in each bin
        volume_by_bin = np.zeros(self.volume_poc_bins)
        
        for i in range(len(recent_data)):
            row = recent_data.iloc[i]
            price_low = row['Low']
            price_high = row['High']
            volume = row['Volume']
            
            # Distribute volume across price range
            if price_high > price_low:
                for j in range(self.volume_poc_bins):
                    bin_low = bins[j]
                    bin_high = bins[j + 1]
                    
                    # Calculate overlap
                    overlap_low = max(price_low, bin_low)
                    overlap_high = min(price_high, bin_high)
                    
                    if overlap_high > overlap_low:
                        overlap_percentage = (overlap_high - overlap_low) / (price_high - price_low)
                        volume_by_bin[j] += volume * overlap_percentage
        
        # Find POC (bin with highest volume)
        poc_bin_idx = np.argmax(volume_by_bin)
        poc_price = (bins[poc_bin_idx] + bins[poc_bin_idx + 1]) / 2
        
        return poc_price
    
    def check_entry_signal(self, df, idx):
        """Check for entry trigger conditions"""
        current_price = df.loc[idx, 'Close']
        
        # Get highest EMA
        ema_values = [df.loc[idx, f'EMA_{period}'] for period in self.ema_periods]
        highest_ema = max(ema_values)
        
        # Calculate Volume Profile POC
        poc_price = self.calculate_volume_profile_poc(df, idx)
        if poc_price is None:
            return False
        
        # Check breakout conditions
        if current_price <= highest_ema:
            return False
        if current_price <= poc_price:
            return False
        
        # Volume confirmation
        if idx < self.volume_ma_period:
            return False
        
        volume_ma = df.iloc[idx - self.volume_ma_period:idx]['Volume'].mean()
        current_volume = df.loc[idx, 'Volume']
        
        if current_volume <= self.volume_threshold_multiplier * volume_ma:
            return False
        
        return True
    
    def calculate_stop_loss(self, df, idx, entry_price):
        """Calculate initial and trailing stop loss"""
        ema_200 = df.loc[idx, 'EMA_200']
        stop_loss = ema_200 * (1 - self.stop_loss_buffer)
        
        return stop_loss
    
    def calculate_take_profit(self, df, idx, entry_price):
        """Calculate take profit using measured move"""
        # Find base setup parameters
        lookback_data = df.iloc[max(0, idx - self.base_lookback_period):idx]
        peak_idx = lookback_data['High'].idxmax()
        peak_price = lookback_data['High'].max()
        peak_position = df.index.get_loc(peak_idx)
        
        # Find lowest price during consolidation
        consolidation_data = df.iloc[peak_position:idx+1]
        lowest_consolidation_price = consolidation_data['Low'].min()
        
        # Calculate base depth
        base_depth = peak_price - lowest_consolidation_price
        
        # Calculate take profit
        take_profit = entry_price + (self.measured_move_multiplier * base_depth)
        
        return take_profit, base_depth
    
    def run_backtest(self, df):
        """Run the backtest on a single stock"""
        trades = []
        position = None
        
        # Calculate indicators
        df = self.calculate_emas(df)
        df = self.create_higher_timeframes(df)
        
        # Slice to backtest period
        backtest_df = df[df.index >= self.backtest_start].copy()
        
        if len(backtest_df) < 50:
            return trades
        
        # Iterate through backtest period
        for idx in backtest_df.index:
            position_idx = df.index.get_loc(idx)
            
            if position is None:
                # Check for setup and entry
                if self.check_broom_setup(df, position_idx):
                    if self.check_entry_signal(df, position_idx):
                        entry_price = df.loc[idx, 'Close']
                        stop_loss = self.calculate_stop_loss(df, position_idx, entry_price)
                        take_profit, base_depth = self.calculate_take_profit(
                            df, position_idx, entry_price
                        )
                        
                        position = {
                            'entry_date': idx,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'base_depth': base_depth,
                            'initial_stop': stop_loss
                        }
                        
                        logger.info(f"Entry signal for {idx.date()}: Price={entry_price:.2f}, "
                                  f"Stop={stop_loss:.2f}, Target={take_profit:.2f}")
            else:
                # Manage existing position
                current_price = df.loc[idx, 'Close']
                current_high = df.loc[idx, 'High']
                current_low = df.loc[idx, 'Low']
                ema_200 = df.loc[idx, 'EMA_200']
                
                # Update trailing stop based on rising EMA
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
                        'exit_reason': 'stop_loss',
                        'base_depth': position['base_depth']
                    })
                    logger.info(f"Exit (Stop Loss) for {idx.date()}: Return={trades[-1]['return_pct']:.2f}%")
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
                        'exit_reason': 'take_profit',
                        'base_depth': position['base_depth']
                    })
                    logger.info(f"Exit (Take Profit) for {idx.date()}: Return={trades[-1]['return_pct']:.2f}%")
                    position = None
        
        # Close any open position at the end
        if position is not None:
            last_idx = backtest_df.index[-1]
            last_price = backtest_df.loc[last_idx, 'Close']
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': last_idx,
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'return_pct': (last_price - position['entry_price']) / position['entry_price'] * 100,
                'exit_reason': 'end_of_period',
                'base_depth': position['base_depth']
            })
        
        return trades
    
    def calculate_performance_metrics(self, df, trades):
        """Calculate performance metrics for a stock"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'avg_return_per_trade': 0,
                'profit_factor': 0
            }
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        # Calculate equity curve
        backtest_df = df[df.index >= self.backtest_start].copy()
        equity = pd.Series(index=backtest_df.index, data=100.0)  # Start with 100
        current_equity = 100.0
        
        for _, trade in trades_df.iterrows():
            trade_return = trade['return_pct'] / 100
            current_equity *= (1 + trade_return)
            if trade['exit_date'] in equity.index:
                equity[trade['exit_date']] = current_equity
        
        # Forward fill equity curve
        equity = equity.ffill()
        
        # Calculate metrics
        total_return = (current_equity - 100) / 100 * 100
        max_drawdown = ((equity.cummax() - equity) / equity.cummax()).max() * 100
        
        # Average return per trade
        avg_return_per_trade = trades_df['return_pct'].mean()
        
        # Profit factor
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
    
    def run_universe_backtest(self, ticker_list, batch_size=10):
        """Run backtest for entire universe with batching"""
        print(f"Starting backtest for {len(ticker_list)} stocks...")
        print(f"Backtest period: {self.backtest_start} to present")
        print("-" * 80)
        
        start_time = time.time()
        processed_count = 0
        failed_count = 0
        
        # Process in batches
        for batch_start in range(0, len(ticker_list), batch_size):
            batch_end = min(batch_start + batch_size, len(ticker_list))
            batch = ticker_list[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: stocks {batch_start+1}-{batch_end}")
            
            for i, ticker in enumerate(batch):
                global_idx = batch_start + i
                try:
                    print(f"[{global_idx+1}/{len(ticker_list)}] Processing {ticker}...")
                    
                    # Download data
                    df = self.download_data(ticker)
                    if df is None or len(df) < 300:
                        print(f"  Skipping {ticker}: insufficient data")
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
                        failed_count += 1
                        continue
                    
                    # Run backtest
                    trades = self.run_backtest(df)
                    
                    # Calculate metrics
                    metrics = self.calculate_performance_metrics(df, trades)
                    metrics['ticker'] = ticker
                    metrics['processed'] = True
                    metrics['reason'] = 'success'
                    
                    self.results.append(metrics)
                    processed_count += 1
                    
                    # Print interim results
                    if trades:
                        print(f"  Trades: {metrics['total_trades']}, "
                              f"Win Rate: {metrics['win_rate']:.1f}%, "
                              f"Return: {metrics['total_return']:.1f}%")
                    else:
                        print(f"  No trades generated")
                    
                    # Clear memory
                    del df
                    del trades
                    gc.collect()
                    
                    # Add delay between stocks
                    time.sleep(1)  # 1 second delay
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}", exc_info=True)
                    print(f"  Error processing {ticker}: {str(e)}")
                    self.results.append({
                        'ticker': ticker,
                        'total_trades': 0,
                        'win_rate': 0,
                        'total_return': 0,
                        'max_drawdown': 0,
                        'avg_return_per_trade': 0,
                        'profit_factor': 0,
                        'processed': False,
                        'reason': f'error: {str(e)}'
                    })
                    failed_count += 1
            
            # Add longer delay between batches
            if batch_end < len(ticker_list):
                logger.info(f"Batch complete. Waiting 5 seconds before next batch...")
                time.sleep(5)
        
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"Backtest completed: {processed_count}/{len(ticker_list)} stocks processed successfully")
        print(f"Failed: {failed_count} stocks")
        print(f"Time elapsed: {elapsed_time/60:.2f} minutes")
        
        return self.results
    
    def export_results(self, filename='nifty500_broom_breakout_results.csv'):
        """Export results to CSV"""
        results_df = pd.DataFrame(self.results)
        
        if results_df.empty:
            logger.error("No results to export")
            return None
        
        # Reorder columns
        column_order = [
            'ticker', 'processed', 'reason', 'total_trades', 'win_rate', 
            'total_return', 'max_drawdown', 'avg_return_per_trade', 'profit_factor'
        ]
        
        # Filter to existing columns
        column_order = [col for col in column_order if col in results_df.columns]
        results_df = results_df[column_order]
        
        results_df.to_csv(filename, index=False)
        print(f"\nResults exported to {filename}")
        print(f"Total records: {len(results_df)}")
        
        # Print summary statistics
        processed_df = results_df[results_df['processed'] == True]
        if len(processed_df) > 0:
            print("\n=== SUMMARY STATISTICS ===")
            print(f"Stocks successfully processed: {len(processed_df)}")
            print(f"Stocks with trades: {len(processed_df[processed_df['total_trades'] > 0])}")
            print(f"Total trades: {processed_df['total_trades'].sum()}")
            print(f"Average win rate: {processed_df['win_rate'].mean():.2f}%")
            print(f"Average return: {processed_df['total_return'].mean():.2f}%")
            print(f"Average max drawdown: {processed_df['max_drawdown'].mean():.2f}%")
            
            # Top 10 performers
            stocks_with_trades = processed_df[processed_df['total_trades'] > 0]
            if len(stocks_with_trades) > 0:
                top_performers = stocks_with_trades.nlargest(10, 'total_return')[
                    ['ticker', 'total_return', 'win_rate', 'total_trades']
                ]
                print("\n=== TOP 10 PERFORMERS ===")
                print(top_performers.to_string(index=False))
        
        return results_df


def get_nifty500_tickers():
    """Get list of Nifty 500 tickers"""
    # This is a sample list - in production, you would fetch this from NSE
    
    nifty_500_sample = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
        # Add more tickers here...
        # In production, fetch complete list from NSE API or CSV file
    ]
    
    return nifty_500_sample


def main():
    """Main execution function"""
    print("=" * 80)
    print("INSTITUTIONAL MOVING AVERAGE 'BROOM' BREAKOUT STRATEGY")
    print("Nifty 500 Universe Backtesting Engine - Enhanced Version")
    print("=" * 80)
    
    # Get universe
    tickers = get_nifty500_tickers()
    print(f"\nUniverse size: {len(tickers)} stocks")
    
    # Initialize backtest engine
    engine = BroomBreakoutBacktest()
    
    # Run backtest with smaller batch size to avoid rate limiting
    results = engine.run_universe_backtest(tickers, batch_size=5)
    
    # Export results
    results_df = engine.export_results()
    
    print("\nBacktest completed successfully!")
    return results_df


if __name__ == "__main__":
    main()
