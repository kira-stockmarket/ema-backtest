"""
Trade Generator - Uses learned parameters to generate current trade signals
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import requests
import time
from typing import Optional, Dict, List, Tuple

# ==================== LOAD LEARNED PARAMS ====================

def load_learned_params():
    params_file = Path('state/current_params.json')
    
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                data = json.load(f)
            return data.get('params', {}), data.get('version', 0)
        except:
            pass
    
    return {
        'broom_compression_threshold': 0.08,
        'volume_threshold_multiplier': 1.5,
        'stop_loss_buffer': 0.015,
        'measured_move_multiplier': 2.0,
    }, 0

learned_params, version = load_learned_params()

COMPRESSION_THRESHOLD = learned_params.get('broom_compression_threshold', 0.08)
VOLUME_MULTIPLIER = learned_params.get('volume_threshold_multiplier', 1.5)
STOP_LOSS_BUFFER = learned_params.get('stop_loss_buffer', 0.015)
MEASURED_MOVE = learned_params.get('measured_move_multiplier', 2.0)

EMA_PERIODS = [20, 50, 100, 200]

# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trade_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== DATA FETCHER ====================

class DataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_data(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(period='1y', auto_adjust=False, timeout=30)
            
            if df is not None and not df.empty:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    return df[required_cols].dropna()
            return None
        except:
            return None

# ==================== TRADE GENERATOR ====================

class TradeGenerator:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.signals = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in EMA_PERIODS:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def check_signal(self, df: pd.DataFrame, ticker: str) -> Optional[Dict]:
        if len(df) < 200:
            return None
        
        idx = len(df) - 1
        current_price = df.iloc[idx]['Close']
        
        # EMA compression
        emas = [df.iloc[idx][f'EMA_{p}'] for p in EMA_PERIODS]
        if any(pd.isna(e) for e in emas):
            return None
        
        compression = (max(emas) - min(emas)) / current_price
        
        if compression >= COMPRESSION_THRESHOLD:
            return None
        
        # Volume check
        volume = df.iloc[idx]['Volume']
        volume_ma = df.iloc[idx]['Volume_MA_50']
        
        if pd.isna(volume_ma) or volume_ma <= 0:
            return None
        
        volume_ratio = volume / volume_ma
        
        if volume_ratio < VOLUME_MULTIPLIER:
            return None
        
        # Price above 200 EMA
        ema_200 = df.iloc[idx]['EMA_200']
        if pd.isna(ema_200) or current_price <= ema_200:
            return None
        
        # Calculate levels
        entry_price = current_price
        stop_loss = ema_200 * (1 - STOP_LOSS_BUFFER)
        
        lookback = df.iloc[max(0, idx-200):idx]
        peak_price = lookback['High'].max()
        peak_idx = df.index.get_loc(lookback['High'].idxmax())
        base_low = df.iloc[peak_idx:idx+1]['Low'].min()
        base_depth = peak_price - base_low
        take_profit = entry_price + (MEASURED_MOVE * base_depth)
        
        rsi = df.iloc[idx]['RSI']
        
        return {
            'ticker': ticker,
            'signal_date': datetime.now().strftime('%Y-%m-%d'),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk': round(entry_price - stop_loss, 2),
            'reward': round(take_profit - entry_price, 2),
            'risk_reward': round((take_profit - entry_price) / (entry_price - stop_loss), 2) if entry_price > stop_loss else 0,
            'compression': round(compression * 100, 2),
            'volume_ratio': round(volume_ratio, 2),
            'rsi': round(rsi, 1) if not pd.isna(rsi) else 0,
            'action': 'BUY',
        }
    
    def generate(self, tickers: List[str]) -> pd.DataFrame:
        logger.info("=" * 80)
        logger.info("GENERATING TRADE SIGNALS")
        logger.info(f"Using learned parameters (v{version})")
        logger.info(f"Compression: {COMPRESSION_THRESHOLD}")
        logger.info(f"Volume: {VOLUME_MULTIPLIER}")
        logger.info(f"Stop: {STOP_LOSS_BUFFER}")
        logger.info(f"Target: {MEASURED_MOVE}")
        logger.info("=" * 80)
        
        for ticker in tickers:
            try:
                df = self.data_fetcher.fetch_data(ticker)
                
                if df is None:
                    continue
                
                df = self.calculate_indicators(df)
                signal = self.check_signal(df, ticker)
                
                if signal:
                    self.signals.append(signal)
                    logger.info(f"🎯 SIGNAL: {ticker}")
                    logger.info(f"   Entry: ₹{signal['entry_price']}")
                    logger.info(f"   Stop: ₹{signal['stop_loss']}")
                    logger.info(f"   Target: ₹{signal['take_profit']}")
                    logger.info(f"   R:R: {signal['risk_reward']}")
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.debug(f"Error for {ticker}: {e}")
        
        signals_df = pd.DataFrame(self.signals)
        
        # ALWAYS save, even if empty
        if signals_df.empty:
            signals_df = pd.DataFrame(columns=[
                'ticker', 'signal_date', 'entry_price', 'stop_loss', 
                'take_profit', 'risk', 'reward', 'risk_reward', 
                'compression', 'volume_ratio', 'rsi', 'action'
            ])
        
        signals_df.to_csv('nifty500_broom_breakout_results.csv', index=False)
        logger.info(f"\n✓ Saved {len(signals_df)} signals to CSV")
        
        return signals_df

# ==================== MAIN ====================

def main():
    # Ensure directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('state', exist_ok=True)
    
    tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
        'DIVISLAB.NS', 'DRREDDY.NS', 'CIPLA.NS', 'BRITANNIA.NS', 'DABUR.NS',
        'PIDILITIND.NS', 'HAVELLS.NS', 'ASTRAL.NS', 'DIXON.NS', 'TRENT.NS',
        'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS',
        'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'MPHASIS.NS', 'COFORGE.NS',
        'PERSISTENT.NS', 'TATACONSUM.NS', 'GODREJCP.NS', 'MARICO.NS', 'UBL.NS',
        'VOLTAS.NS', 'CROMPTON.NS', 'KEI.NS', 'POLYCAB.NS', 'SUPREMEIND.NS',
    ]
    
    generator = TradeGenerator()
    signals = generator.generate(tickers)
    
    print(f"\n✅ Generated {len(signals)} trade signals")
    print(f"✅ CSV file created: nifty500_broom_breakout_results.csv")
    
    return signals

if __name__ == "__main__":
    main()
