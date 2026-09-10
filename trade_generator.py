"""
UPDATED Trade Generator - Full Nifty 500 Universe
Uses learned parameters to scan all 500 stocks and appends new signals.
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import requests
import time
from typing import Optional, Dict, List

# ==================== LOAD LEARNED PARAMS ====================

def load_learned_params():
    params_file = Path('state/current_params.json')
    
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                data = json.load(f)
            return data.get('params', {}), data.get('version', 0)
        except Exception:
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

# Add these two lines right here!
os.makedirs('logs', exist_ok=True)
os.makedirs('state', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trade_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== NIFTY 500 UNIVERSE ====================

def get_nifty500_tickers() -> List[str]:
    """Complete Nifty 500 stock list"""
    return [
        # Large Cap (Top 50)
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
        'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS',
        'BAJAJFINSV.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
        'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
        'GRASIM.NS', 'HCLTECH.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
        'HINDALCO.NS', 'INDUSINDBK.NS', 'JSWSTEEL.NS', 'M&M.NS',
        'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'SBILIFE.NS',
        'SHRIRAMFIN.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
        'TECHM.NS', 'UPL.NS',
        
        # Mid Cap (51-200)
        'ABB.NS', 'ACC.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS', 'ALKEM.NS',
        'AMBUJACEM.NS', 'APLAPOLLO.NS', 'ASHOKLEY.NS', 'ASTRAL.NS', 'ATGL.NS',
        'AUROPHARMA.NS', 'BAJAJHLDNG.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'BATAINDIA.NS',
        'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BIOCON.NS', 'BOSCHLTD.NS',
        'CANBK.NS', 'CHOLAFIN.NS', 'COLPAL.NS', 'CONCOR.NS', 'CROMPTON.NS',
        'CUMMINSIND.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DLF.NS', 'DMART.NS',
        'ESCORTS.NS', 'FEDERALBNK.NS', 'FORTIS.NS', 'GAIL.NS', 'GICRE.NS',
        'GLAND.NS', 'GODREJCP.NS', 'GODREJPROP.NS', 'HAVELLS.NS', 'HINDCOPPER.NS',
        'HONAUT.NS', 'ICICIGI.NS', 'IDEA.NS', 'IDFCFIRSTB.NS', 'INDHOTEL.NS',
        'IEX.NS', 'IGL.NS', 'INDUSTOWER.NS', 'IRCTC.NS', 'JINDALSTEL.NS',
        'JUBLFOOD.NS', 'LICHSGFIN.NS', 'LUPIN.NS', 'M&MFIN.NS', 'MANAPPURAM.NS',
        'MARICO.NS', 'MFSL.NS', 'MOTHERSON.NS', 'MPHASIS.NS', 'MRF.NS',
        'MUTHOOTFIN.NS', 'NATIONALUM.NS', 'NAUKRI.NS', 'NAVINFLUOR.NS', 'NMDC.NS',
        'OBEROIRLTY.NS', 'OFSS.NS', 'PAGEIND.NS', 'PEL.NS', 'PERSISTENT.NS',
        'PETRONET.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNB.NS', 'POLYCAB.NS',
        'POONAWALLA.NS', 'POWERFIN.NS', 'PRESTIGE.NS', 'RAMCOCEM.NS', 'RBLBANK.NS',
        'RECLTD.NS', 'SAIL.NS', 'SBICARD.NS', 'SHREECEM.NS', 'SRF.NS',
        'SUNTV.NS', 'SUPREMEIND.NS', 'TATAPOWER.NS', 'TORNTPHARM.NS', 'TRENT.NS',
        'TVSMOTOR.NS', 'UBL.NS', 'UNITDSPR.NS', 'VBL.NS', 'VOLTAS.NS',
        'YESBANK.NS', 'ZYDUSLIFE.NS',
        
        # Small Cap (201-500)
        'AARTIIND.NS', 'ABBOTINDIA.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ADANIPOWER.NS',
        'AIAENG.NS', 'AJANTPHARM.NS', 'ALKYLAMINE.NS', 'ALLCARGO.NS', 'ANGELONE.NS',
        'APARINDS.NS', 'APOLLOTYRE.NS', 'ASAHIINDIA.NS', 'ASTRAMICRO.NS', 'ATUL.NS',
        'AVANTIFEED.NS', 'BAJAJELEC.NS', 'BALKRISIND.NS', 'BANDHANBNK.NS', 'BBTC.NS',
        'BDL.NS', 'BERGEPAINT.NS', 'BHARATRAS.NS', 'BIRLACORPN.NS', 'BLUEDART.NS',
        'BLUESTARCO.NS', 'BRIGADE.NS', 'BSOFT.NS', 'CANFINHOME.NS', 'CARBORUNIV.NS',
        'CASTROLIND.NS', 'CEATLTD.NS', 'CENTURYPLY.NS', 'CERA.NS', 'CHAMBLFERT.NS',
        'CGPOWER.NS', 'CLEAN.NS', 'COFORGE.NS', 'COROMANDEL.NS', 'CREDITACC.NS',
        'CUB.NS', 'CYIENT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS',
        'DHANI.NS', 'DISHTV.NS', 'DIXON.NS', 'EIDPARRY.NS', 'ELGIEQUIP.NS',
        'EMAMI.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'EPL.NS', 'EQUITASBNK.NS',
        'ERIS.NS', 'EXIDEIND.NS', 'FACT.NS', 'FINEORG.NS', 'FINPIPE.NS',
        'FSL.NS', 'GALAXYSURF.NS', 'GARFIBRES.NS', 'GENUSPOWER.NS', 'GHCL.NS',
        'GILLETTE.NS', 'GLENMARK.NS', 'GMMPFAUDLR.NS', 'GNFC.NS', 'GODFRYPHLP.NS',
        'GODREJAGRO.NS', 'GODREJIND.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRINDWELL.NS',
        'GSFC.NS', 'GUJGASLTD.NS', 'HAL.NS', 'HAPPSTMNDS.NS', 'HATSUN.NS',
        'HBLPOWER.NS', 'HEG.NS', 'HEIDELBERG.NS', 'HESTERBIO.NS', 'HFCL.NS',
        'HINDPETRO.NS', 'HITECH.NS', 'HUDCO.NS', 'IBULHSGFIN.NS', 'INDIAMART.NS',
        'INDIANB.NS', 'INDIGO.NS', 'INDOCO.NS', 'INTELLECT.NS', 'IOB.NS',
        'IRB.NS', 'ISEC.NS', 'ITI.NS', 'JBCHEPHARM.NS', 'JCHAC.NS',
        'JKCEMENT.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 'JSL.NS', 'JSWENERGY.NS',
        'JUBLINGREA.NS', 'JUSTDIAL.NS', 'JYOTHYLAB.NS', 'KALPATPOWR.NS', 'KANSAINER.NS',
        'KARURVYSYA.NS', 'KAJARIACER.NS', 'KEI.NS', 'KFINTECH.NS', 'KNRCON.NS',
        'KPRMILL.NS', 'KRBL.NS', 'L&TFH.NS', 'LALPATHLAB.NS', 'LAOPALA.NS',
        'LAURUSLABS.NS', 'LEMONTREE.NS', 'LINDEINDIA.NS', 'LODHA.NS', 'LTTS.NS',
        'MAHABANK.NS', 'MAHLIFE.NS', 'MAHINDCIE.NS', 'MANINFRA.NS', 'MARKSANS.NS',
        'MASFIN.NS', 'MAXHEALTH.NS', 'MASTEK.NS', 'MATRIMONY.NS', 'MAZDOCK.NS',
        'MBLINFRA.NS', 'MCX.NS', 'MEDPLUS.NS', 'METROPOLIS.NS', 'MIDHANI.NS',
        'MINDACORP.NS', 'MINDTREE.NS', 'MOL.NS', 'MOTILALOFS.NS', 'MRPL.NS',
        'MSTC.NS', 'MTARTECH.NS', 'NATCOPHARM.NS', 'NBCC.NS', 'NCC.NS',
        'NELCO.NS', 'NETWORK18.NS', 'NH.NS', 'NHPC.NS', 'NIACL.NS',
        'NLCINDIA.NS', 'NOCIL.NS', 'NUVOCO.NS', 'NYKAA.NS', 'OIL.NS',
        'OLECTRA.NS', 'OMAXE.NS', 'ONMOBILE.NS', 'ORIENTELEC.NS', 'PATANJALI.NS',
        'PAYTM.NS', 'PCBL.NS', 'PDSL.NS', 'PFIZER.NS', 'PHOENIXLTD.NS',
        'PIRHEALTH.NS', 'PNBHOUSING.NS', 'POLYMED.NS', 'PPLPHARMA.NS', 'PRINCEPIPE.NS',
        'PTC.NS', 'PVR.NS', 'QUESS.NS', 'RADICO.NS', 'RAILTEL.NS',
        'RAIN.NS', 'RALLIS.NS', 'RATNAMANI.NS', 'RAYMOND.NS', 'RBA.NS',
        'REDINGTON.NS', 'RELAXO.NS', 'RENUKA.NS', 'RHIM.NS', 'RITES.NS',
        'RKFORGE.NS', 'ROSSARI.NS', 'ROUTE.NS', 'RPG.NS', 'RVNL.NS',
        'SAPPHIRE.NS', 'SAREGAMA.NS', 'SUNFLAG.NS', 'SUNTECK.NS', 'SUPRAJIT.NS',
        'SUVENPHAR.NS', 'SWANENERGY.NS', 'SYNGENE.NS', 'TANLA.NS', 'TATACHEM.NS',
        'TATACOFFEE.NS', 'TATAELXSI.NS', 'TATAINVEST.NS', 'TCI.NS', 'TEJASNET.NS',
        'THERMAX.NS', 'TIMKEN.NS', 'TINPLATE.NS', 'TIINDIA.NS', 'TMB.NS',
        'TORNTPOWER.NS', 'TRIDENT.NS', 'TRITURBINE.NS', 'TTKPRESTIG.NS', 'TV18BRDCST.NS',
        'UCOBANK.NS', 'UFLEX.NS', 'UNIONBANK.NS', 'UTIAMC.NS', 'VAIBHAVGBL.NS',
        'VARROC.NS', 'VGUARD.NS', 'VIPIND.NS', 'VOLTAMP.NS', 'WELCORP.NS',
        'WELSPUNIND.NS', 'WESTLIFE.NS', 'WHIRLPOOL.NS', 'WOCKPHARMA.NS', 'ZENSARTECH.NS',
        'ZOMATO.NS',
    ]

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
        except Exception:
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
        
        # Price above 200 EMA
        ema_200 = df.iloc[idx]['EMA_200']
        if pd.isna(ema_200) or current_price <= ema_200:
            return None
        
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
        logger.info("GENERATING TRADE SIGNALS - NIFTY 500")
        logger.info(f"Universe: {len(tickers)} stocks")
        logger.info(f"Using learned parameters (v{version})")
        logger.info("=" * 80)
        
        processed = 0
        failed = 0
        
        for ticker in tickers:
            try:
                df = self.data_fetcher.fetch_data(ticker)
                
                if df is None:
                    failed += 1
                    continue
                
                df = self.calculate_indicators(df)
                signal = self.check_signal(df, ticker)
                
                if signal:
                    self.signals.append(signal)
                    logger.info(f"🎯 SIGNAL: {ticker}")
                
                processed += 1
                
                if processed % 20 == 0:
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)
                
            except Exception as e:
                failed += 1
                logger.debug(f"Error for {ticker}: {e}")
        
        logger.info(f"\n✓ Processed: {processed} | Failed: {failed}")
        
        signals_df = pd.DataFrame(self.signals)
        file_path = 'nifty500_broom_breakout_results.csv'
        
        # Check if file exists, then append and drop duplicates
        if not signals_df.empty:
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
                combined_df = pd.concat([existing_df, signals_df], ignore_index=True)
                # Keep the latest signal if multiple exist for the same ticker on the same date
                combined_df = combined_df.drop_duplicates(subset=['ticker', 'signal_date'], keep='last')
                combined_df.to_csv(file_path, index=False)
                logger.info(f"\n✓ Appended new signals to CSV. Total signals tracked: {len(combined_df)}")
            else:
                signals_df.to_csv(file_path, index=False)
                logger.info(f"\n✓ Saved {len(signals_df)} signals to new CSV")
        else:
            logger.info("\n✓ No new signals generated. Existing CSV preserved.")
            
        return signals_df

# ==================== MAIN ====================

def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('state', exist_ok=True)
    
    tickers = get_nifty500_tickers()
    logger.info(f"Total tickers to scan: {len(tickers)}")
    
    generator = TradeGenerator()
    signals = generator.generate(tickers)
    
    print(f"\n✅ Scan complete. Found {len(signals)} new trade signals today.")
    
    return signals

if __name__ == "__main__":
    main()
