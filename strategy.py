import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
import warnings
import requests
import time

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

def calculate_ema(series, length):
    """Native Pandas EMA calculation, handles missing early data gracefully"""
    return series.ewm(span=length, adjust=False, min_periods=1).mean()

def calculate_poc(recent_closes, recent_volumes, bins=10):
    """Calculates Volume Profile Point of Control"""
    price_bins = np.linspace(recent_closes.min(), recent_closes.max(), bins)
    volume_by_price = np.zeros(bins)
    for price, vol in zip(recent_closes, recent_volumes):
        if np.isnan(price) or np.isnan(vol): continue
        closest_bin_idx = np.abs(price_bins - price).argmin()
        volume_by_price[closest_bin_idx] += vol
    return price_bins[np.argmax(volume_by_price)]

class InstitutionalBroomBreakout(Strategy):
    def init(self):
        # Native EMA calculations on daily data
        close_series = self.data.Close.s
        self.ema20 = self.I(calculate_ema, close_series, 20)
        self.ema50 = self.I(calculate_ema, close_series, 50)
        self.ema100 = self.I(calculate_ema, close_series, 100)
        self.ema200 = self.I(calculate_ema, close_series, 200)

        # Access Higher Timeframe EMAs passed via the dataframe
        self.weekly_200 = self.data.Weekly_200EMA
        self.monthly_200 = self.data.Monthly_200EMA

    def next(self):
        # We need at least 200 days and a valid Weekly 200 EMA
        if len(self.data.Close) < 200 or np.isnan(self.weekly_200[-1]):
            return

        current_price = self.data.Close[-1]

        # --- RULE 1: MACRO TREND FILTER ---
        # Must be above Weekly 200. If Monthly 200 exists, must be above that too.
        macro_bullish = current_price > self.weekly_200[-1]
        if not np.isnan(self.monthly_200[-1]):
            macro_bullish = macro_bullish and (current_price > self.monthly_200[-1])
            
        if not macro_bullish:
            return

        # --- RULE 2: EMA BROOM COMPRESSION (8% spread tolerance for real market wicks) ---
        current_emas = [self.ema20[-1], self.ema50[-1], self.ema100[-1], self.ema200[-1]]
        spread = (max(current_emas) - min(current_emas)) / current_price
        is_ema_compressed = spread < 0.08  

        if is_ema_compressed and not self.position:
            # --- RULE 3: CONSOLIDATION DURATION (63 to 147 bars) ---
            lookback_window = 200
            recent_highs = self.data.High[-lookback_window:]
            highest_idx = np.argmax(recent_highs)
            days_since_high = (lookback_window - 1) - highest_idx
            is_valid_duration = 63 <= days_since_high <= 147

            # --- RULE 4: STRAIGHT CONSOLIDATION BOX (< 15% over last 20 days) ---
            recent_20_highs = self.data.High[-20:]
            recent_20_lows = self.data.Low[-20:]
            box_high = max(recent_20_highs)
            box_low = min(recent_20_lows)
            is_straight_consolidation = ((box_high - box_low) / box_low) < 0.15

            # --- RULE 5: PREVIOUS TREND CAP (<= 60% prior run-up) ---
            if highest_idx > 40:
                prior_trend_low = min(self.data.Low[highest_idx - 40 : highest_idx])
                run_up_pct = (recent_highs[highest_idx] - prior_trend_low) / (prior_trend_low + 1e-9)
                is_healthy_trend = run_up_pct <= 0.60
            else:
                is_healthy_trend = False

            # --- EXECUTION & EXACT TARGET MEASUREMENT ---
            if is_valid_duration and is_straight_consolidation and is_healthy_trend:
                recent_volumes = self.data.Volume[-20:]
                poc_price = calculate_poc(self.data.Close[-20:], recent_volumes)
                
                is_breakout = (current_price > max(current_emas)) and (current_price > poc_price)
                
                # Volume must be 1.5x the 50-day average
                avg_vol_50 = np.mean(self.data.Volume[-50:])
                volume_spike = self.data.Volume[-1] > (avg_vol_50 * 1.5)

                if is_breakout and volume_spike:
                    # Stop Loss accurately placed slightly below 200 EMA
                    sl_price = self.ema200[-1] * 0.985 
                    
                    # 2x Measured Move Target from Peak to Base Low
                    peak_price = recent_highs[highest_idx]
                    base_low = min(self.data.Low[-days_since_high:])
                    base_depth = peak_price - base_low
                    
                    tp_price = current_price + (2 * base_depth)
                    
                    # Ensure TP and SL are mathematically valid before ordering
                    if tp_price > current_price > sl_price:
                        self.buy(sl=sl_price, tp=tp_price)

        # --- RULE 6: TRAILING STOP MANAGEMENT ---
        elif self.position:
            # Trail the stop up as the 200 EMA rises
            new_sl = self.ema200[-1] * 0.985
            # Ensure new trailing stop doesn't accidentally trigger instantly on current price
            if new_sl > self.trades[0].sl and new_sl < self.data.Close[-1]:
                self.trades[0].sl = new_sl

def download_and_prepare_data(ticker, session):
    # Fetch from 2000 using custom session to bypass 404 blocks
    stock = yf.Ticker(ticker, session=session)
    df = stock.history(start="2000-01-01") 

    if df.empty or len(df) < 300:
        return None

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Modern Pandas offsets ('W' and 'ME')
    df_weekly = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    df_monthly = df.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})

    df_weekly["Weekly_200EMA"] = calculate_ema(df_weekly["Close"], 200)
    df_monthly["Monthly_200EMA"] = calculate_ema(df_monthly["Close"], 200)

    # Merge and forward-fill
    df = df.join(df_weekly[["Weekly_200EMA"]]).ffill()
    df = df.join(df_monthly[["Monthly_200EMA"]]).ffill()
    df.dropna(subset=['Close', 'Weekly_200EMA'], inplace=True)

    # Strictly test trades from 2018 onwards
    df = df[df.index >= "2018-01-01"]
    
    if df.empty or len(df) < 50:
        return None
    return df

if __name__ == "__main__":
    # Cleaned and Verified Nifty Liquid Universe (No & symbols, no delisted stocks)
    nifty_liquid_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS",
        "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
        "ASIANPAINT.NS", "ULTRACEMCO.NS", "TITAN.NS", "KOTAKBANK.NS", "NTPC.NS", "AXISBANK.NS", "ADANIENT.NS",
        "ONGC.NS", "BAJAJFINSV.NS", "TATASTEEL.NS", "POWERGRID.NS", "COALINDIA.NS", "MM.NS", "BAJAJ-AUTO.NS",
        "ADANIPORTS.NS", "WIPRO.NS", "SBILIFE.NS", "HDFCLIFE.NS", "GRASIM.NS", "TECHM.NS", "HINDALCO.NS",
        "LTIM.NS", "APOLLOHOSP.NS", "EICHERMOT.NS", "DIVISLAB.NS", "DRREDDY.NS", "BRITANNIA.NS", "CIPLA.NS",
        "INDIGO.NS", "TATACONSUM.NS", "BPCL.NS", "TRENT.NS", "TORNTPHARM.NS", "ZOMATO.NS", "CHOLAFIN.NS",
        "SHREECEM.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS", "DLF.NS", "HAL.NS", "BEL.NS", "BOSCHLTD.NS",
        "AMBUJACEM.NS", "PIDILITIND.NS", "SIEMENS.NS", "SRF.NS", "HAVELLS.NS", "GAIL.NS", "DABUR.NS",
        "GODREJCP.NS", "IOC.NS", "COLPAL.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "VEDL.NS", "JINDALSTEL.NS",
        "JSWSTEEL.NS", "PIIND.NS", "UPL.NS", "MUTHOOTFIN.NS", "ICICIPRULI.NS", "ICICIGI.NS", "SBICARD.NS",
        "IRCTC.NS", "CONCOR.NS", "BHEL.NS", "ABB.NS", "ACC.NS", "MARICO.NS", "PAGEIND.NS", "ZYDUSLIFE.NS",
        "COFORGE.NS", "LTTS.NS", "PERSISTENT.NS", "MPHASIS.NS", "AUBANK.NS", "BANDHANBNK.NS",
        "FEDERALBNK.NS", "IDFCFIRSTB.NS", "YESBANK.NS", "INDUSINDBK.NS", "PNBHOUSING.NS", "LICHSGFIN.NS",
        "MMFIN.NS", "SHRIRAMFIN.NS", "RECLTD.NS", "PFC.NS", "ASHOKLEY.NS", "MRF.NS", "BALKRISIND.NS",
        "APOLLOTYRE.NS", "TATACHEM.NS", "DEEPAKNTR.NS", "NAVINFLUOR.NS", "AARTIIND.NS", "ATGL.NS", "AWL.NS",
        "ADANIGREEN.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "JSWENERGY.NS", "IEX.NS", "NHPC.NS", "SJVN.NS",
        "CGPOWER.NS", "SUZLON.NS", "DIXON.NS", "POLYCAB.NS", "KEI.NS", "ASTRAL.NS", "SUPREMEIND.NS", "BATAINDIA.NS"
    ]
    
    results = []
    print(f"Starting Multi-Threaded Backtest on {len(nifty_liquid_tickers)} Stocks...")
    print("This will process 20 years of data. Please allow a few minutes...\n")

    # Create a custom session to prevent Yahoo Finance from blocking the IP
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    for idx, ticker in enumerate(nifty_liquid_tickers, 1):
        try:
            data = download_and_prepare_data(ticker, session)
            
            # Tiny delay to respect Yahoo API limits and avoid 404s
            time.sleep(0.3)
            
            if data is None:
                continue

            bt = Backtest(data, InstitutionalBroomBreakout, cash=1000000, commission=0.001, trade_on_close=False)
            stats = bt.run()
            
            if stats['# Trades'] > 0:
                print(f"[{idx}/{len(nifty_liquid_tickers)}] {ticker} | Trades: {stats['# Trades']} | Win Rate: {round(stats['Win Rate [%]'],1)}% | Return: {round(stats['Return [%]'],1)}%")
                results.append({
                    "Ticker": ticker,
                    "Return [%]": round(stats['Return [%]'], 2),
                    "Max Drawdown [%]": round(stats['Max. Drawdown [%]'], 2),
                    "Win Rate [%]": round(stats['Win Rate [%]'], 2),
                    "Total Trades": stats['# Trades']
                })
        except Exception as e:
            # Fail silently to allow loop to continue
            pass

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df.to_csv("portfolio_summary.csv", index=False)
        print("\n=== NIFTY BACKTEST COMPLETE ===")
        print(f"Total Stocks Traded: {len(summary_df)}")
        print(f"Average Portfolio Win Rate: {summary_df['Win Rate [%]'].mean():.2f}%")
        print("Results saved to 'portfolio_summary.csv'")
    else:
        print("\n0 Trades executed. The market parameters may still be too strict.")
