import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
import warnings
from datetime import datetime

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

def calculate_ema(series, length):
    """Native Pandas EMA calculation, handles missing early data gracefully"""
    return series.ewm(span=length, adjust=False, min_periods=1).mean()

def calculate_poc(recent_closes, recent_volumes, bins=10):
    price_bins = np.linspace(recent_closes.min(), recent_closes.max(), bins)
    volume_by_price = np.zeros(bins)
    for price, vol in zip(recent_closes, recent_volumes):
        if np.isnan(price) or np.isnan(vol):
            continue
        closest_bin_idx = np.abs(price_bins - price).argmin()
        volume_by_price[closest_bin_idx] += vol
    poc_index = np.argmax(volume_by_price)
    return price_bins[poc_index]

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
        # Ensure we have enough lookback data and the indicator is valid
        if len(self.data.Close) < 200 or np.isnan(self.monthly_200[-1]):
            return

        current_price = self.data.Close[-1]

        # --- RULE 1: MACRO TREND FILTER ---
        macro_bullish = (current_price > self.monthly_200[-1]) and (current_price > self.weekly_200[-1])
        if not macro_bullish:
            return

        # --- RULE 2: EMA BROOM COMPRESSION (Relaxed to 5% to account for market noise) ---
        current_emas = [self.ema20[-1], self.ema50[-1], self.ema100[-1], self.ema200[-1]]
        spread = (max(current_emas) - min(current_emas)) / current_price
        is_ema_compressed = spread < 0.05  

        if is_ema_compressed and not self.position:
            # --- RULE 3: CONSOLIDATION DURATION (63 to 147 bars) ---
            lookback_window = 200
            recent_highs = self.data.High[-lookback_window:]
            highest_idx = np.argmax(recent_highs)
            days_since_high = (lookback_window - 1) - highest_idx
            is_valid_duration = 63 <= days_since_high <= 147

            # --- RULE 4: STRAIGHT CONSOLIDATION BOX (Relaxed to < 12% over last 20 days) ---
            recent_20_highs = self.data.High[-20:]
            recent_20_lows = self.data.Low[-20:]
            box_high = max(recent_20_highs)
            box_low = min(recent_20_lows)
            is_straight_consolidation = ((box_high - box_low) / box_low) < 0.12

            # --- RULE 5: PREVIOUS TREND CAP (<= 40% prior run-up) ---
            if highest_idx > 40:
                prior_trend_low = min(self.data.Low[highest_idx - 40 : highest_idx])
                run_up_pct = (recent_highs[highest_idx] - prior_trend_low) / (prior_trend_low + 1e-9)
                is_healthy_trend = run_up_pct <= 0.40
            else:
                is_healthy_trend = False

            # --- EXECUTION & MEASURED MOVE TARGET ---
            if is_valid_duration and is_straight_consolidation and is_healthy_trend:
                recent_volumes = self.data.Volume[-20:]
                poc_price = calculate_poc(self.data.Close[-20:], recent_volumes)
                
                is_breakout = (current_price > max(current_emas)) and (current_price > poc_price)
                volume_spike = self.data.Volume[-1] > (np.mean(recent_volumes) * 1.5 + 1e-9)

                if is_breakout and volume_spike:
                    sl_price = self.ema200[-1] * 0.99
                    peak_price = recent_highs[highest_idx]
                    base_low = min(self.data.Low[-days_since_high:])
                    base_depth = peak_price - base_low
                    tp_price = current_price + (2 * base_depth)

                    self.buy(sl=sl_price, tp=tp_price)

        # --- RULE 6: TRAILING STOP MANAGEMENT ---
        elif self.position:
            new_sl = self.ema200[-1] * 0.99
            if new_sl > self.trades[0].sl:
                self.trades[0].sl = new_sl

def download_and_prepare_data(ticker):
    # CRITICAL FIX: Fetch from year 2000 to give the 200 Monthly EMA 16+ years to warm up
    stock = yf.Ticker(ticker)
    df = stock.history(start="2000-01-01")

    if df.empty or len(df) < 250:
        return None

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Modern Pandas offsets
    df_weekly = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    df_monthly = df.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})

    df_weekly["Weekly_200EMA"] = calculate_ema(df_weekly["Close"], 200)
    df_monthly["Monthly_200EMA"] = calculate_ema(df_monthly["Close"], 200)

    # Merge data and forward-fill to prevent lookahead bias
    df = df.join(df_weekly[["Weekly_200EMA"]]).ffill()
    df = df.join(df_monthly[["Monthly_200EMA"]]).ffill()
    df.dropna(inplace=True)

    # CRITICAL FIX: Slice the dataframe to only test trades from 2018 onwards
    df = df[df.index >= "2018-01-01"]
    
    # Check again if data exists post-2018
    if df.empty or len(df) < 50:
        return None

    return df

if __name__ == "__main__":
    # TRUE NIFTY 100 LIST
    nifty_100_tickers = [
        "ABB.NS", "ACC.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "AMBUJACEM.NS", 
        "APOLLOHOSP.NS", "ASIANPAINT.NS", "DMART.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", 
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "BANKBARODA.NS", "BEL.NS", "BHARATFORG.NS", 
        "BPCL.NS", "BHARTIARTL.NS", "BHEL.NS", "BOSCHLTD.NS", "CANBK.NS", "CIPLA.NS", 
        "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "CROMPTON.NS", "DABUR.NS", 
        "DIVISLAB.NS", "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", "GAIL.NS", "GODREJCP.NS", 
        "GODREJPROP.NS", "GRASIM.NS", "HAVELLS.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", 
        "HEROMOTOCO.NS", "HINDALCO.NS", "HAL.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", 
        "ICICIPRULI.NS", "ITC.NS", "IOC.NS", "IRCTC.NS", "INFY.NS", "INDIGO.NS", "JSWSTEEL.NS", 
        "JINDALSTEL.NS", "KOTAKBANK.NS", "L&T.NS", "LTIM.NS", "LTTS.NS", "M&M.NS", "MARICO.NS", 
        "MARUTI.NS", "MUTHOOTFIN.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "PAGEIND.NS", 
        "PIDILITIND.NS", "PIIND.NS", "POWERGRID.NS", "PNB.NS", "RELIANCE.NS", "SBICARD.NS", 
        "SBILIFE.NS", "SBIN.NS", "SRF.NS", "MOTHERSON.NS", "SHREECEM.NS", "SIEMENS.NS", 
        "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", 
        "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "ULTRACEMCO.NS", 
        "UPL.NS", "VEDL.NS", "WIPRO.NS", "ZOMATO.NS", "ZYDUSLIFE.NS"
    ]
    
    results = []
    print(f"Starting Backtest on {len(nifty_100_tickers)} Nifty Stocks since 2018...\n")

    for ticker in nifty_100_tickers:
        print(f"Processing {ticker}...")
        try:
            data = download_and_prepare_data(ticker)
            if data is None:
                print(f"  -> Skipped (Insufficient Data)")
                continue

            bt = Backtest(data, InstitutionalBroomBreakout, cash=1000000, commission=0.001)
            stats = bt.run()
            
            # Only record if at least one trade was taken
            if stats['# Trades'] > 0:
                results.append({
                    "Ticker": ticker,
                    "Return [%]": round(stats['Return [%]'], 2),
                    "Max Drawdown [%]": round(stats['Max. Drawdown [%]'], 2),
                    "Win Rate [%]": round(stats['Win Rate [%]'], 2) if not np.isnan(stats['Win Rate [%]']) else 0.0,
                    "Total Trades": stats['# Trades']
                })
        except Exception as e:
            print(f"  -> Error on {ticker}: {e}")

    # Compile Summary
    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df.to_csv("portfolio_summary.csv", index=False)
        print("\n=== PORTFOLIO BACKTEST COMPLETE ===")
        print(summary_df)
        print(f"\nTotal Stocks Traded: {len(summary_df)}")
        print("Results saved to 'portfolio_summary.csv'")
    else:
        print("\n0 Trades executed across all 100 stocks. The parameters may still be too strict.")
