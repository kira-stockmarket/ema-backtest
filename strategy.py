import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from backtesting import Backtest, Strategy


def calculate_poc(recent_closes, recent_volumes, bins=10):
  """Calculates the Point of Control (POC) using Volume Profile."""
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
    # Daily EMAs
    self.ema20 = self.I(ta.ema, self.data.Close.s, length=20)
    self.ema50 = self.I(ta.ema, self.data.Close.s, length=50)
    self.ema100 = self.I(ta.ema, self.data.Close.s, length=100)
    self.ema200 = self.I(ta.ema, self.data.Close.s, length=200)

    # Higher Timeframe EMAs passed via DataFrame
    self.weekly_200 = self.data.Weekly_200EMA
    self.monthly_200 = self.data.Monthly_200EMA

  def next(self):
    if len(self.data.Close) < 200 or np.isnan(self.monthly_200[-1]):
      return

    current_price = self.data.Close[-1]

    # --- RULE 1: MACRO TREND FILTER ---
    macro_bullish = (current_price > self.monthly_200[-1]) and (
        current_price > self.weekly_200[-1]
    )
    if not macro_bullish:
      return

    # --- RULE 2: EMA BROOM COMPRESSION ---
    current_emas = [
        self.ema20[-1],
        self.ema50[-1],
        self.ema100[-1],
        self.ema200[-1],
    ]
    spread = (max(current_emas) - min(current_emas)) / current_price
    is_ema_compressed = spread < 0.03  # EMAs compressed within 3%

    if is_ema_compressed and not self.position:
      # --- RULE 3: CONSOLIDATION DURATION (3 to 6 months ~ 63 to 147 bars) ---
      lookback_window = 200
      recent_highs = self.data.High[-lookback_window:]
      highest_idx = np.argmax(recent_highs)
      days_since_high = (lookback_window - 1) - highest_idx
      is_valid_duration = 63 <= days_since_high <= 147

      # --- RULE 4: STRAIGHT CONSOLIDATION BOX (< 8% over last 20 days) ---
      recent_20_highs = self.data.High[-20:]
      recent_20_lows = self.data.Low[-20:]
      box_high = max(recent_20_highs)
      box_low = min(recent_20_lows)
      is_straight_consolidation = ((box_high - box_low) / box_low) < 0.08

      # --- RULE 5: PREVIOUS TREND CAP (<= 40% prior run-up) ---
      if highest_idx > 40:
        prior_trend_low = min(self.data.Low[highest_idx - 40 : highest_idx])
        run_up_pct = (recent_highs[highest_idx] - prior_trend_low) / (
            prior_trend_low + 1e-9
        )
        is_healthy_trend = run_up_pct <= 0.40
      else:
        is_healthy_trend = False

      # --- EXECUTION & ACCUMULATION VALIDATION ---
      if is_valid_duration and is_straight_consolidation and is_healthy_trend:
        recent_volumes = self.data.Volume[-20:]
        poc_price = calculate_poc(self.data.Close[-20:], recent_volumes)

        # Volume Profile POC breakout + Volume spike confirmation
        is_breakout = (current_price > max(current_emas)) and (
            current_price > poc_price
        )
        volume_spike = self.data.Volume[-1] > (
            np.mean(recent_volumes) * 1.5 + 1e-9
        )

        if is_breakout and volume_spike:
          # Stop Loss just below 200 EMA
          sl_price = self.ema200[-1] * 0.99

          # Measured Move Target (2x Base Depth)
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


def download_and_prepare_data(ticker="RELIANCE.NS"):
  # Download historical daily data
  df = yf.download(ticker, start="2018-01-01", end="2026-01-01", progress=False)
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  # Resample using modern Pandas offsets ('W' for weekly, 'ME' for month-end)
  df_weekly = df.resample("W").agg({
      "Open": "first",
      "High": "max",
      "Low": "min",
      "Close": "last",
      "Volume": "sum",
  })
  df_monthly = df.resample("ME").agg({
      "Open": "first",
      "High": "max",
      "Low": "min",
      "Close": "last",
      "Volume": "sum",
  })

  df_weekly["Weekly_200EMA"] = ta.ema(df_weekly["Close"], length=200)
  df_monthly["Monthly_200EMA"] = ta.ema(df_monthly["Close"], length=200)

  # Merge back to avoid lookahead bias via forward filling
  df = df.join(df_weekly[["Weekly_200EMA"]]).ffill()
  df = df.join(df_monthly[["Monthly_200EMA"]]).ffill()
  df.dropna(inplace=True)
  return df


if __name__ == "__main__":
  # Run backtest simulation
  data = download_and_prepare_data("RELIANCE.NS")
  bt = Backtest(
      data, InstitutionalBroomBreakout, cash=1000000, commission=0.001
  )
  stats = bt.run()
  print(stats)

  # Export interactive report to HTML for GitHub artifact viewing
  bt.plot(filename="backtest_report.html", open_browser=False)
