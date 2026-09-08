```python
"""
Institutional Broom Breakout Backtest
-------------------------------------

Strategy rules:

1. Macro trend filter:
   Price > Weekly 200 EMA AND Price > Monthly 200 EMA

2. EMA broom compression:
   20/50/100/200 daily EMAs are within 3%

3. Consolidation duration:
   Recent major high occurred 63-147 trading days ago

4. Straight consolidation:
   Last 20 trading days have <8% high-low range

5. Previous trend cap:
   Prior 40-day run-up <=40%

6. Breakout confirmation:
   Price breaks above daily EMAs and Volume Profile POC
   with >=1.5x 20-day average volume.

Risk management:
- Initial SL = 1% below 200 EMA
- TP = current price + 2x measured base depth
- Trailing SL = 1% below 200 EMA

Important:
- Historical warm-up data is downloaded before the actual
  backtest period so that the 200-month EMA can be calculated.
- The backtest itself is restricted to BACKTEST_START/BACKTEST_END.
"""

import warnings

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from backtesting import Backtest, Strategy


# ============================================================
# CONFIGURATION
# ============================================================

TICKER = "RELIANCE.NS"

BACKTEST_START = "2018-01-01"
BACKTEST_END = "2026-01-01"

INITIAL_CASH = 1_000_000
COMMISSION = 0.001

# Daily EMA settings
EMA_FAST = 20
EMA_MEDIUM = 50
EMA_SLOW = 100
EMA_LONG = 200

# Higher timeframe EMA
HTF_EMA_LENGTH = 200

# Strategy thresholds
EMA_COMPRESSION = 0.03
CONSOLIDATION_MIN = 63
CONSOLIDATION_MAX = 147
BOX_LOOKBACK = 20
BOX_MAX_RANGE = 0.08

PRIOR_TREND_LOOKBACK = 40
MAX_PRIOR_RUNUP = 0.40

VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLIER = 1.5

STOP_EMA_BUFFER = 0.99
TARGET_MULTIPLIER = 2.0

POC_BINS = 10


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_poc(
    recent_closes: pd.Series,
    recent_volumes: pd.Series,
    bins: int = POC_BINS,
) -> float:
    """
    Calculate Volume Profile Point of Control (POC).

    POC = price level with the highest accumulated volume.

    Returns NaN when valid data is insufficient.
    """

    prices = pd.to_numeric(recent_closes, errors="coerce")
    volumes = pd.to_numeric(recent_volumes, errors="coerce")

    valid = (
        prices.notna()
        & volumes.notna()
        & np.isfinite(prices)
        & np.isfinite(volumes)
        & (volumes >= 0)
    )

    prices = prices[valid].to_numpy(dtype=float)
    volumes = volumes[valid].to_numpy(dtype=float)

    if len(prices) == 0:
        return np.nan

    price_min = prices.min()
    price_max = prices.max()

    # If every close is identical, there is no meaningful
    # price distribution to bin.
    if not np.isfinite(price_min) or not np.isfinite(price_max):
        return np.nan

    if price_min == price_max:
        return float(price_min)

    # np.linspace creates the price levels used for the
    # volume profile.
    price_bins = np.linspace(
        price_min,
        price_max,
        bins,
    )

    volume_by_price = np.zeros(len(price_bins), dtype=float)

    for price, volume in zip(prices, volumes):
        closest_bin = np.abs(price_bins - price).argmin()
        volume_by_price[closest_bin] += volume

    if not np.any(volume_by_price > 0):
        return np.nan

    poc_index = int(np.argmax(volume_by_price))

    return float(price_bins[poc_index])


# ============================================================
# STRATEGY
# ============================================================

class InstitutionalBroomBreakout(Strategy):
    """
    Institutional Broom Breakout strategy.
    """

    def init(self):

        # ----------------------------------------------------
        # Daily EMAs
        # ----------------------------------------------------

        self.ema20 = self.I(
            ta.ema,
            self.data.Close.s,
            length=EMA_FAST,
        )

        self.ema50 = self.I(
            ta.ema,
            self.data.Close.s,
            length=EMA_MEDIUM,
        )

        self.ema100 = self.I(
            ta.ema,
            self.data.Close.s,
            length=EMA_SLOW,
        )

        self.ema200 = self.I(
            ta.ema,
            self.data.Close.s,
            length=EMA_LONG,
        )

        # ----------------------------------------------------
        # Higher timeframe EMAs
        # ----------------------------------------------------

        self.weekly_200 = self.data.Weekly_200EMA

        self.monthly_200 = self.data.Monthly_200EMA

    def next(self):

        current_bar = len(self.data.Close)

        # ----------------------------------------------------
        # BASIC DATA REQUIREMENT
        # ----------------------------------------------------

        # Need enough data for:
        # - 200 EMA
        # - 200-day consolidation search
        # - 40-day prior trend
        # - 20-day box
        # - 20-day volume profile
        minimum_bars = max(
            EMA_LONG,
            200,
            BOX_LOOKBACK,
            VOLUME_LOOKBACK,
            PRIOR_TREND_LOOKBACK,
        )

        if current_bar < minimum_bars:
            return

        # ----------------------------------------------------
        # CURRENT VALUES
        # ----------------------------------------------------

        current_price = self.data.Close[-1]

        weekly_200 = self.weekly_200[-1]
        monthly_200 = self.monthly_200[-1]

        ema20 = self.ema20[-1]
        ema50 = self.ema50[-1]
        ema100 = self.ema100[-1]
        ema200 = self.ema200[-1]

        # Reject invalid indicator values.
        indicator_values = [
            current_price,
            weekly_200,
            monthly_200,
            ema20,
            ema50,
            ema100,
            ema200,
        ]

        if not all(np.isfinite(value) for value in indicator_values):
            return

        if current_price <= 0:
            return

        # ====================================================
        # RULE 1
        # MACRO TREND FILTER
        # ====================================================

        macro_bullish = (
            current_price > monthly_200
            and current_price > weekly_200
        )

        if not macro_bullish:
            return

        # ====================================================
        # RULE 2
        # EMA BROOM COMPRESSION
        # ====================================================

        current_emas = [
            ema20,
            ema50,
            ema100,
            ema200,
        ]

        ema_spread = (
            max(current_emas) - min(current_emas)
        ) / current_price

        is_ema_compressed = (
            ema_spread < EMA_COMPRESSION
        )

        # ====================================================
        # POSITION MANAGEMENT
        # ====================================================

        if self.position:

            # Trail stop using the 200 EMA.
            new_sl = ema200 * STOP_EMA_BUFFER

            # Current trade's stop.
            trade = self.trades[0]

            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl

            return

        # No position = look for entry.
        if not is_ema_compressed:
            return

        # ====================================================
        # RULE 3
        # CONSOLIDATION DURATION
        # ====================================================

        lookback_window = 200

        recent_highs = np.asarray(
            self.data.High[-lookback_window:],
            dtype=float,
        )

        recent_lows = np.asarray(
            self.data.Low[-lookback_window:],
            dtype=float,
        )

        if len(recent_highs) < lookback_window:
            return

        if not np.all(np.isfinite(recent_highs)):
            return

        if not np.all(np.isfinite(recent_lows)):
            return

        highest_idx = int(
            np.argmax(recent_highs)
        )

        days_since_high = (
            lookback_window - 1 - highest_idx
        )

        is_valid_duration = (
            CONSOLIDATION_MIN
            <= days_since_high
            <= CONSOLIDATION_MAX
        )

        if not is_valid_duration:
            return

        # ====================================================
        # RULE 4
        # STRAIGHT CONSOLIDATION BOX
        # ====================================================

        recent_20_highs = np.asarray(
            self.data.High[-BOX_LOOKBACK:],
            dtype=float,
        )

        recent_20_lows = np.asarray(
            self.data.Low[-BOX_LOOKBACK:],
            dtype=float,
        )

        if len(recent_20_highs) < BOX_LOOKBACK:
            return

        box_high = np.max(recent_20_highs)
        box_low = np.min(recent_20_lows)

        if box_low <= 0:
            return

        box_range = (
            box_high - box_low
        ) / box_low

        is_straight_consolidation = (
            box_range < BOX_MAX_RANGE
        )

        if not is_straight_consolidation:
            return

        # ====================================================
        # RULE 5
        # PREVIOUS TREND CAP
        # ====================================================

        if highest_idx <= PRIOR_TREND_LOOKBACK:
            return

        prior_trend_start = (
            highest_idx - PRIOR_TREND_LOOKBACK
        )

        prior_trend_lows = recent_lows[
            prior_trend_start:highest_idx
        ]

        if len(prior_trend_lows) == 0:
            return

        prior_trend_low = np.min(
            prior_trend_lows
        )

        peak_price = recent_highs[highest_idx]

        if prior_trend_low <= 0:
            return

        run_up_pct = (
            peak_price - prior_trend_low
        ) / prior_trend_low

        is_healthy_trend = (
            run_up_pct <= MAX_PRIOR_RUNUP
        )

        if not is_healthy_trend:
            return

        # ====================================================
        # VOLUME PROFILE
        # ====================================================

        recent_closes = self.data.Close[
            -VOLUME_LOOKBACK:
        ]

        recent_volumes = self.data.Volume[
            -VOLUME_LOOKBACK:
        ]

        if len(recent_closes) < VOLUME_LOOKBACK:
            return

        poc_price = calculate_poc(
            recent_closes,
            recent_volumes,
        )

        if not np.isfinite(poc_price):
            return

        # ====================================================
        # BREAKOUT CONFIRMATION
        # ====================================================

        max_ema = max(current_emas)

        is_breakout = (
            current_price > max_ema
            and current_price > poc_price
        )

        if not is_breakout:
            return

        # ====================================================
        # VOLUME SPIKE
        # ====================================================

        recent_volume_array = np.asarray(
            recent_volumes,
            dtype=float,
        )

        if not np.all(
            np.isfinite(recent_volume_array)
        ):
            return

        average_volume = np.mean(
            recent_volume_array
        )

        current_volume = float(
            self.data.Volume[-1]
        )

        if average_volume <= 0:
            return

        volume_spike = (
            current_volume
            > average_volume
            * VOLUME_SPIKE_MULTIPLIER
        )

        if not volume_spike:
            return

        # ====================================================
        # STOP LOSS
        # ====================================================

        sl_price = (
            ema200 * STOP_EMA_BUFFER
        )

        # Stop must be below current price.
        if sl_price >= current_price:
            return

        # ====================================================
        # MEASURED MOVE TARGET
        # ====================================================

        # Base low over the period since the major high.
        #
        # Include the available consolidation period.
        base_period = recent_lows[
            highest_idx:
        ]

        if len(base_period) == 0:
            return

        base_low = np.min(base_period)

        if base_low <= 0:
            return

        base_depth = (
            peak_price - base_low
        )

        if base_depth <= 0:
            return

        tp_price = (
            current_price
            + TARGET_MULTIPLIER
            * base_depth
        )

        # Target must be above entry.
        if tp_price <= current_price:
            return

        # ====================================================
        # EXECUTION
        # ====================================================

        self.buy(
            sl=float(sl_price),
            tp=float(tp_price),
        )


# ============================================================
# DATA DOWNLOAD AND PREPARATION
# ============================================================

def download_and_prepare_data(
    ticker: str = TICKER,
    backtest_start: str = BACKTEST_START,
    backtest_end: str = BACKTEST_END,
) -> pd.DataFrame:
    """
    Download data and prepare all indicators.

    IMPORTANT:
    We intentionally download maximum available history.

    Why?

    The strategy uses a 200-month EMA.

    200 months ~= 16.7 years.

    If we downloaded only 2018-2026 data, there would not
    be enough monthly observations to calculate the indicator.

    We therefore:

        1. Download maximum available history.
        2. Calculate Weekly 200 EMA.
        3. Calculate Monthly 200 EMA.
        4. Align those indicators to daily data.
        5. Remove the warm-up period.
        6. Keep only the requested backtest period.
    """

    print("=" * 70)
    print("DOWNLOADING DATA")
    print("=" * 70)

    print(f"Ticker       : {ticker}")
    print(f"Backtest     : {backtest_start} -> {backtest_end}")

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    try:
        stock = yf.Ticker(ticker)

        df = stock.history(
            period="max",
            auto_adjust=False,
            actions=False,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Yahoo Finance download failed for {ticker}: {exc}"
        ) from exc

    if df is None or df.empty:
        raise ValueError(
            f"Yahoo Finance returned EMPTY data for {ticker}."
        )

    # --------------------------------------------------------
    # CLEAN INDEX
    # --------------------------------------------------------

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()

    # Remove duplicate dates.
    df = df[~df.index.duplicated(keep="last")]

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Yahoo Finance data is missing columns: "
            f"{missing_columns}"
        )

    df = df[required_columns].copy()

    # --------------------------------------------------------
    # NUMERIC CLEANING
    # --------------------------------------------------------

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ],
        inplace=True,
    )

    # --------------------------------------------------------
    # WEEKLY DATA
    # --------------------------------------------------------

    print("\nCalculating Weekly 200 EMA...")

    df_weekly = df.resample("W").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    df_weekly.dropna(
        subset=["Close"],
        inplace=True,
    )

    df_weekly["Weekly_200EMA"] = ta.ema(
        df_weekly["Close"],
        length=HTF_EMA_LENGTH,
    )

    # --------------------------------------------------------
    # MONTHLY DATA
    # --------------------------------------------------------

    print("Calculating Monthly 200 EMA...")

    df_monthly = df.resample("ME").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    df_monthly.dropna(
        subset=["Close"],
        inplace=True,
    )

    df_monthly["Monthly_200EMA"] = ta.ema(
        df_monthly["Close"],
        length=HTF_EMA_LENGTH,
    )

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    valid_weekly = (
        df_weekly["Weekly_200EMA"]
        .notna()
        .sum()
    )

    valid_monthly = (
        df_monthly["Monthly_200EMA"]
        .notna()
        .sum()
    )

    print(
        f"Weekly 200 EMA valid observations : {valid_weekly}"
    )

    print(
        f"Monthly 200 EMA valid observations: {valid_monthly}"
    )

    if valid_monthly == 0:
        raise ValueError(
            "Monthly 200 EMA could not be calculated. "
            "Yahoo Finance did not provide enough historical "
            "monthly data."
        )

    # --------------------------------------------------------
    # ALIGN HIGHER TIMEFRAME INDICATORS
    # --------------------------------------------------------

    # IMPORTANT:
    #
    # Monthly EMA is only known after the monthly candle closes.
    #
    # By joining month-end values to daily data and forward
    # filling, the completed monthly EMA becomes available
    # from the following daily observations.
    #
    # This prevents using a future monthly EMA value.
    #
    # Same concept applies to Weekly EMA.

    df["Weekly_200EMA"] = (
        df_weekly["Weekly_200EMA"]
        .reindex(df.index)
        .ffill()
    )

    df["Monthly_200EMA"] = (
        df_monthly["Monthly_200EMA"]
        .reindex(df.index)
        .ffill()
    )

    # --------------------------------------------------------
    # DAILY EMAS
    # --------------------------------------------------------

    # Calculate these here as a diagnostic/reference.
    # The actual Strategy calculates them again through
    # self.I(), which is required by backtesting.py.
    df["EMA20"] = ta.ema(
        df["Close"],
        length=EMA_FAST,
    )

    df["EMA50"] = ta.ema(
        df["Close"],
        length=EMA_MEDIUM,
    )

    df["EMA100"] = ta.ema(
        df["Close"],
        length=EMA_SLOW,
    )

    df["EMA200"] = ta.ema(
        df["Close"],
        length=EMA_LONG,
    )

    # --------------------------------------------------------
    # CHECK BEFORE BACKTEST PERIOD
    # --------------------------------------------------------

    # DO NOT use dropna() on the complete dataset here.
    #
    # That could accidentally delete the entire dataset because
    # of an indicator that is still warming up.
    #
    # Instead, select the actual backtest period first.

    backtest_start_ts = pd.Timestamp(
        backtest_start
    )

    backtest_end_ts = pd.Timestamp(
        backtest_end
    )

    df = df.loc[
        (df.index >= backtest_start_ts)
        & (df.index < backtest_end_ts)
    ].copy()

    # --------------------------------------------------------
    # FINAL DATA CLEANING
    # --------------------------------------------------------

    final_required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Weekly_200EMA",
        "Monthly_200EMA",
    ]

    df.dropna(
        subset=final_required_columns,
        inplace=True,
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if df.empty:
        raise ValueError(
            "FINAL DATAFRAME IS EMPTY after data preparation.\n"
            "Possible reasons:\n"
            "1. Insufficient Yahoo Finance history.\n"
            "2. Requested backtest period is unavailable.\n"
            "3. Higher timeframe EMA could not be aligned."
        )

    if len(df) < 250:
        raise ValueError(
            f"Only {len(df)} daily rows are available for the "
            "backtest. At least 250 rows are recommended."
        )

    # Check OHLC integrity.
    invalid_ohlc = (
        (df["High"] < df["Low"])
        | (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
    )

    if invalid_ohlc.any():
        bad_rows = int(invalid_ohlc.sum())

        raise ValueError(
            f"Found {bad_rows} rows with invalid OHLC data."
        )

    # --------------------------------------------------------
    # REMOVE EXTRA DIAGNOSTIC EMA COLUMNS
    # --------------------------------------------------------

    df.drop(
        columns=[
            "EMA20",
            "EMA50",
            "EMA100",
            "EMA200",
        ],
        inplace=True,
        errors="ignore",
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA READY")
    print("=" * 70)

    print(
        f"Rows         : {len(df):,}"
    )

    print(
        f"First date   : {df.index.min().date()}"
    )

    print(
        f"Last date    : {df.index.max().date()}"
    )

    print(
        f"Monthly EMA  : {df['Monthly_200EMA'].notna().sum():,} valid"
    )

    print(
        f"Weekly EMA   : {df['Weekly_200EMA'].notna().sum():,} valid"
    )

    print("=" * 70)

    return df


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    ticker: str = TICKER,
) -> None:
    """
    Run the complete backtest.
    """

    data = download_and_prepare_data(
        ticker=ticker,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
    )

    # --------------------------------------------------------
    # FINAL SAFETY CHECK
    # --------------------------------------------------------

    if data.empty:
        raise ValueError(
            "Cannot start backtest because OHLC data is empty."
        )

    ohlc_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_ohlc = [
        column
        for column in ohlc_columns
        if column not in data.columns
    ]

    if missing_ohlc:
        raise ValueError(
            f"Missing OHLC columns: {missing_ohlc}"
        )

    # --------------------------------------------------------
    # CREATE BACKTEST
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING BACKTEST")
    print("=" * 70)

    bt = Backtest(
        data,
        InstitutionalBroomBreakout,
        cash=INITIAL_CASH,
        commission=COMMISSION,
        exclusive_orders=True,
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    stats = bt.run()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(stats)

    print("=" * 70)

    # --------------------------------------------------------
    # EXPORT HTML REPORT
    # --------------------------------------------------------

    print("\nGenerating HTML backtest report...")

    try:

        bt.plot(
            filename="backtest_report.html",
            open_browser=False,
        )

        print(
            "HTML report created: backtest_report.html"
        )

    except Exception as exc:

        # Plotting failure should not hide the actual backtest
        # result.
        warnings.warn(
            f"Backtest completed, but HTML plot generation "
            f"failed: {exc}"
        )

    print("\nBacktest completed successfully.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        run_backtest(
            ticker=TICKER,
        )

    except Exception as exc:

        print("\n" + "=" * 70)
        print("BACKTEST FAILED")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print("=" * 70)

        # Non-zero exit code makes GitHub Actions correctly
        # mark the workflow as failed.
        raise
```
