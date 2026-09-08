import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BACKTEST_START = "2018-01-01"
BACKTEST_END = "2026-01-01"

INITIAL_CAPITAL = 1_000_000

COMMISSION = 0.001

# Daily EMAs
EMA_PERIODS = [20, 50, 100, 200]

# Macro trend
WEEKLY_EMA_PERIOD = 200
MONTHLY_EMA_PERIOD = 200

# Consolidation
BOX_LOOKBACK = 20
MAX_BOX_RANGE = 0.08

# Previous peak
PEAK_LOOKBACK = 200
MIN_DAYS_SINCE_PEAK = 63
MAX_DAYS_SINCE_PEAK = 147

# Prior trend
PRIOR_TREND_LOOKBACK = 40
MAX_PRIOR_RUNUP = 0.40

# Volume
VOLUME_LOOKBACK = 20
VOLUME_MULTIPLIER = 1.50

# Stop / target
STOP_EMA_MULTIPLIER = 0.99
TARGET_MULTIPLIER = 2.0

# Breakout
BREAKOUT_BUFFER = 0.0

# Yahoo download
MAX_WORKERS = 8
RETRY_COUNT = 3
RETRY_SLEEP = 2

# Minimum number of daily observations
MIN_DAILY_ROWS = 300


# ============================================================
# NIFTY 500
# ============================================================

NIFTY500_URL = (
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
)


def get_nifty500_symbols():

    print("\nDownloading Nifty 500 constituent list...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/",
    }

    response = requests.get(
        NIFTY500_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(response.text))

    if "Symbol" not in df.columns:
        raise ValueError(
            "Nifty 500 CSV downloaded but Symbol column was not found."
        )

    symbols = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    if len(symbols) < 400:
        raise ValueError(
            f"Only {len(symbols)} Nifty symbols were downloaded. "
            "NSE data may have blocked the request."
        )

    yahoo_symbols = [symbol + ".NS" for symbol in symbols]

    print(f"Nifty 500 symbols loaded: {len(yahoo_symbols)}")

    return yahoo_symbols


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_stock(ticker):

    for attempt in range(1, RETRY_COUNT + 1):

        try:

            df = yf.download(
                ticker,
                period="max",
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False
            )

            if df is None or df.empty:
                return None, "EMPTY"

            # Handle possible MultiIndex from Yahoo
            if isinstance(df.columns, pd.MultiIndex):

                # Try to extract ticker level
                if ticker in df.columns.get_level_values(-1):
                    try:
                        df = df.xs(
                            ticker,
                            axis=1,
                            level=-1
                        )
                    except Exception:
                        pass

                elif ticker in df.columns.get_level_values(0):
                    try:
                        df = df.xs(
                            ticker,
                            axis=1,
                            level=0
                        )
                    except Exception:
                        pass

            required = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]

            missing = [
                c for c in required
                if c not in df.columns
            ]

            if missing:
                return None, "MISSING_COLUMNS"

            df = df[required].copy()

            # Remove timezone
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)

            df = df.sort_index()

            # Numeric conversion
            for col in required:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df = df.dropna(
                subset=[
                    "Open",
                    "High",
                    "Low",
                    "Close"
                ]
            )

            if len(df) < MIN_DAILY_ROWS:
                return df, "INSUFFICIENT_HISTORY"

            return df, "OK"

        except Exception as e:

            if attempt < RETRY_COUNT:
                time.sleep(RETRY_SLEEP * attempt)
            else:
                return None, f"ERROR: {str(e)[:100]}"

    return None, "UNKNOWN"


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # Daily EMAs
    # --------------------------------------------------------

    for period in EMA_PERIODS:

        df[f"EMA{period}"] = (
            df["Close"]
            .ewm(
                span=period,
                adjust=False,
                min_periods=period
            )
            .mean()
        )

    # --------------------------------------------------------
    # Weekly data
    # --------------------------------------------------------

    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    weekly["Weekly_200EMA"] = (
        weekly["Close"]
        .ewm(
            span=WEEKLY_EMA_PERIOD,
            adjust=False,
            min_periods=WEEKLY_EMA_PERIOD
        )
        .mean()
    )

    # --------------------------------------------------------
    # Monthly data
    # --------------------------------------------------------

    monthly = df.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    monthly["Monthly_200EMA"] = (
        monthly["Close"]
        .ewm(
            span=MONTHLY_EMA_PERIOD,
            adjust=False,
            min_periods=MONTHLY_EMA_PERIOD
        )
        .mean()
    )

    # --------------------------------------------------------
    # Align HTF EMA to daily data
    # --------------------------------------------------------

    df["Weekly_200EMA"] = (
        weekly["Weekly_200EMA"]
        .reindex(df.index)
        .ffill()
    )

    df["Monthly_200EMA"] = (
        monthly["Monthly_200EMA"]
        .reindex(df.index)
        .ffill()
    )

    return df


# ============================================================
# POINT OF CONTROL
# ============================================================

def calculate_poc(closes, volumes, bins=10):

    closes = np.asarray(
        closes,
        dtype=float
    )

    volumes = np.asarray(
        volumes,
        dtype=float
    )

    mask = (
        np.isfinite(closes)
        & np.isfinite(volumes)
        & (volumes >= 0)
    )

    closes = closes[mask]
    volumes = volumes[mask]

    if len(closes) == 0:
        return np.nan

    price_min = np.min(closes)
    price_max = np.max(closes)

    if not np.isfinite(price_min):
        return np.nan

    if price_min == price_max:
        return price_min

    price_bins = np.linspace(
        price_min,
        price_max,
        bins
    )

    volume_by_price = np.zeros(
        bins,
        dtype=float
    )

    for price, volume in zip(
        closes,
        volumes
    ):

        index = np.abs(
            price_bins - price
        ).argmin()

        volume_by_price[index] += volume

    return price_bins[
        np.argmax(volume_by_price)
    ]


# ============================================================
# SIGNAL DIAGNOSTICS
# ============================================================

def evaluate_signal(df, i):

    result = {
        "macro": False,
        "ema_compression": False,
        "peak_duration": False,
        "box_consolidation": False,
        "healthy_runup": False,
        "breakout_above_box": False,
        "breakout_above_ema": False,
        "above_poc": False,
        "volume_spike": False,
        "final_signal": False
    }

    if i < max(
        PEAK_LOOKBACK,
        BOX_LOOKBACK + 1,
        VOLUME_LOOKBACK + 1,
        200
    ):
        return result

    row = df.iloc[i]

    current_price = row["Close"]

    # ========================================================
    # 1. MACRO TREND
    # ========================================================

    if (
        pd.notna(row["Monthly_200EMA"])
        and pd.notna(row["Weekly_200EMA"])
    ):

        result["macro"] = (
            current_price > row["Monthly_200EMA"]
            and
            current_price > row["Weekly_200EMA"]
        )

    if not result["macro"]:
        return result

    # ========================================================
    # 2. EMA COMPRESSION
    # ========================================================

    emas = [
        row["EMA20"],
        row["EMA50"],
        row["EMA100"],
        row["EMA200"]
    ]

    if not any(pd.isna(x) for x in emas):

        spread = (
            max(emas) - min(emas)
        ) / current_price

        result["ema_compression"] = (
            spread < 0.03
        )

    if not result["ema_compression"]:
        return result

    # ========================================================
    # 3. PREVIOUS 200 DAYS PEAK
    #
    # IMPORTANT:
    # Current breakout day is excluded.
    # ========================================================

    peak_start = i - PEAK_LOOKBACK
    peak_end = i

    peak_window = df.iloc[
        peak_start:peak_end
    ]

    if len(peak_window) < PEAK_LOOKBACK:
        return result

    highest_position = (
        peak_window["High"]
        .values
        .argmax()
    )

    peak_price = (
        peak_window["High"]
        .iloc[highest_position]
    )

    peak_absolute_index = (
        peak_start + highest_position
    )

    days_since_peak = (
        i - 1 - peak_absolute_index
    )

    result["peak_duration"] = (
        MIN_DAYS_SINCE_PEAK
        <= days_since_peak
        <= MAX_DAYS_SINCE_PEAK
    )

    if not result["peak_duration"]:
        return result

    # ========================================================
    # 4. PREVIOUS 20-DAY CONSOLIDATION
    #
    # CRITICAL:
    # The breakout day is NOT included.
    # ========================================================

    box_start = i - BOX_LOOKBACK - 1
    box_end = i - 1

    box = df.iloc[
        box_start:box_end
    ]

    if len(box) < BOX_LOOKBACK:
        return result

    box_high = box["High"].max()
    box_low = box["Low"].min()

    if box_low <= 0:
        return result

    box_range = (
        box_high - box_low
    ) / box_low

    result["box_consolidation"] = (
        box_range < MAX_BOX_RANGE
    )

    if not result["box_consolidation"]:
        return result

    # ========================================================
    # 5. HEALTHY PRIOR RUN-UP
    # ========================================================

    if highest_position >= PRIOR_TREND_LOOKBACK:

        prior_start = (
            peak_absolute_index
            - PRIOR_TREND_LOOKBACK
        )

        prior_end = peak_absolute_index

        prior_low = df.iloc[
            prior_start:prior_end
        ]["Low"].min()

        if prior_low > 0:

            runup = (
                peak_price - prior_low
            ) / prior_low

            result["healthy_runup"] = (
                runup <= MAX_PRIOR_RUNUP
            )

    if not result["healthy_runup"]:
        return result

    # ========================================================
    # 6. BREAKOUT ABOVE CONSOLIDATION
    # ========================================================

    result["breakout_above_box"] = (
        current_price
        >
        box_high * (1 + BREAKOUT_BUFFER)
    )

    # ========================================================
    # 7. BREAKOUT ABOVE DAILY EMAs
    # ========================================================

    result["breakout_above_ema"] = (
        current_price > row["EMA20"]
        and
        current_price > row["EMA50"]
        and
        current_price > row["EMA100"]
        and
        current_price > row["EMA200"]
    )

    # ========================================================
    # 8. POC
    #
    # Previous 20 days only.
    # ========================================================

    poc = calculate_poc(
        box["Close"].values,
        box["Volume"].values
    )

    if pd.notna(poc):

        result["above_poc"] = (
            current_price > poc
        )

    # ========================================================
    # 9. VOLUME
    #
    # Previous 20 days only.
    # Current breakout volume excluded.
    # ========================================================

    average_volume = box["Volume"].mean()

    if average_volume > 0:

        result["volume_spike"] = (
            row["Volume"]
            >
            average_volume * VOLUME_MULTIPLIER
        )

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    result["final_signal"] = (
        result["breakout_above_box"]
        and
        result["breakout_above_ema"]
        and
        result["above_poc"]
        and
        result["volume_spike"]
    )

    return result


# ============================================================
# BACKTEST ONE STOCK
# ============================================================

def backtest_stock(ticker, raw_df):

    diagnostics = {
        "Ticker": ticker,
        "Raw_Rows": len(raw_df),
        "Rows_After_Indicators": 0,
        "Macro_Pass": 0,
        "EMA_Compression_Pass": 0,
        "Peak_Duration_Pass": 0,
        "Box_Pass": 0,
        "Healthy_Runup_Pass": 0,
        "Breakout_Box_Pass": 0,
        "Breakout_EMA_Pass": 0,
        "POC_Pass": 0,
        "Volume_Pass": 0,
        "Final_Signals": 0
    }

    df = calculate_indicators(
        raw_df
    )

    # Restrict actual backtest period AFTER
    # calculating long-term indicators.
    df = df.loc[
        BACKTEST_START:BACKTEST_END
    ].copy()

    diagnostics[
        "Rows_After_Indicators"
    ] = len(df)

    if len(df) < 300:

        return (
            None,
            [],
            diagnostics
        )

    # --------------------------------------------------------
    # Equity
    # --------------------------------------------------------

    cash = float(INITIAL_CAPITAL)

    position = False

    shares = 0

    entry_price = np.nan
    entry_date = None

    stop_price = np.nan
    target_price = np.nan

    peak_price_for_trade = np.nan

    trades = []

    equity_curve = []

    pending_entry = False

    pending_entry_signal_index = None

    # ========================================================
    # DAILY LOOP
    # ========================================================

    for i in range(1, len(df)):

        row = df.iloc[i]

        current_date = df.index[i]

        # ====================================================
        # FIRST: HANDLE EXISTING POSITION
        # ====================================================

        if position:

            exit_price = None
            exit_reason = None

            # -----------------------------------------------
            # Stop
            # -----------------------------------------------

            if row["Low"] <= stop_price:

                exit_price = stop_price
                exit_reason = "STOP"

            # -----------------------------------------------
            # Target
            # -----------------------------------------------

            elif row["High"] >= target_price:

                exit_price = target_price
                exit_reason = "TARGET"

            # -----------------------------------------------
            # Trailing stop
            # -----------------------------------------------

            new_stop = (
                row["EMA200"]
                * STOP_EMA_MULTIPLIER
            )

            if pd.notna(new_stop):

                if new_stop > stop_price:

                    stop_price = new_stop

            # -----------------------------------------------
            # Check stop again after trailing
            # -----------------------------------------------

            if (
                exit_price is None
                and
                row["Low"] <= stop_price
            ):

                exit_price = stop_price
                exit_reason = "TRAILING_STOP"

            # -----------------------------------------------
            # EXIT
            # -----------------------------------------------

            if exit_price is not None:

                gross_value = (
                    shares * exit_price
                )

                sell_commission = (
                    gross_value * COMMISSION
                )

                cash = (
                    gross_value
                    - sell_commission
                )

                gross_pnl = (
                    exit_price - entry_price
                ) * shares

                buy_commission = (
                    shares
                    * entry_price
                    * COMMISSION
                )

                net_pnl = (
                    gross_pnl
                    - buy_commission
                    - sell_commission
                )

                return_pct = (
                    net_pnl
                    /
                    (
                        shares
                        * entry_price
                        + buy_commission
                    )
                ) * 100

                trades.append({
                    "Ticker": ticker,
                    "Entry_Date": entry_date,
                    "Entry_Price": entry_price,
                    "Exit_Date": current_date,
                    "Exit_Price": exit_price,
                    "Shares": shares,
                    "Stop_Initial": (
                        entry_price
                        if pd.isna(stop_price)
                        else stop_price
                    ),
                    "Target": target_price,
                    "Exit_Reason": exit_reason,
                    "Net_PnL": net_pnl,
                    "Return_Pct": return_pct
                })

                position = False
                shares = 0
                entry_price = np.nan
                entry_date = None
                stop_price = np.nan
                target_price = np.nan

        # ====================================================
        # SECOND: ENTRY
        #
        # Signal generated at previous close.
        # Enter at today's open.
        # ====================================================

        if (
            not position
            and
            pending_entry
            and
            pending_entry_signal_index == i - 1
        ):

            entry_price_today = row["Open"]

            if (
                pd.notna(entry_price_today)
                and
                entry_price_today > 0
            ):

                # -------------------------------------------
                # Calculate previous signal day's EMA200
                # -------------------------------------------

                signal_row = df.iloc[i - 1]

                sl = (
                    signal_row["EMA200"]
                    * STOP_EMA_MULTIPLIER
                )

                # Reconstruct peak/base
                peak_window = df.iloc[
                    max(
                        0,
                        i - 1 - PEAK_LOOKBACK
                    ):
                    i - 1
                ]

                if len(peak_window) >= PEAK_LOOKBACK:

                    peak_position = (
                        peak_window["High"]
                        .values
                        .argmax()
                    )

                    peak_price = (
                        peak_window["High"]
                        .iloc[peak_position]
                    )

                    peak_absolute = (
                        max(
                            0,
                            i - 1 - PEAK_LOOKBACK
                        )
                        + peak_position
                    )

                    base_low = df.iloc[
                        peak_absolute:i - 1
                    ]["Low"].min()

                    if (
                        pd.notna(base_low)
                        and
                        base_low > 0
                    ):

                        base_depth = (
                            peak_price - base_low
                        )

                        target = (
                            entry_price_today
                            +
                            TARGET_MULTIPLIER
                            * base_depth
                        )

                        # -----------------------------------
                        # Position sizing:
                        # Use entire capital.
                        # -----------------------------------

                        buy_commission_est = (
                            COMMISSION
                            * cash
                        )

                        available_cash = (
                            cash
                            /
                            (
                                1 + COMMISSION
                            )
                        )

                        shares_to_buy = int(
                            available_cash
                            /
                            entry_price_today
                        )

                        if shares_to_buy > 0:

                            purchase_value = (
                                shares_to_buy
                                * entry_price_today
                            )

                            commission_paid = (
                                purchase_value
                                * COMMISSION
                            )

                            total_cost = (
                                purchase_value
                                + commission_paid
                            )

                            if total_cost <= cash:

                                cash -= total_cost

                                shares = (
                                    shares_to_buy
                                )

                                entry_price = (
                                    entry_price_today
                                )

                                entry_date = (
                                    current_date
                                )

                                stop_price = sl

                                target_price = target

                                position = True

            pending_entry = False
            pending_entry_signal_index = None

        # ====================================================
        # THIRD: GENERATE NEW SIGNAL AT TODAY'S CLOSE
        # ====================================================

        if not position:

            signal = evaluate_signal(
                df,
                i
            )

            # Diagnostics
            if signal["macro"]:
                diagnostics["Macro_Pass"] += 1

            if signal["ema_compression"]:
                diagnostics[
                    "EMA_Compression_Pass"
                ] += 1

            if signal["peak_duration"]:
                diagnostics[
                    "Peak_Duration_Pass"
                ] += 1

            if signal["box_consolidation"]:
                diagnostics[
                    "Box_Pass"
                ] += 1

            if signal["healthy_runup"]:
                diagnostics[
                    "Healthy_Runup_Pass"
                ] += 1

            if signal["breakout_above_box"]:
                diagnostics[
                    "Breakout_Box_Pass"
                ] += 1

            if signal["breakout_above_ema"]:
                diagnostics[
                    "Breakout_EMA_Pass"
                ] += 1

            if signal["above_poc"]:
                diagnostics[
                    "POC_Pass"
                ] += 1

            if signal["volume_spike"]:
                diagnostics[
                    "Volume_Pass"
                ] += 1

            if signal["final_signal"]:

                diagnostics[
                    "Final_Signals"
                ] += 1

                pending_entry = True
                pending_entry_signal_index = i

        # ====================================================
        # EQUITY CURVE
        # ====================================================

        if position:

            market_value = (
                shares * row["Close"]
            )

            equity = (
                cash + market_value
            )

        else:

            equity = cash

        equity_curve.append({
            "Date": current_date,
            "Equity": equity
        })

    # ========================================================
    # CLOSE OPEN POSITION AT LAST CLOSE
    # ========================================================

    if position and len(df) > 0:

        final_row = df.iloc[-1]

        exit_price = final_row["Close"]

        gross_value = (
            shares * exit_price
        )

        sell_commission = (
            gross_value * COMMISSION
        )

        cash = (
            gross_value
            - sell_commission
        )

        gross_pnl = (
            exit_price - entry_price
        ) * shares

        buy_commission = (
            shares
            * entry_price
            * COMMISSION
        )

        net_pnl = (
            gross_pnl
            - buy_commission
            - sell_commission
        )

        return_pct = (
            net_pnl
            /
            (
                shares
                * entry_price
                + buy_commission
            )
        ) * 100

        trades.append({
            "Ticker": ticker,
            "Entry_Date": entry_date,
            "Entry_Price": entry_price,
            "Exit_Date": df.index[-1],
            "Exit_Price": exit_price,
            "Shares": shares,
            "Stop_Initial": stop_price,
            "Target": target_price,
            "Exit_Reason": "END_OF_TEST",
            "Net_PnL": net_pnl,
            "Return_Pct": return_pct
        })

    # ========================================================
    # SUMMARY
    # ========================================================

    final_equity = cash

    if trades:

        total_pnl = sum(
            t["Net_PnL"]
            for t in trades
        )

        winning = [
            t for t in trades
            if t["Net_PnL"] > 0
        ]

        losing = [
            t for t in trades
            if t["Net_PnL"] <= 0
        ]

        win_rate = (
            len(winning)
            /
            len(trades)
            * 100
        )

        returns = [
            t["Return_Pct"]
            for t in trades
        ]

        best_trade = max(returns)
        worst_trade = min(returns)

    else:

        total_pnl = 0
        win_rate = 0
        best_trade = 0
        worst_trade = 0

    # ========================================================
    # MAX DRAWDOWN
    # ========================================================

    if equity_curve:

        eq = pd.Series(
            [
                x["Equity"]
                for x in equity_curve
            ]
        )

        rolling_max = eq.cummax()

        drawdown = (
            eq / rolling_max - 1
        )

        max_drawdown = (
            drawdown.min() * 100
        )

    else:

        max_drawdown = 0

    summary = {
        "Ticker": ticker,
        "Trades": len(trades),
        "Winning_Trades": sum(
            1
            for t in trades
            if t["Net_PnL"] > 0
        ),
        "Losing_Trades": sum(
            1
            for t in trades
            if t["Net_PnL"] <= 0
        ),
        "Win_Rate_Pct": win_rate,
        "Total_PnL": total_pnl,
        "Final_Equity": final_equity,
        "Total_Return_Pct": (
            (final_equity / INITIAL_CAPITAL - 1)
            * 100
        ),
        "Max_Drawdown_Pct": max_drawdown,
        "Best_Trade_Pct": best_trade,
        "Worst_Trade_Pct": worst_trade
    }

    return (
        summary,
        trades,
        diagnostics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NIFTY 500 INSTITUTIONAL BROOM BREAKOUT BACKTEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------

    tickers = get_nifty500_symbols()

    print(
        f"\nBacktest period: "
        f"{BACKTEST_START} → {BACKTEST_END}"
    )

    print(
        f"Universe: {len(tickers)} stocks"
    )

    summaries = []
    all_trades = []
    diagnostics = []

    loaded = 0
    failed = 0

    # --------------------------------------------------------
    # Process each stock
    # --------------------------------------------------------

    for number, ticker in enumerate(
        tickers,
        start=1
    ):

        print(
            f"[{number}/{len(tickers)}] "
            f"{ticker}",
            end=" "
        )

        df, status = download_stock(
            ticker
        )

        if df is None:

            failed += 1

            print(
                f"→ FAILED ({status})"
            )

            diagnostics.append({
                "Ticker": ticker,
                "Raw_Rows": 0,
                "Rows_After_Indicators": 0,
                "Macro_Pass": 0,
                "EMA_Compression_Pass": 0,
                "Peak_Duration_Pass": 0,
                "Box_Pass": 0,
                "Healthy_Runup_Pass": 0,
                "Breakout_Box_Pass": 0,
                "Breakout_EMA_Pass": 0,
                "POC_Pass": 0,
                "Volume_Pass": 0,
                "Final_Signals": 0,
                "Download_Status": status
            })

            continue

        loaded += 1

        try:

            summary, trades, diag = (
                backtest_stock(
                    ticker,
                    df
                )
            )

            diag["Download_Status"] = status

            diagnostics.append(
                diag
            )

            if summary is not None:

                summaries.append(
                    summary
                )

            if trades:

                all_trades.extend(
                    trades
                )

                print(
                    f"→ OK | "
                    f"Trades: {len(trades)}"
                )

            else:

                print(
                    "→ OK | Trades: 0"
                )

        except Exception as e:

            failed += 1

            print(
                f"→ BACKTEST ERROR: "
                f"{str(e)[:100]}"
            )

            diagnostics.append({
                "Ticker": ticker,
                "Raw_Rows": len(df),
                "Rows_After_Indicators": 0,
                "Macro_Pass": 0,
                "EMA_Compression_Pass": 0,
                "Peak_Duration_Pass": 0,
                "Box_Pass": 0,
                "Healthy_Runup_Pass": 0,
                "Breakout_Box_Pass": 0,
                "Breakout_EMA_Pass": 0,
                "POC_Pass": 0,
                "Volume_Pass": 0,
                "Final_Signals": 0,
                "Download_Status": f"BACKTEST_ERROR: {str(e)}"
            })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        summaries
    )

    trades_df = pd.DataFrame(
        all_trades
    )

    diagnostics_df = pd.DataFrame(
        diagnostics
    )

    if not results_df.empty:

        results_df = results_df.sort_values(
            "Total_Return_Pct",
            ascending=False
        )

    if not trades_df.empty:

        trades_df = trades_df.sort_values(
            "Entry_Date"
        )

    diagnostics_df.to_csv(
        "nifty500_diagnostics.csv",
        index=False
    )

    results_df.to_csv(
        "nifty500_backtest_results.csv",
        index=False
    )

    trades_df.to_csv(
        "nifty500_trades.csv",
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    print(
        f"Total Nifty 500 stocks : {len(tickers)}"
    )

    print(
        f"Successfully downloaded : {loaded}"
    )

    print(
        f"Failed downloads/errors : {failed}"
    )

    print(
        f"Stocks with results      : {len(results_df)}"
    )

    print(
        f"Stocks with trades       : "
        f"{sum(results_df['Trades'] > 0) if not results_df.empty else 0}"
    )

    print(
        f"TOTAL TRADES             : "
        f"{len(trades_df)}"
    )

    # --------------------------------------------------------
    # Diagnostic totals
    # --------------------------------------------------------

    if not diagnostics_df.empty:

        print("\nFILTER DIAGNOSTICS")
        print("-" * 70)

        diagnostic_columns = [
            "Macro_Pass",
            "EMA_Compression_Pass",
            "Peak_Duration_Pass",
            "Box_Pass",
            "Healthy_Runup_Pass",
            "Breakout_Box_Pass",
            "Breakout_EMA_Pass",
            "POC_Pass",
            "Volume_Pass",
            "Final_Signals"
        ]

        for col in diagnostic_columns:

            print(
                f"{col:<28} "
                f"{diagnostics_df[col].sum():>8}"
            )

    # --------------------------------------------------------
    # Top stocks
    # --------------------------------------------------------

    if not results_df.empty:

        print("\nTOP 20 STOCKS BY RETURN")
        print("-" * 70)

        print(
            results_df[
                [
                    "Ticker",
                    "Trades",
                    "Win_Rate_Pct",
                    "Total_Return_Pct",
                    "Max_Drawdown_Pct"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Trade list
    # --------------------------------------------------------

    if not trades_df.empty:

        print("\nALL TRADES")
        print("-" * 70)

        print(
            trades_df[
                [
                    "Ticker",
                    "Entry_Date",
                    "Entry_Price",
                    "Exit_Date",
                    "Exit_Price",
                    "Exit_Reason",
                    "Net_PnL",
                    "Return_Pct"
                ]
            ]
            .to_string(index=False)
        )

    else:

        print(
            "\nNO TRADES GENERATED."
        )

    print("\nFiles created:")
    print("1. nifty500_backtest_results.csv")
    print("2. nifty500_trades.csv")
    print("3. nifty500_diagnostics.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
