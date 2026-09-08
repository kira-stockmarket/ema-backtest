import numpy as np
import pandas as pd
import yfinance as yf
import requests
import io
import time

from backtesting import Backtest, Strategy


# ============================================================
# CONFIGURATION
# ============================================================

BACKTEST_START = "2018-01-01"
BACKTEST_END = "2026-01-01"

INITIAL_CASH_PER_STOCK = 1_000_000
COMMISSION = 0.001

# Strategy parameters
EMA_COMPRESSION = 0.03
CONSOLIDATION_MIN_DAYS = 63
CONSOLIDATION_MAX_DAYS = 147

BOX_LOOKBACK = 20
MAX_BOX_RANGE = 0.08

MAX_PRIOR_RUNUP = 0.40

VOLUME_MULTIPLIER = 1.50

STOP_EMA_MULTIPLIER = 0.99
TARGET_MULTIPLIER = 2.0

# Small breakout confirmation buffer.
# 0.0 = close simply has to exceed previous 20-day high.
BREAKOUT_BUFFER = 0.0

# Download settings
DOWNLOAD_PERIOD = "max"
DOWNLOAD_THREADS = 8

NIFTY500_URL = (
    "https://archives.nseindia.com/"
    "content/indices/ind_nifty500list.csv"
)


# ============================================================
# NIFTY 500 UNIVERSE
# ============================================================

def get_nifty500_symbols():

    print("=" * 70)
    print("DOWNLOADING CURRENT NIFTY 500 UNIVERSE")
    print("=" * 70)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,text/plain,*/*",
        "Referer": "https://www.niftyindices.com/"
    }

    response = requests.get(
        NIFTY500_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(
            response.content.decode(
                "utf-8-sig"
            )
        )
    )

    # NSE normally provides a column called Symbol.
    symbol_column = None

    for column in df.columns:

        if str(column).strip().lower() == "symbol":
            symbol_column = column
            break

    if symbol_column is None:

        raise ValueError(
            "Could not find SYMBOL column "
            "in Nifty 500 CSV.\n"
            f"Columns received: {list(df.columns)}"
        )

    symbols = (
        df[symbol_column]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    # Convert NSE symbols to Yahoo Finance symbols.
    tickers = [
        f"{symbol}.NS"
        for symbol in symbols
        if symbol
    ]

    print(
        f"Nifty 500 symbols found: {len(tickers)}"
    )

    if len(tickers) < 400:

        raise ValueError(
            "Nifty 500 download returned "
            f"only {len(tickers)} symbols. "
            "Refusing to continue because "
            "the universe appears incomplete."
        )

    return tickers


# ============================================================
# VOLUME PROFILE POC
# ============================================================

def calculate_poc(
    recent_closes,
    recent_volumes,
    bins=10
):

    prices = np.asarray(
        recent_closes,
        dtype=float
    )

    volumes = np.asarray(
        recent_volumes,
        dtype=float
    )

    valid = (
        np.isfinite(prices)
        &
        np.isfinite(volumes)
    )

    prices = prices[valid]
    volumes = volumes[valid]

    if len(prices) == 0:
        return np.nan

    price_min = float(
        np.min(prices)
    )

    price_max = float(
        np.max(prices)
    )

    if np.isclose(
        price_min,
        price_max
    ):
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
        prices,
        volumes
    ):

        index = int(
            np.abs(
                price_bins - price
            ).argmin()
        )

        volume_by_price[index] += volume

    poc_index = int(
        np.argmax(
            volume_by_price
        )
    )

    return float(
        price_bins[poc_index]
    )


# ============================================================
# EMA FUNCTION
# ============================================================

def calculate_ema(
    series,
    period
):

    return (
        pd.Series(series)
        .ewm(
            span=period,
            adjust=False,
            min_periods=period
        )
        .mean()
        .to_numpy()
    )


# ============================================================
# STRATEGY
# ============================================================

class InstitutionalBroomBreakout(
    Strategy
):

    def init(self):

        close = self.data.Close

        # ----------------------------------------------------
        # DAILY EMAs
        # ----------------------------------------------------

        self.ema20 = self.I(
            calculate_ema,
            close,
            20,
            name="EMA20"
        )

        self.ema50 = self.I(
            calculate_ema,
            close,
            50,
            name="EMA50"
        )

        self.ema100 = self.I(
            calculate_ema,
            close,
            100,
            name="EMA100"
        )

        self.ema200 = self.I(
            calculate_ema,
            close,
            200,
            name="EMA200"
        )

        # ----------------------------------------------------
        # HIGHER TIMEFRAME EMAs
        # ----------------------------------------------------

        self.weekly_200 = (
            self.data.Weekly_200EMA
        )

        self.monthly_200 = (
            self.data.Monthly_200EMA
        )

    # ========================================================
    # NEXT
    # ========================================================

    def next(self):

        # Need sufficient daily history.
        if len(self.data.Close) < 200:
            return

        current_price = float(
            self.data.Close[-1]
        )

        weekly_ema = float(
            self.weekly_200[-1]
        )

        monthly_ema = float(
            self.monthly_200[-1]
        )

        if not np.isfinite(
            current_price
        ):
            return

        if not np.isfinite(
            weekly_ema
        ):
            return

        if not np.isfinite(
            monthly_ema
        ):
            return

        # ====================================================
        # TRAILING STOP
        # ====================================================

        if self.position:

            new_sl = (
                float(self.ema200[-1])
                * STOP_EMA_MULTIPLIER
            )

            if not np.isfinite(new_sl):
                return

            for trade in self.trades:

                if trade.sl is None:

                    trade.sl = new_sl

                elif new_sl > trade.sl:

                    trade.sl = new_sl

            return

        # ====================================================
        # RULE 1
        # MACRO TREND
        # ====================================================

        macro_bullish = (
            current_price > monthly_ema
            and
            current_price > weekly_ema
        )

        if not macro_bullish:
            return

        # ====================================================
        # DAILY EMA VALUES
        # ====================================================

        current_emas = np.array(
            [
                self.ema20[-1],
                self.ema50[-1],
                self.ema100[-1],
                self.ema200[-1]
            ],
            dtype=float
        )

        if not np.all(
            np.isfinite(current_emas)
        ):
            return

        # ====================================================
        # RULE 2
        # EMA BROOM COMPRESSION
        # ====================================================

        ema_spread = (
            (
                np.max(current_emas)
                -
                np.min(current_emas)
            )
            /
            current_price
        )

        if ema_spread >= EMA_COMPRESSION:
            return

        # ====================================================
        # RULE 3
        # MAJOR HIGH 63-147 DAYS AGO
        # ====================================================

        lookback_window = 200

        recent_highs = np.asarray(
            self.data.High[
                -lookback_window:
            ],
            dtype=float
        )

        if len(recent_highs) < 200:
            return

        highest_idx = int(
            np.argmax(recent_highs)
        )

        days_since_high = (
            lookback_window - 1
            -
            highest_idx
        )

        if not (
            CONSOLIDATION_MIN_DAYS
            <=
            days_since_high
            <=
            CONSOLIDATION_MAX_DAYS
        ):
            return

        peak_price = float(
            recent_highs[highest_idx]
        )

        # ====================================================
        # RULE 4
        # PREVIOUS 20-DAY CONSOLIDATION
        #
        # IMPORTANT:
        # CURRENT BREAKOUT CANDLE IS EXCLUDED.
        #
        # [-21:-1] =
        # 20 COMPLETED DAYS BEFORE TODAY.
        # ====================================================

        previous_20_highs = np.asarray(
            self.data.High[-21:-1],
            dtype=float
        )

        previous_20_lows = np.asarray(
            self.data.Low[-21:-1],
            dtype=float
        )

        previous_20_closes = np.asarray(
            self.data.Close[-21:-1],
            dtype=float
        )

        previous_20_volumes = np.asarray(
            self.data.Volume[-21:-1],
            dtype=float
        )

        if len(previous_20_highs) < 20:
            return

        if len(previous_20_lows) < 20:
            return

        if len(previous_20_closes) < 20:
            return

        if len(previous_20_volumes) < 20:
            return

        # ----------------------------------------------------
        # CONSOLIDATION BOX
        # ----------------------------------------------------

        box_high = float(
            np.max(previous_20_highs)
        )

        box_low = float(
            np.min(previous_20_lows)
        )

        if box_low <= 0:
            return

        box_range = (
            box_high - box_low
        ) / box_low

        if box_range >= MAX_BOX_RANGE:
            return

        # ====================================================
        # RULE 5
        # PREVIOUS TREND CAP
        # ====================================================

        if highest_idx <= 40:
            return

        prior_trend_low = float(
            np.min(
                np.asarray(
                    self.data.Low[
                        highest_idx - 40:
                        highest_idx
                    ],
                    dtype=float
                )
            )
        )

        if prior_trend_low <= 0:
            return

        prior_runup = (
            peak_price
            -
            prior_trend_low
        ) / prior_trend_low

        if prior_runup > MAX_PRIOR_RUNUP:
            return

        # ====================================================
        # RULE 6
        # ACTUAL BREAKOUT TRIGGER
        # ====================================================

        breakout_level = (
            box_high
            *
            (1.0 + BREAKOUT_BUFFER)
        )

        # Today's CLOSE must break the
        # previous 20-day consolidation high.
        price_breakout = (
            current_price
            >
            breakout_level
        )

        if not price_breakout:
            return

        # ----------------------------------------------------
        # PRICE MUST ALSO BE ABOVE ALL EMAs
        # ----------------------------------------------------

        above_all_emas = (
            current_price
            >
            np.max(current_emas)
        )

        if not above_all_emas:
            return

        # ====================================================
        # VOLUME PROFILE
        #
        # IMPORTANT:
        # POC uses ONLY PREVIOUS 20 DAYS.
        # Current breakout day is excluded.
        # ====================================================

        poc_price = calculate_poc(
            previous_20_closes,
            previous_20_volumes,
            bins=10
        )

        if not np.isfinite(
            poc_price
        ):
            return

        above_poc = (
            current_price
            >
            poc_price
        )

        if not above_poc:
            return

        # ====================================================
        # VOLUME SPIKE
        #
        # IMPORTANT:
        # Average volume uses PREVIOUS 20 DAYS.
        # Today's breakout volume is NOT included
        # in the denominator.
        # ====================================================

        average_volume = float(
            np.mean(
                previous_20_volumes
            )
        )

        if average_volume <= 0:
            return

        current_volume = float(
            self.data.Volume[-1]
        )

        volume_breakout = (
            current_volume
            >
            average_volume
            *
            VOLUME_MULTIPLIER
        )

        if not volume_breakout:
            return

        # ====================================================
        # ALL ENTRY CONDITIONS PASSED
        # ====================================================

        # ----------------------------------------------------
        # STOP LOSS
        # ----------------------------------------------------

        sl_price = (
            float(self.ema200[-1])
            *
            STOP_EMA_MULTIPLIER
        )

        if not np.isfinite(
            sl_price
        ):
            return

        if sl_price >= current_price:
            return

        # ====================================================
        # MEASURED MOVE
        # ====================================================

        # Base is the consolidation after
        # the major high and before today.
        post_peak_lows = np.asarray(
            self.data.Low[
                highest_idx:-1
            ],
            dtype=float
        )

        if len(post_peak_lows) == 0:
            return

        base_low = float(
            np.min(post_peak_lows)
        )

        base_depth = (
            peak_price
            -
            base_low
        )

        if base_depth <= 0:
            return

        tp_price = (
            current_price
            +
            TARGET_MULTIPLIER
            *
            base_depth
        )

        if tp_price <= current_price:
            return

        # ====================================================
        # BUY
        # ====================================================

        self.buy(
            sl=sl_price,
            tp=tp_price
        )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_stock_data(
    raw_df,
    ticker
):

    if raw_df is None:
        return None

    if raw_df.empty:
        return None

    df = raw_df.copy()

    # --------------------------------------------------------
    # MultiIndex handling
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        # If this is a single ticker extracted
        # from yf.download(), flatten columns.
        if len(
            df.columns.levels
        ) > 1:

            df.columns = [
                column[0]
                if isinstance(
                    column,
                    tuple
                )
                else column
                for column in df.columns
            ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in required:

        if column not in df.columns:
            return None

    df = df[
        required
    ].copy()

    # --------------------------------------------------------
    # Clean index
    # --------------------------------------------------------

    if df.index.tz is not None:

        df.index = (
            df.index
            .tz_localize(None)
        )

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    )

    if len(df) < 1000:
        return None

    # ========================================================
    # WEEKLY
    # ========================================================

    weekly = (
        df
        .resample("W")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }
        )
        .dropna(
            subset=["Close"]
        )
    )

    weekly[
        "Weekly_200EMA"
    ] = (
        weekly["Close"]
        .ewm(
            span=200,
            adjust=False,
            min_periods=200
        )
        .mean()
    )

    # ========================================================
    # MONTHLY
    # ========================================================

    monthly = (
        df
        .resample("ME")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }
        )
        .dropna(
            subset=["Close"]
        )
    )

    monthly[
        "Monthly_200EMA"
    ] = (
        monthly["Close"]
        .ewm(
            span=200,
            adjust=False,
            min_periods=200
        )
        .mean()
    )

    # ========================================================
    # ALIGN HIGHER TIMEFRAME EMA
    # ========================================================

    df[
        "Weekly_200EMA"
    ] = (
        weekly["Weekly_200EMA"]
        .reindex(df.index)
        .ffill()
    )

    df[
        "Monthly_200EMA"
    ] = (
        monthly["Monthly_200EMA"]
        .reindex(df.index)
        .ffill()
    )

    # ========================================================
    # BACKTEST PERIOD
    # ========================================================

    start = pd.Timestamp(
        BACKTEST_START
    )

    end = pd.Timestamp(
        BACKTEST_END
    )

    df = df.loc[
        (df.index >= start)
        &
        (df.index < end)
    ].copy()

    # ========================================================
    # REQUIRED INDICATOR DATA
    # ========================================================

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Weekly_200EMA",
            "Monthly_200EMA"
        ]
    )

    if df.empty:
        return None

    return df


# ============================================================
# DOWNLOAD ALL NIFTY 500 DATA
# ============================================================

def download_nifty500_data(
    tickers
):

    print("\n")
    print("=" * 70)
    print(
        f"DOWNLOADING DATA FOR "
        f"{len(tickers)} NIFTY 500 STOCKS"
    )
    print("=" * 70)

    print(
        "This may take several minutes."
    )

    # --------------------------------------------------------
    # Yahoo supports multiple ticker downloads.
    # Chunking reduces the chance of rate limits.
    # --------------------------------------------------------

    chunk_size = 50

    all_data = {}

    for start_index in range(
        0,
        len(tickers),
        chunk_size
    ):

        chunk = tickers[
            start_index:
            start_index + chunk_size
        ]

        chunk_number = (
            start_index // chunk_size
        ) + 1

        total_chunks = int(
            np.ceil(
                len(tickers)
                /
                chunk_size
            )
        )

        print(
            f"\nDownloading chunk "
            f"{chunk_number}/{total_chunks} "
            f"({len(chunk)} stocks)..."
        )

        try:

            raw = yf.download(
                chunk,
                period=DOWNLOAD_PERIOD,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                threads=DOWNLOAD_THREADS,
                progress=False,
                timeout=30,
                multi_level_index=True
            )

        except Exception as error:

            print(
                "Chunk download failed:"
            )

            print(error)

            continue

        if raw is None or raw.empty:

            print(
                "No data returned "
                "for this chunk."
            )

            continue

        # ----------------------------------------------------
        # Extract each ticker.
        # ----------------------------------------------------

        for ticker in chunk:

            try:

                if (
                    isinstance(
                        raw.columns,
                        pd.MultiIndex
                    )
                    and
                    ticker in raw.columns
                ):

                    stock_df = raw[
                        ticker
                    ].copy()

                else:

                    # Single-level fallback.
                    stock_df = raw.copy()

                if stock_df.empty:
                    continue

                prepared = prepare_stock_data(
                    stock_df,
                    ticker
                )

                if prepared is not None:

                    all_data[ticker] = (
                        prepared
                    )

            except Exception as error:

                print(
                    f"Skipping {ticker}: "
                    f"{error}"
                )

        print(
            f"Usable stocks so far: "
            f"{len(all_data)}"
        )

        # Small pause between chunks.
        time.sleep(2)

    return all_data


# ============================================================
# RUN ONE STOCK
# ============================================================

def run_single_stock(
    ticker,
    data
):

    try:

        bt = Backtest(
            data,
            InstitutionalBroomBreakout,
            cash=INITIAL_CASH_PER_STOCK,
            commission=COMMISSION,
            exclusive_orders=True
        )

        stats = bt.run()

        trades = int(
            stats["# Trades"]
        )

        if trades == 0:

            return {
                "Ticker": ticker,
                "Trades": 0,
                "Return [%]": 0.0,
                "Win Rate [%]": 0.0,
                "Max Drawdown [%]": 0.0,
                "Equity Final [$]":
                    INITIAL_CASH_PER_STOCK
            }

        return {
            "Ticker": ticker,
            "Trades": trades,
            "Return [%]":
                float(stats["Return [%]"]),
            "Win Rate [%]":
                float(stats["Win Rate [%]"]),
            "Max Drawdown [%]":
                float(stats["Max. Drawdown [%]"]),
            "Equity Final [$]":
                float(stats["Equity Final [$]"])
        }

    except Exception as error:

        print(
            f"ERROR in {ticker}: "
            f"{error}"
        )

        return {
            "Ticker": ticker,
            "Trades": -1,
            "Return [%]": np.nan,
            "Win Rate [%]": np.nan,
            "Max Drawdown [%]": np.nan,
            "Equity Final [$]": np.nan
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print(
        "INSTITUTIONAL BROOM BREAKOUT"
    )
    print(
        "NIFTY 500 UNIVERSE BACKTEST"
    )
    print("=" * 70)

    print(
        f"Backtest: "
        f"{BACKTEST_START} "
        f"to "
        f"{BACKTEST_END}"
    )

    # ========================================================
    # GET UNIVERSE
    # ========================================================

    tickers = get_nifty500_symbols()

    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    all_data = download_nifty500_data(
        tickers
    )

    print("\n")
    print("=" * 70)
    print(
        f"USABLE STOCKS: "
        f"{len(all_data)}"
    )
    print("=" * 70)

    # ========================================================
    # RUN BACKTESTS
    # ========================================================

    results = []

    total = len(all_data)

    for number, (
        ticker,
        data
    ) in enumerate(
        all_data.items(),
        start=1
    ):

        print(
            f"[{number}/{total}] "
            f"Testing {ticker}..."
        )

        result = run_single_stock(
            ticker,
            data
        )

        results.append(
            result
        )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise ValueError(
            "No backtest results generated."
        )

    # ========================================================
    # SORT
    # ========================================================

    results_df = results_df.sort_values(
        by="Return [%]",
        ascending=False
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        "nifty500_backtest_results.csv",
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    successful = results_df[
        results_df["Trades"] >= 0
    ]

    stocks_with_trades = results_df[
        results_df["Trades"] > 0
    ]

    total_trades = int(
        stocks_with_trades[
            "Trades"
        ].sum()
    )

    print("\n")
    print("=" * 70)
    print(
        "NIFTY 500 BACKTEST COMPLETE"
    )
    print("=" * 70)

    print(
        f"Universe size: "
        f"{len(tickers)}"
    )

    print(
        f"Stocks successfully tested: "
        f"{len(successful)}"
    )

    print(
        f"Stocks generating trades: "
        f"{len(stocks_with_trades)}"
    )

    print(
        f"Total trades: "
        f"{total_trades}"
    )

    # ========================================================
    # TOP STOCKS
    # ========================================================

    if not stocks_with_trades.empty:

        print("\n")
        print(
            "TOP 20 STOCKS BY INDIVIDUAL "
            "BACKTEST RETURN"
        )

        print(
            stocks_with_trades[
                [
                    "Ticker",
                    "Trades",
                    "Return [%]",
                    "Win Rate [%]",
                    "Max Drawdown [%]"
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # TRADE COUNT DISTRIBUTION
    # ========================================================

    print("\n")
    print(
        "TRADE COUNT DISTRIBUTION"
    )

    print(
        results_df[
            "Trades"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    print("\n")
    print(
        "Results saved to:"
    )

    print(
        "nifty500_backtest_results.csv"
    )

    print("\n")
    print(
        "BACKTEST COMPLETED."
    )
