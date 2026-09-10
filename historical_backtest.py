"""
Institutional Historical Backtest - Nifty 100
Features: 1% Portfolio Risk Sizing, Macro Regime Filter, & Dynamic 20-EMA Trailing Exits.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURED PARAMETERS ---
PARAMS = {
    'broom_compression_threshold': 0.15,
    'base_duration_min': 63,
    'base_duration_max': 200,
    'box_consolidation_height': 0.25,
    'volume_threshold_multiplier': 1.2,
    'stop_loss_buffer': 0.01
}

INITIAL_CAPITAL = 100000.0
RISK_PER_TRADE_PCT = 0.01  # Risk exactly 1% of total portfolio equity per trade
MAX_POSITIONS = 10

NIFTY_100 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 
    'SBIN.NS', 'INFY.NS', 'LICI.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 
    'BAJFINANCE.NS', 'HCLTECH.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'ADANIENT.NS', 
    'KOTAKBANK.NS', 'TITAN.NS', 'ONGC.NS', 'TATAMOTORS.NS', 'NTPC.NS', 
    'AXISBANK.NS', 'DMART.NS', 'ADANIPORTS.NS', 'ULTRACEMCO.NS', 'ASIANPAINT.NS', 
    'COALINDIA.NS', 'BAJAJFINSV.NS', 'BAJAJ-AUTO.NS', 'POWERGRID.NS', 
    'HAL.NS', 'M&M.NS', 'TATASTEEL.NS', 'SIEMENS.NS', 'SBICARD.NS',
    'VBL.NS', 'IOC.NS', 'ZOMATO.NS', 'GRASIM.NS', 'JIOFIN.NS', 'WIPRO.NS',
    'TRENT.NS', 'BEL.NS', 'DLF.NS', 'VEDL.NS', 'INDUSINDBK.NS', 'PFC.NS',
    'INDIGO.NS', 'RECLTD.NS', 'GODREJCP.NS', 'TATACHEM.NS', 'JSWSTEEL.NS',
    'CHOLAFIN.NS', 'HDFCLIFE.NS', 'DRREDDY.NS', 'BOSCHLTD.NS', 'BPCL.NS',
    'PNB.NS', 'GAIL.NS', 'BRITANNIA.NS', 'TECHM.NS', 'EICHERMOT.NS',
    'SHRIRAMFIN.NS', 'CIPLA.NS', 'CANBK.NS', 'TVSMOTOR.NS', 'AMBUJACEM.NS',
    'APOLLOHOSP.NS', 'BANKBARODA.NS', 'HINDALCO.NS', 'TORNTPHARM.NS', 
    'CGPOWER.NS', 'ZYDUSLIFE.NS', 'IDBI.NS', 'MAXHEALTH.NS', 'JINDALSTEL.NS', 
    'DIVISLAB.NS', 'ICICIGI.NS', 'LODHA.NS', 'TIINDIA.NS', 'TATACOMM.NS',
    'UNIONBANK.NS', 'MCDOWELL-N.NS', 'HEROMOTOCO.NS', 'MANKIND.NS',
    'INDIANB.NS', 'SBILIFE.NS', 'YESBANK.NS', 'SRF.NS', 'MOTHERSON.NS',
    'IRFC.NS', 'PIDILITIND.NS', 'MUTHOOTFIN.NS', 'COLPAL.NS', 'PIIND.NS',
    'MARICO.NS', 'UBL.NS', 'BANDHANBNK.NS', 'AWL.NS', 'NHPC.NS'
]

def backtest():
    print("Fetching historical data for Nifty 100 + Nifty 50 Index (5 Years)...")
    # Added ^NSEI (Nifty 50) for the Macro Regime Filter
    tickers_to_fetch = NIFTY_100 + ['^NSEI']
    data = yf.download(tickers_to_fetch, period="5y", progress=False)
    
    print("Calculating Macro Regime Filter...")
    nifty_index = data.xs('^NSEI', level=1, axis=1).dropna(how='all').copy()
    nifty_index['EMA_100'] = nifty_index['Close'].ewm(span=100, adjust=False).mean()
    nifty_index['Regime_Bull'] = nifty_index['Close'] > nifty_index['EMA_100']
    
    print("Calculating indicators & checking breakout conditions...")
    emas = [20, 50, 100, 200]
    signals = []
    
    for ticker in NIFTY_100:
        try:
            df = data.xs(ticker, level=1, axis=1).dropna(how='all').copy()
            if len(df) < 250: continue
            
            for p in emas:
                df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
            df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
            df = df.dropna()
            
            for i in range(PARAMS['base_duration_max'], len(df)):
                current_price = df['Close'].iloc[i]
                ema_200 = df['EMA_200'].iloc[i]
                current_date = df.index[i]
                
                if current_price <= ema_200: continue
                
                # Check Macro Regime Filter (Nifty 50 must be Bullish)
                try:
                    if not nifty_index.loc[current_date, 'Regime_Bull']: continue
                except KeyError:
                    continue # Skip if index data is missing for this day
                
                current_emas = [df[f'EMA_{p}'].iloc[i] for p in emas]
                compression = (max(current_emas) - min(current_emas)) / current_price
                if compression >= PARAMS['broom_compression_threshold']: continue
                
                vol_ma = df['Volume_MA_50'].iloc[i]
                if vol_ma <= 0: continue
                vol_ratio = df['Volume'].iloc[i] / vol_ma
                if vol_ratio < PARAMS['volume_threshold_multiplier']: continue
                
                lookback = df['High'].iloc[max(0, i - PARAMS['base_duration_max']):i]
                peak_price = lookback.max()
                peak_idx = df.index.get_loc(lookback.idxmax())
                
                base_duration = i - peak_idx
                if not (PARAMS['base_duration_min'] <= base_duration <= PARAMS['base_duration_max']): continue
                
                base_low = df['Low'].iloc[peak_idx:i+1].min()
                consolidation_height = (peak_price - base_low) / peak_price
                if consolidation_height > PARAMS['box_consolidation_height']: continue
                
                # Initial Stop Loss
                stop_loss = ema_200 * (1 - PARAMS['stop_loss_buffer'])
                
                signals.append({
                    'ticker': ticker,
                    'date': current_date,
                    'entry_price': current_price,
                    'initial_stop_loss': stop_loss
                })
        except Exception:
            continue
            
    signals.sort(key=lambda x: x['date'])
    
    # --- SIMULATION ENGINE ---
    print(f"Simulating Portfolio... Starting Cash: ₹{INITIAL_CAPITAL}")
    cash = INITIAL_CAPITAL
    portfolio = {}
    trade_history = []
    equity_curve = []
    
    all_dates = data.index.sort_values().unique()
    signal_idx = 0
    
    for current_date in all_dates:
        current_date_ts = pd.Timestamp(current_date)
        closed_this_day = []
        
        # 1. Update Exits (Dynamic 20-EMA Trailing Stop)
        for ticker, pos in portfolio.items():
            try:
                day_data = data.xs(ticker, level=1, axis=1).loc[current_date]
                if pd.isna(day_data['Open']): continue
                
                # Calculate current EMA 20 for trailing
                current_ema_20 = day_data['Close'] # Approximated via close for backtest efficiency
                try:
                    current_ema_20 = data.xs(ticker, level=1, axis=1).loc[:current_date]['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
                except:
                    pass
                
                # Trail the stop loss up, never down
                pos['current_stop'] = max(pos['current_stop'], current_ema_20)
                
                low = day_data['Low']
                exit_price = None
                
                # Exit if low breaches the trailing stop
                if low <= pos['current_stop']:
                    # Slippage simulation: exit at stop price or open, whichever is worse
                    exit_price = min(pos['current_stop'], day_data['Open']) 
                    
                if exit_price:
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    cash += (exit_price * pos['shares'])
                    trade_history.append({
                        'Ticker': ticker,
                        'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'Exit Date': current_date_ts.strftime('%Y-%m-%d'),
                        'Shares': pos['shares'],
                        'Entry Price': round(pos['entry_price'], 2),
                        'Exit Price': round(exit_price, 2),
                        'Profit/Loss (₹)': round(pnl, 2),
                        'Return (%)': round(((exit_price / pos['entry_price']) - 1) * 100, 2)
                    })
                    closed_this_day.append(ticker)
            except KeyError:
                continue
                
        for ticker in closed_this_day:
            del portfolio[ticker]
            
        # 2. Track daily equity (Cash + Positions)
        daily_equity = cash
        for t, p in portfolio.items():
            try:
                daily_equity += p['shares'] * data.xs(t, level=1, axis=1).loc[current_date]['Close']
            except Exception:
                daily_equity += p['shares'] * p['entry_price']
        
        # 3. Enter new setups
        while signal_idx < len(signals) and signals[signal_idx]['date'] == current_date:
            sig = signals[signal_idx]
            signal_idx += 1
            
            if sig['ticker'] not in portfolio and len(portfolio) < MAX_POSITIONS:
                # Calculate 1% Portfolio Risk Sizing
                risk_amount = daily_equity * RISK_PER_TRADE_PCT
                risk_per_share = sig['entry_price'] - sig['initial_stop_loss']
                
                if risk_per_share > 0:
                    shares = int(risk_amount / risk_per_share)
                    trade_cost = shares * sig['entry_price']
                    
                    # Ensure we don't spend more cash than we have
                    if trade_cost > cash:
                        shares = int(cash / sig['entry_price'])
                        trade_cost = shares * sig['entry_price']
                    
                    if shares > 0:
                        cash -= trade_cost
                        portfolio[sig['ticker']] = {
                            'entry_date': sig['date'],
                            'entry_price': sig['entry_price'],
                            'current_stop': sig['initial_stop_loss'],
                            'shares': shares
                        }
                        
        equity_curve.append({'Date': current_date_ts, 'Equity': daily_equity})
        
    # --- CALCULATE METRICS (Including Calmar) ---
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    
    # CAGR
    final_equity = equity_df['Equity'].iloc[-1]
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years)) - 1
    
    # Maximum Drawdown
    equity_df['Peak'] = equity_df['Equity'].expanding(min_periods=1).max()
    equity_df['Drawdown'] = (equity_df['Equity'] / equity_df['Peak']) - 1
    max_drawdown = equity_df['Drawdown'].min()
    
    # Calmar Ratio
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    print(f"\n--- INSTITUTIONAL BACKTEST RESULTS ---")
    print(f"Total Trades Taken: {len(trade_history)}")
    print(f"Final Portfolio Value: ₹{final_equity:,.2f}")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    
    pd.DataFrame(trade_history).to_csv('backtest_trades.csv', index=False)
    
    monthly_equity = equity_df.resample('ME').last()
    monthly_equity['Monthly Profit (₹)'] = monthly_equity['Equity'].diff().fillna(monthly_equity['Equity'] - INITIAL_CAPITAL)
    monthly_equity['Monthly Return (%)'] = monthly_equity['Equity'].pct_change().fillna((monthly_equity['Equity'] / INITIAL_CAPITAL) - 1) * 100
    
    monthly_equity = monthly_equity.reset_index()
    monthly_equity['Date'] = monthly_equity['Date'].dt.strftime('%Y-%m')
    monthly_equity = monthly_equity.round(2)
    monthly_equity.to_csv('monthly_profits.csv', index=False)
    
    print("Saved: 'backtest_trades.csv' and 'monthly_profits.csv'")

if __name__ == "__main__":
    backtest()
