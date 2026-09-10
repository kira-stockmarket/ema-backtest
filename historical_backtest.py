"""
Historical Backtest - Nifty 100
Simulates a ₹100,000 portfolio using new parameters with Base Duration & Box Consolidation filters.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os

# --- CONFIGURED PARAMETERS ---
PARAMS = {
    'broom_compression_threshold': 0.15,
    'base_duration_min': 63,
    'base_duration_max': 200,
    'box_consolidation_height': 0.25,
    'volume_threshold_multiplier': 1.2,
    'stop_loss_buffer': 0.01,
    'measured_move_multiplier': 1.5
}

INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.10  # Allocate 10% of current equity per trade
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
    print("Fetching historical data for Nifty 100 (5 Years)...")
    data = yf.download(NIFTY_100, period="5y", progress=False)
    
    print("Calculating indicators & checking breakout conditions...")
    emas = [20, 50, 100, 200]
    signals = []
    
    for ticker in NIFTY_100:
        try:
            df = data.xs(ticker, level=1, axis=1).dropna(how='all').copy()
            if len(df) < 250:
                continue
            
            for p in emas:
                df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
            df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
            df = df.dropna()
            
            for i in range(PARAMS['base_duration_max'], len(df)):
                current_price = df['Close'].iloc[i]
                ema_200 = df['EMA_200'].iloc[i]
                
                # 1. Trend Filter
                if current_price <= ema_200:
                    continue
                
                # 2. EMA Compression Filter
                current_emas = [df[f'EMA_{p}'].iloc[i] for p in emas]
                compression = (max(current_emas) - min(current_emas)) / current_price
                if compression >= PARAMS['broom_compression_threshold']:
                    continue
                
                # 3. Volume Spike Filter
                vol_ma = df['Volume_MA_50'].iloc[i]
                if vol_ma <= 0:
                    continue
                vol_ratio = df['Volume'].iloc[i] / vol_ma
                if vol_ratio < PARAMS['volume_threshold_multiplier']:
                    continue
                
                # 4. Base & Box Consolidation Filter
                lookback = df['High'].iloc[max(0, i - PARAMS['base_duration_max']):i]
                peak_price = lookback.max()
                peak_date = lookback.idxmax()
                peak_idx = df.index.get_loc(peak_date)
                
                base_duration = i - peak_idx
                # Check base duration window (63 to 200 days)
                if not (PARAMS['base_duration_min'] <= base_duration <= PARAMS['base_duration_max']):
                    continue
                
                base_low = df['Low'].iloc[peak_idx:i+1].min()
                base_depth = peak_price - base_low
                consolidation_height = base_depth / peak_price
                
                # Check consolidation height cap (<= 25%)
                if consolidation_height > PARAMS['box_consolidation_height']:
                    continue
                
                take_profit = current_price + (PARAMS['measured_move_multiplier'] * base_depth)
                stop_loss = ema_200 * (1 - PARAMS['stop_loss_buffer'])
                
                signals.append({
                    'ticker': ticker,
                    'date': df.index[i],
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
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
        
        # Check exits
        for ticker, pos in portfolio.items():
            try:
                day_data = data.xs(ticker, level=1, axis=1).loc[current_date]
                if pd.isna(day_data['Open']):
                    continue
                
                high = day_data['High']
                low = day_data['Low']
                exit_price = None
                reason = None
                
                if high >= pos['take_profit']:
                    exit_price = pos['take_profit']
                    reason = "TAKE_PROFIT"
                elif low <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    reason = "STOP_LOSS"
                    
                if exit_price:
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    cash += (exit_price * pos['shares'])
                    trade_history.append({
                        'Ticker': ticker,
                        'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'Exit Date': current_date_ts.strftime('%Y-%m-%d'),
                        'Reason': reason,
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
            
        # Enter new setups
        while signal_idx < len(signals) and signals[signal_idx]['date'] == current_date:
            sig = signals[signal_idx]
            signal_idx += 1
            
            if sig['ticker'] not in portfolio and len(portfolio) < MAX_POSITIONS:
                current_equity = cash
                for t, p in portfolio.items():
                    try:
                        current_equity += p['shares'] * data.xs(t, level=1, axis=1).loc[current_date]['Close']
                    except Exception:
                        current_equity += p['shares'] * p['entry_price']
                        
                trade_alloc = current_equity * POSITION_SIZE_PCT
                if cash >= trade_alloc:
                    shares = int(trade_alloc / sig['entry_price'])
                    if shares > 0:
                        cash -= (shares * sig['entry_price'])
                        portfolio[sig['ticker']] = {
                            'entry_date': sig['date'],
                            'entry_price': sig['entry_price'],
                            'stop_loss': sig['stop_loss'],
                            'take_profit': sig['take_profit'],
                            'shares': shares
                        }
                        
        # Track daily equity
        daily_equity = cash
        for t, p in portfolio.items():
            try:
                daily_equity += p['shares'] * data.xs(t, level=1, axis=1).loc[current_date]['Close']
            except Exception:
                daily_equity += p['shares'] * p['entry_price']
                
        equity_curve.append({'Date': current_date_ts, 'Equity': daily_equity})
        
    # Final metrics
    final_equity = equity_curve[-1]['Equity']
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years)) - 1
    
    print(f"\n--- BACKTEST RESULTS ---")
    print(f"Total Trades Taken: {len(trade_history)}")
    print(f"Final Portfolio Value: ₹{final_equity:,.2f}")
    print(f"CAGR: {cagr * 100:.2f}%")
    
    pd.DataFrame(trade_history).to_csv('backtest_trades.csv', index=False)
    
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
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
