"""
Paper Trading Terminal with Trailing Stop-Loss
Tracks open positions, updates trailing stops, and logs closed trades.
"""

import pandas as pd
import json
import os
from datetime import datetime
import yfinance as yf
import logging
import sys

# ==================== LOGGING & SETUP ====================
os.makedirs('state', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = 'state/paper_portfolio.json'
TRADE_HISTORY_FILE = 'paper_trade_history.csv'
SIGNALS_FILE = 'nifty500_broom_breakout_results.csv'
PARAMS_FILE = 'state/current_params.json'

def load_ai_params():
    try:
        with open(PARAMS_FILE, 'r') as f:
            return json.load(f).get('params', {})
    except:
        return {'stop_loss_buffer': 0.015}

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=4)

def execute_new_signals(portfolio):
    """Finds today's signals and adds them to the open portfolio"""
    if not os.path.exists(SIGNALS_FILE):
        return portfolio
        
    df = pd.read_csv(SIGNALS_FILE)
    if df.empty:
        return portfolio
        
    today_str = datetime.now().strftime('%Y-%m-%d')
    new_signals = df[df['signal_date'] == today_str]
    
    for _, row in new_signals.iterrows():
        ticker = row['ticker']
        # Don't buy if we are already holding it
        if ticker not in portfolio:
            portfolio[ticker] = {
                'entry_date': today_str,
                'entry_price': row['entry_price'],
                'current_price': row['entry_price'],
                'highest_price': row['entry_price'],
                'initial_stop': row['stop_loss'],
                'trailing_stop': row['stop_loss'],
                'take_profit': row['take_profit'],
                'confidence': row.get('confidence_pct', 0)
            }
            logger.info(f"🟢 BOUGHT: {ticker} at ₹{row['entry_price']:.2f}")
            
    return portfolio

def fetch_live_prices(tickers):
    """Fetches current market price using yfinance"""
    if not tickers:
        return {}
    
    prices = {}
    try:
        # Download data for all tickers at once for speed
        data = yf.download(tickers, period="1d", progress=False)
        if len(tickers) == 1:
            prices[tickers[0]] = data['Close'].iloc[-1]
        else:
            for ticker in tickers:
                prices[ticker] = data['Close'][ticker].iloc[-1]
    except Exception as e:
        logger.error(f"Error fetching live prices: {e}")
        
    return prices

def log_closed_trade(trade_record):
    """Saves closed trades to a CSV history file"""
    df = pd.DataFrame([trade_record])
    if os.path.exists(TRADE_HISTORY_FILE):
        df.to_csv(TRADE_HISTORY_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(TRADE_HISTORY_FILE, index=False)

def manage_positions():
    logger.info("=" * 60)
    logger.info("OPENING PAPER TRADING TERMINAL")
    logger.info("=" * 60)
    
    portfolio = load_portfolio()
    params = load_ai_params()
    sl_buffer = params.get('stop_loss_buffer', 0.015)
    
    # 1. Onboard new trades from today's scanner
    portfolio = execute_new_signals(portfolio)
    
    if not portfolio:
        logger.info("Portfolio is empty. Waiting for setups.")
        return
        
    # 2. Get live prices for open positions
    open_tickers = list(portfolio.keys())
    live_prices = fetch_live_prices(open_tickers)
    
    closed_tickers = []
    
    # 3. Evaluate each position
    for ticker, trade in portfolio.items():
        if ticker not in live_prices or pd.isna(live_prices[ticker]):
            continue
            
        current_price = live_prices[ticker]
        trade['current_price'] = round(current_price, 2)
        
        # --- TRAILING STOP LOGIC ---
        # If price makes a new high, recalculate the stop loss
        if current_price > trade['highest_price']:
            trade['highest_price'] = current_price
            
            # The stop loss trails a set percentage behind the highest seen price
            new_trailing_stop = current_price * (1 - sl_buffer)
            
            # Ensure the stop loss only moves UP, never down
            if new_trailing_stop > trade['trailing_stop']:
                trade['trailing_stop'] = round(new_trailing_stop, 2)
                logger.info(f"🛡️ {ticker} Trailing Stop moved UP to ₹{trade['trailing_stop']}")

        # --- EXIT LOGIC ---
        exit_reason = None
        
        if current_price <= trade['trailing_stop']:
            exit_reason = "STOP_LOSS"
        elif current_price >= trade['take_profit']:
            exit_reason = "TAKE_PROFIT"
            
        if exit_reason:
            pnl_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
            
            log_closed_trade({
                'ticker': ticker,
                'entry_date': trade['entry_date'],
                'exit_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': trade['entry_price'],
                'exit_price': current_price,
                'pnl_pct': round(pnl_pct, 2),
                'reason': exit_reason,
                'confidence': trade['confidence']
            })
            
            emoji = "🤑" if pnl_pct > 0 else "🩸"
            logger.info(f"{emoji} SOLD {ticker}: {exit_reason} | PnL: {pnl_pct:.2f}%")
            closed_tickers.append(ticker)

    # 4. Remove closed trades from the active portfolio
    for ticker in closed_tickers:
        del portfolio[ticker]
        
    # 5. Save the updated state
    save_portfolio(portfolio)
    logger.info(f"Terminal closed. {len(portfolio)} active positions remaining.")

if __name__ == "__main__":
    manage_positions()
