"""
Pure Price Action ML Builder
Focuses on strict chronological learning and core structural features to prevent overfitting.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

warnings.filterwarnings('ignore')

DATA_COLLECTION_PARAMS = {
    'broom_compression_threshold': 0.20,
    'volume_threshold_multiplier': 1.0,
    'base_duration_max': 250,
    'stop_loss_buffer': 0.015,
    'measured_move_multiplier': 1.5
}

# Top 50 Nifty Stocks - Solid sample size without unnecessary noise
NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 'SBIN.NS', 'INFY.NS', 
    'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 
    'ADANIENT.NS', 'KOTAKBANK.NS', 'TITAN.NS', 'ONGC.NS', 'TATAMOTORS.NS', 'NTPC.NS', 'AXISBANK.NS', 
    'DMART.NS', 'ADANIPORTS.NS', 'ULTRACEMCO.NS', 'ASIANPAINT.NS', 'COALINDIA.NS', 'BAJAJFINSV.NS', 
    'BAJAJ-AUTO.NS', 'POWERGRID.NS', 'HAL.NS', 'M&M.NS', 'TATASTEEL.NS', 'SIEMENS.NS', 'SBICARD.NS',
    'VBL.NS', 'IOC.NS', 'ZOMATO.NS', 'GRASIM.NS', 'JIOFIN.NS', 'WIPRO.NS', 'TRENT.NS', 'BEL.NS', 
    'DLF.NS', 'VEDL.NS', 'INDUSINDBK.NS', 'PFC.NS', 'INDIGO.NS', 'RECLTD.NS', 'GODREJCP.NS', 'TATACHEM.NS'
]

def build_dataset():
    print("Step 1: Gathering Historical Data (Chronological Sequence)...")
    data = yf.download(NIFTY_50, period="8y", progress=False)
    emas = [20, 50, 100, 200]
    dataset = []
    
    for ticker in NIFTY_50:
        try:
            df = data.xs(ticker, level=1, axis=1).dropna(how='all').copy()
            if len(df) < 300: continue
            
            for p in emas:
                df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
            df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
            df = df.dropna()
            
            for i in range(250, len(df) - 40):
                current_price = df['Close'].iloc[i]
                ema_200 = df['EMA_200'].iloc[i]
                current_date = df.index[i]
                
                if current_price <= ema_200: continue
                
                current_emas = [df[f'EMA_{p}'].iloc[i] for p in emas]
                compression = (max(current_emas) - min(current_emas)) / current_price
                if compression >= DATA_COLLECTION_PARAMS['broom_compression_threshold']: continue
                
                vol_ma = df['Volume_MA_50'].iloc[i]
                if vol_ma <= 0: continue
                vol_ratio = df['Volume'].iloc[i] / vol_ma
                if vol_ratio < DATA_COLLECTION_PARAMS['volume_threshold_multiplier']: continue
                
                lookback = df['High'].iloc[max(0, i - DATA_COLLECTION_PARAMS['base_duration_max']):i]
                if lookback.empty: continue
                
                peak_price = lookback.max()
                peak_idx = lookback.idxmax()
                base_duration = i - df.index.get_loc(peak_idx)
                
                base_low = df['Low'].iloc[df.index.get_loc(peak_idx):i+1].min()
                base_depth_pct = (peak_price - base_low) / peak_price
                trend_strength = (current_price - ema_200) / ema_200
                
                take_profit = current_price + (DATA_COLLECTION_PARAMS['measured_move_multiplier'] * (peak_price - base_low))
                stop_loss = ema_200 * (1 - DATA_COLLECTION_PARAMS['stop_loss_buffer'])
                
                future_data = df.iloc[i+1:i+41]
                outcome = 0 
                for _, row in future_data.iterrows():
                    if row['High'] >= take_profit:
                        outcome = 1
                        break
                    elif row['Low'] <= stop_loss:
                        outcome = 0
                        break
                
                dataset.append({
                    'date': current_date,
                    'compression': compression,
                    'volume_ratio': vol_ratio,
                    'base_duration': base_duration,
                    'base_depth_pct': base_depth_pct,
                    'trend_strength': trend_strength,
                    'target': outcome
                })
                
        except Exception:
            continue
            
    df_out = pd.DataFrame(dataset)
    # Sort chronologically to prevent future-leaking during training
    df_out = df_out.sort_values('date').reset_index(drop=True)
    return df_out

def train_model(df):
    print(f"Step 2: Training Chronological ML Model on {len(df)} setups...")
    
    features = ['compression', 'volume_ratio', 'base_duration', 'base_depth_pct', 'trend_strength']
    
    # 80/20 Chronological Split (Train on the past, test on the future)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']
    
    # Class weight 'balanced' forces the AI to respect losses without throwing away data
    # max_depth=6 prevents the AI from memorizing noise (overfitting)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    print("\n--- TRUE LIVE-MARKET PREDICTIVE PERFORMANCE ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"Precision (Buy Signal Win Rate): {precision_score(y_test, predictions, zero_division=0) * 100:.2f}%")
    
    print("\n--- CORE STRUCTURAL FEATURE IMPORTANCE ---")
    importances = list(zip(features, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for feature, imp in importances:
        print(f"{feature}: {imp*100:.1f}%")
        
    os.makedirs('state', exist_ok=True)
    joblib.dump(model, 'state/ml_brain.pkl', compress=3)
    print("\n✅ Clean AI Brain saved to 'state/ml_brain.pkl'")

if __name__ == "__main__":
    df = build_dataset()
    if not df.empty:
        train_model(df)
    else:
        print("Error: No data found.")
