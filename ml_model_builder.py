"""
Supervised Machine Learning Builder
Generates a dataset of historical setups and trains a Random Forest to predict trade success.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report

# We relax the parameters slightly to generate a dataset with both good AND bad trades
# The ML model needs to see failures to learn what to avoid.
DATA_COLLECTION_PARAMS = {
    'broom_compression_threshold': 0.25,  # Relaxed
    'volume_threshold_multiplier': 1.0,   # Relaxed
    'base_duration_max': 250,
    'stop_loss_buffer': 0.015,
    'measured_move_multiplier': 1.5
}

NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS',
    'INFY.NS', 'ITC.NS', 'SBIN.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS',
    'LARSEN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'HCLTECH.NS', 'ASIANPAINT.NS',
    'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'TATAMOTORS.NS'
]

def build_dataset():
    print("Step 1: Gathering Historical Data for ML Training...")
    data = yf.download(NIFTY_50, period="5y", progress=False)
    emas = [20, 50, 100, 200]
    
    dataset = []
    
    for ticker in NIFTY_50:
        try:
            df = data.xs(ticker, level=1, axis=1).dropna(how='all').copy()
            if len(df) < 300: continue
            
            for p in emas:
                df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
            df['Volume_MA_50'] = df['Volume'].rolling(50).mean()
            
            # Additional ML Features
            df['RSI_14'] = calculate_rsi(df['Close'], 14)
            df = df.dropna()
            
            for i in range(250, len(df) - 40): # Leave 40 days to check the future
                current_price = df['Close'].iloc[i]
                ema_200 = df['EMA_200'].iloc[i]
                
                if current_price <= ema_200: continue
                
                # --- EXTRACT FEATURES (X) ---
                current_emas = [df[f'EMA_{p}'].iloc[i] for p in emas]
                compression = (max(current_emas) - min(current_emas)) / current_price
                if compression >= DATA_COLLECTION_PARAMS['broom_compression_threshold']: continue
                
                vol_ma = df['Volume_MA_50'].iloc[i]
                vol_ratio = df['Volume'].iloc[i] / vol_ma if vol_ma > 0 else 0
                if vol_ratio < DATA_COLLECTION_PARAMS['volume_threshold_multiplier']: continue
                
                lookback = df['High'].iloc[max(0, i - DATA_COLLECTION_PARAMS['base_duration_max']):i]
                peak_price = lookback.max()
                peak_idx = lookback.idxmax()
                base_duration = i - df.index.get_loc(peak_idx)
                
                base_low = df['Low'].iloc[df.index.get_loc(peak_idx):i+1].min()
                base_depth_pct = (peak_price - base_low) / peak_price
                
                # Trend strength (How far above 200 EMA)
                trend_strength = (current_price - ema_200) / ema_200
                
                # --- CHECK FUTURE OUTCOME (y) ---
                take_profit = current_price + (DATA_COLLECTION_PARAMS['measured_move_multiplier'] * (peak_price - base_low))
                stop_loss = ema_200 * (1 - DATA_COLLECTION_PARAMS['stop_loss_buffer'])
                
                future_data = df.iloc[i+1:i+41]
                outcome = 0 # Default to loss
                
                for _, row in future_data.iterrows():
                    if row['High'] >= take_profit:
                        outcome = 1 # Win
                        break
                    elif row['Low'] <= stop_loss:
                        outcome = 0 # Loss
                        break
                
                # Append row to dataset
                dataset.append({
                    'compression': compression,
                    'volume_ratio': vol_ratio,
                    'base_duration': base_duration,
                    'base_depth_pct': base_depth_pct,
                    'trend_strength': trend_strength,
                    'rsi': df['RSI_14'].iloc[i],
                    'target': outcome
                })
                
        except Exception as e:
            continue
            
    return pd.DataFrame(dataset)

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(df):
    print(f"Step 2: Training Machine Learning Model on {len(df)} setups...")
    
    # Balance the dataset (optional but recommended for Random Forests)
    wins = df[df['target'] == 1]
    losses = df[df['target'] == 0]
    min_len = min(len(wins), len(losses))
    balanced_df = pd.concat([wins.sample(min_len), losses.sample(min_len)])
    
    # Define Features (X) and Target (y)
    features = ['compression', 'volume_ratio', 'base_duration', 'base_depth_pct', 'trend_strength', 'rsi']
    X = balanced_df[features]
    y = balanced_df['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"Precision (When it says BUY, how often is it right?): {precision_score(y_test, predictions) * 100:.2f}%")
    
    print("\n--- FEATURE IMPORTANCE ---")
    for feature, imp in zip(features, model.feature_importances_):
        print(f"{feature}: {imp*100:.1f}%")
        
    # Save the brain
    os.makedirs('state', exist_ok=True)
    joblib.dump(model, 'state/ml_brain.pkl')
    print("\n✅ AI Brain saved to 'state/ml_brain.pkl'")

if __name__ == "__main__":
    df = build_dataset()
    if not df.empty:
        train_model(df)
    else:
        print("Not enough data to train the model.")
