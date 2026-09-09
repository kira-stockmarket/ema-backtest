"""
Dashboard Generator - Creates interactive HTML dashboard
"""

import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

def load_signals():
    """Load trade signals"""
    try:
        df = pd.read_csv('nifty500_broom_breakout_results.csv')
        return df
    except:
        return pd.DataFrame()

def load_learned_params():
    """Load learned parameters"""
    try:
        with open('state/current_params.json', 'r') as f:
            data = json.load(f)
        return data
    except:
        return {'params': {}, 'version': 0}

def generate_table(df):
    """Generate HTML table"""
    if df.empty:
        return "<p style='text-align:center; color:#666; padding:20px;'>No active signals currently</p>"
    
    html = """<table>
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Action</th>
                <th>Entry Price</th>
                <th>Stop Loss</th>
                <th>Target</th>
                <th>Risk</th>
                <th>Reward</th>
                <th>R:R</th>
                <th>Volume</th>
                <th>RSI</th>
            </tr>
        </thead>
        <tbody>"""
    
    for _, row in df.iterrows():
        action = row.get('action', 'BUY')
        badge_class = 'buy-badge' if action == 'BUY' else 'sell-badge'
        
        html += "<tr>"
        html += "<td><strong>{}</strong></td>".format(row.get('ticker', ''))
        html += '<td><span class="{}">{}</span></td>'.format(badge_class, action)
        html += "<td>₹{:,.2f}</td>".format(row.get('entry_price', 0))
        html += "<td>₹{:,.2f}</td>".format(row.get('stop_loss', 0))
        html += "<td>₹{:,.2f}</td>".format(row.get('take_profit', 0))
        html += "<td>₹{:,.2f}</td>".format(row.get('risk', 0))
        html += "<td>₹{:,.2f}</td>".format(row.get('reward', 0))
        html += "<td><strong>{:.2f}</strong></td>".format(row.get('risk_reward', 0))
        html += "<td>{:.2f}x</td>".format(row.get('volume_ratio', 0))
        html += "<td>{:.1f}</td>".format(row.get('rsi', 0))
        html += "</tr>"
    
    html += "</tbody></table>"
    return html

def generate_dashboard():
    """Generate HTML dashboard"""
    signals = load_signals()
    params = load_learned_params()
    
    version = params.get('version', 0)
    updated_at = params.get('updated_at', 'Never')
    learned = params.get('params', {})
    
    # Use .format() instead of f-string to avoid CSS brace issues
    css = """
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { 
            background: white; border-radius: 15px; padding: 30px;
            margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header h1 { font-size: 2.5em; color: #333; margin-bottom: 10px; }
        .header .subtitle { color: #666; font-size: 1.1em; }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px; margin-bottom: 20px;
        }
        .stat-card {
            background: white; border-radius: 15px; padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1); text-align: center;
        }
        .stat-card .value { font-size: 2.5em; font-weight: bold; color: #667eea; }
        .stat-card .label { color: #666; margin-top: 5px; }
        .signals-section {
            background: white; border-radius: 15px; padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .signals-section h2 { color: #333; margin-bottom: 20px; font-size: 1.8em; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #667eea; color: white; padding: 15px; text-align: left; font-weight: 600; }
        td { padding: 12px 15px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f8f9fa; }
        .buy-badge { 
            background: #28a745; color: white; padding: 5px 15px;
            border-radius: 20px; font-weight: bold;
        }
        .sell-badge { 
            background: #dc3545; color: white; padding: 5px 15px;
            border-radius: 20px; font-weight: bold;
        }
        .params-section {
            background: white; border-radius: 15px; padding: 25px;
            margin-bottom: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .params-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-top: 15px;
        }
        .param-item {
            background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;
        }
        .param-item .param-value { font-size: 1.5em; font-weight: bold; color: #667eea; }
        .param-item .param-label { color: #666; margin-top: 5px; font-size: 0.9em; }
        .footer { text-align: center; color: white; margin-top: 20px; opacity: 0.8; }
    </style>
    """
    
    signals_count = len(signals)
    buy_count = len(signals[signals['action'] == 'BUY']) if not signals.empty else 0
    sell_count = len(signals[signals['action'] == 'SELL']) if not signals.empty else 0
    
    if not signals.empty:
        avg_rr = signals['risk_reward'].mean()
    else:
        avg_rr = 0
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Broom Breakout - Live Trading Dashboard</title>
    {css}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Broom Breakout Strategy</h1>
            <p class="subtitle">Live Trading Dashboard | Auto-Updated</p>
            <p class="subtitle">Last Updated: {update_time}</p>
            <p class="subtitle">Version: {version}</p>
        </div>
        
        <div class="params-section">
            <h2>🧠 Learned Parameters</h2>
            <div class="params-grid">
                <div class="param-item">
                    <div class="param-value">{compression:.3f}</div>
                    <div class="param-label">Compression Threshold</div>
                </div>
                <div class="param-item">
                    <div class="param-value">{volume:.2f}x</div>
                    <div class="param-label">Volume Multiplier</div>
                </div>
                <div class="param-item">
                    <div class="param-value">{stop:.3f}</div>
                    <div class="param-label">Stop Loss Buffer</div>
                </div>
                <div class="param-item">
                    <div class="param-value">{target:.2f}x</div>
                    <div class="param-label">Measured Move</div>
                </div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{signals_count}</div>
                <div class="label">Active Signals</div>
            </div>
            <div class="stat-card">
                <div class="value">{buy_count}</div>
                <div class="label">Buy Signals</div>
            </div>
            <div class="stat-card">
                <div class="value">{sell_count}</div>
                <div class="label">Sell Signals</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_rr:.2f}</div>
                <div class="label">Avg Risk/Reward</div>
            </div>
        </div>
        
        <div class="signals-section">
            <h2>📊 Live Trade Signals</h2>
            {table}
        </div>
        
        <div class="footer">
            <p>Auto-generated by Broom Breakout Strategy | GitHub Actions</p>
            <p>⚠️ For educational purposes only. Not financial advice.</p>
        </div>
    </div>
</body>
</html>""".format(
        css=css,
        update_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        version=version,
        compression=learned.get('broom_compression_threshold', 0.08),
        volume=learned.get('volume_threshold_multiplier', 1.5),
        stop=learned.get('stop_loss_buffer', 0.015),
        target=learned.get('measured_move_multiplier', 2.0),
        signals_count=signals_count,
        buy_count=buy_count,
        sell_count=sell_count,
        avg_rr=avg_rr,
        table=generate_table(signals)
    )
    
    with open('index.html', 'w') as f:
        f.write(html)
    
    print("✓ Dashboard generated: index.html")
    return html

if __name__ == "__main__":
    generate_dashboard()
