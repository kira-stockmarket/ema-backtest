"""
Dashboard Generator - Creates interactive HTML dashboard
"""

import pandas as pd
import json
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

def generate_dashboard():
    """Generate HTML dashboard"""
    signals = load_signals()
    params = load_learned_params()
    
    version = params.get('version', 0)
    updated_at = params.get('updated_at', 'Never')
    learned = params.get('params', {})
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Broom Breakout - Live Trading Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(
