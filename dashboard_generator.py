"""
Dashboard Generator
Reads the latest parameters and trade signals to generate a static HTML dashboard.
"""

import pandas as pd
import json
import os
from datetime import datetime

def load_parameters():
    """Load the latest AI-learned parameters"""
    params_file = 'state/current_params.json'
    default_params = {
        'broom_compression_threshold': 0.08,
        'volume_threshold_multiplier': 1.5,
        'stop_loss_buffer': 0.015,
        'measured_move_multiplier': 2.0
    }
    version = 0
    
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                data = json.load(f)
                return data.get('params', default_params), data.get('version', 0)
        except Exception as e:
            print(f"Error loading params: {e}")
            
    return default_params, version

def load_signals():
    """Load the latest trade signals"""
    csv_file = 'nifty500_broom_breakout_results.csv'
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            # Sort by date descending, then by confidence descending
            if 'confidence_pct' in df.columns:
                df = df.sort_values(by=['signal_date', 'confidence_pct'], ascending=[False, False])
            else:
                df = df.sort_values(by=['signal_date'], ascending=[False])
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
    return pd.DataFrame()

def generate_html(params, version, df):
    """Generate the HTML content"""
    
    # Format the parameters for display
    comp_thresh = f"{params.get('broom_compression_threshold', 0) * 100:.2f}%"
    vol_mult = f"{params.get('volume_threshold_multiplier', 0):.2f}x"
    sl_buffer = f"{params.get('stop_loss_buffer', 0) * 100:.2f}%"
    target_mult = f"{params.get('measured_move_multiplier', 0):.2f}x"
    
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Generate Table Rows
    table_rows = ""
    if not df.empty:
        for _, row in df.iterrows():
            confidence = row.get('confidence_pct', 'N/A')
            conf_badge = "bg-gray-700"
            if isinstance(confidence, (int, float)):
                if confidence >= 80:
                    conf_badge = "bg-green-600"
                elif confidence >= 50:
                    conf_badge = "bg-yellow-600"
                else:
                    conf_badge = "bg-red-600"
                confidence = f"{confidence}%"

            table_rows += f"""
            <tr class="border-b border-gray-700 hover:bg-gray-800 transition-colors">
                <td class="p-3 font-bold text-blue-400">{row.get('ticker', '')}</td>
                <td class="p-3">{row.get('signal_date', '')}</td>
                <td class="p-3 font-semibold text-green-400">{row.get('action', 'BUY')}</td>
                <td class="p-3">₹{row.get('entry_price', 0):.2f}</td>
                <td class="p-3 text-red-400">₹{row.get('stop_loss', 0):.2f}</td>
                <td class="p-3 text-green-400">₹{row.get('take_profit', 0):.2f}</td>
                <td class="p-3 font-mono">{row.get('risk_reward', 0):.2f}</td>
                <td class="p-3">{row.get('compression', 0):.2f}%</td>
                <td class="p-3">{row.get('volume_ratio', 0):.2f}x</td>
                <td class="p-3"><span class="px-2 py-1 rounded text-xs font-bold {conf_badge} text-white">{confidence}</span></td>
            </tr>
            """
    else:
        table_rows = """
        <tr>
            <td colspan="10" class="p-8 text-center text-gray-400">No active signals found. The AI is still hunting...</td>
        </tr>
        """

    # HTML Template
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Trade Scanner | Nifty 500</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
        <style>
            /* Custom Dark Theme for DataTables */
            .dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, 
            .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_processing, 
            .dataTables_wrapper .dataTables_paginate {{ color: #9ca3af; margin-bottom: 1rem; }}
            .dataTables_wrapper .dataTables_paginate .paginate_button {{ color: #9ca3af !important; }}
            table.dataTable tbody tr {{ background-color: transparent; }}
        </style>
    </head>
    <body class="bg-gray-900 text-gray-100 font-sans p-6 min-h-screen">
        
        <div class="max-w-7xl mx-auto">
            <!-- Header -->
            <div class="flex flex-col md:flex-row justify-between items-center border-b border-gray-700 pb-4 mb-8">
                <div>
                    <h1 class="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
                        Nifty 500 AI Scanner
                    </h1>
                    <p class="text-gray-400 mt-1">Broom Breakout Strategy</p>
                </div>
                <div class="text-right mt-4 md:mt-0">
                    <p class="text-sm text-gray-400">Last Updated</p>
                    <p class="font-mono text-emerald-400">{update_time}</p>
                </div>
            </div>

            <!-- Parameters Cards -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                    <p class="text-xs text-gray-400 uppercase tracking-wider mb-1">AI Version</p>
                    <p class="text-2xl font-bold text-white">v{version}</p>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                    <p class="text-xs text-gray-400 uppercase tracking-wider mb-1">Max Compression</p>
                    <p class="text-2xl font-bold text-blue-400">{comp_thresh}</p>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                    <p class="text-xs text-gray-400 uppercase tracking-wider mb-1">Min Volume</p>
                    <p class="text-2xl font-bold text-purple-400">{vol_mult}</p>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                    <p class="text-xs text-gray-400 uppercase tracking-wider mb-1">Target / Stop</p>
                    <p class="text-xl font-bold text-emerald-400">{target_mult} <span class="text-gray-500">/</span> <span class="text-red-400">{sl_buffer}</span></p>
                </div>
            </div>

            <!-- Data Table -->
            <div class="bg-gray-800 rounded-lg border border-gray-700 shadow-lg p-6 overflow-x-auto">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <svg class="w-5 h-5 mr-2 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                    Active Trade Signals
                </h2>
                
                <table id="signalsTable" class="w-full text-left text-sm whitespace-nowrap">
                    <thead class="text-xs text-gray-400 uppercase bg-gray-900/50">
                        <tr>
                            <th class="p-3 rounded-tl-lg">Ticker</th>
                            <th class="p-3">Date</th>
                            <th class="p-3">Action</th>
                            <th class="p-3">Entry</th>
                            <th class="p-3">Stop Loss</th>
                            <th class="p-3">Target</th>
                            <th class="p-3">R:R</th>
                            <th class="p-3">Compression</th>
                            <th class="p-3">Vol Ratio</th>
                            <th class="p-3 rounded-tr-lg">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script>
            $(document).ready(function() {{
                $('#signalsTable').DataTable({{
                    "order": [[ 1, "desc" ], [ 9, "desc" ]], // Sort by Date, then Confidence
                    "pageLength": 25,
                    "language": {{
                        "search": "Filter Tickers:",
                        "lengthMenu": "Show _MENU_ signals"
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("✓ Dashboard generated: index.html")

def main():
    print("Generating HTML Dashboard...")
    params, version = load_parameters()
    df = load_signals()
    generate_html(params, version, df)

if __name__ == "__main__":
    main()
