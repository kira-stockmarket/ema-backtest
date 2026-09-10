# 3. Enter new setups
        while signal_idx < len(signals) and signals[signal_idx]['date'] == current_date:
            sig = signals[signal_idx]
            signal_idx += 1
            
            if sig['ticker'] not in portfolio and len(portfolio) < MAX_POSITIONS:
                # Restored 10% Equity Allocation for Maximum CAGR
                trade_alloc = daily_equity * 0.10
                
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
