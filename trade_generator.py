if not signals_df.empty:
            # Check if file exists
            file_path = 'nifty500_broom_breakout_results.csv'
            if os.path.exists(file_path):
                # Load existing and append
                existing_df = pd.read_csv(file_path)
                combined_df = pd.concat([existing_df, signals_df], ignore_index=True)
                # Drop exact duplicates if any
                combined_df = combined_df.drop_duplicates(subset=['ticker', 'signal_date'], keep='last')
                combined_df.to_csv(file_path, index=False)
            else:
                # If no file exists, just save it
                signals_df.to_csv(file_path, index=False)
            logger.info(f"\n✓ Appended {len(signals_df)} new signals to CSV")
        else:
            # Don't overwrite if we found nothing today
            logger.info("\n✓ No new signals found today. CSV not overwritten.")
        
        return signals_df
