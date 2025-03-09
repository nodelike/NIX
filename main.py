import os
import sys
import subprocess
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add the project path to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.data_loader import fetch_new_data, save_to_csv, get_latest_data
from src.data_processing.features import extract_features
from src.alphazero.trader import AlphaZeroTrader

def main():
    parser = argparse.ArgumentParser(description='AlphaZero Trading System for NIFTY and VIX')
    parser.add_argument('--mode', type=str, default='app', choices=['app', 'train', 'backtest', 'fetch_data'],
                       help='Mode to run the application in')
    parser.add_argument('--trade_time', type=str, default='09:05',
                       help='Time of day to trade (format: HH:MM)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of self-play episodes for training')
    parser.add_argument('--batches', type=int, default=20,
                       help='Number of training batches')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to fetch data for')
    
    args = parser.parse_args()
    
    if args.mode == 'fetch_data':
        # Fetch new data only
        print(f"Fetching {args.days} days of data...")
        end_date = datetime.now(ZoneInfo("Asia/Kolkata"))
        start_date = end_date - timedelta(days=args.days)
        
        nifty_df, vix_df = fetch_new_data(start_date, end_date)
        if nifty_df is not None and vix_df is not None:
            save_to_csv(nifty_df, vix_df)
            print("Data fetched and saved successfully")
        else:
            print("Error fetching data")
    
    elif args.mode == 'train':
        # Train the model
        print("Loading data and initializing model...")
        nifty_df, vix_df = get_latest_data(load_from_disk=True, fetch_days=args.days)
        
        if nifty_df is None or vix_df is None:
            print("Error loading data")
            return
        
        # Initialize trader
        trader = AlphaZeroTrader(
            nifty_data=nifty_df,
            vix_data=vix_df,
            features_extractor=extract_features,
            input_shape=(1, 10),
            trade_time=args.trade_time
        )
        
        # Try to load existing model
        trader.load_model()
        
        # Run self-play and training
        print(f"Running {args.episodes} self-play episodes...")
        trader.self_play(episodes=args.episodes)
        
        print(f"Training model with {args.batches} batches...")
        trader.train(num_batches=args.batches)
        
        # Save model
        trader.save_model()
        print("Model trained and saved successfully")
    
    elif args.mode == 'backtest':
        # Run backtest
        print("Loading data and initializing model...")
        nifty_df, vix_df = get_latest_data(load_from_disk=True, fetch_days=args.days)
        
        if nifty_df is None or vix_df is None:
            print("Error loading data")
            return
        
        # Initialize trader
        trader = AlphaZeroTrader(
            nifty_data=nifty_df,
            vix_data=vix_df,
            features_extractor=extract_features,
            input_shape=(1, 10),
            trade_time=args.trade_time
        )
        
        # Try to load existing model
        if not trader.load_model():
            print("Error loading model. Please train a model first.")
            return
        
        # Run backtest
        print("Running backtest...")
        results_df = trader.backtest(use_mcts=False)
        
        if results_df is not None and len(results_df) > 0:
            # Save backtest results
            results_path = os.path.join('data', f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            results_df.to_csv(results_path)
            print(f"Backtest results saved to {results_path}")
    
    elif args.mode == 'app':
        # Run the Streamlit app
        print("Starting Streamlit app...")
        streamlit_path = os.path.join('app', 'app.py')
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', streamlit_path])

if __name__ == "__main__":
    main()
