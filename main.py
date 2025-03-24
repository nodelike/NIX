import os
import sys
import subprocess
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import glob
import numpy as np
import json

# Add the project path to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.data_loader import fetch_new_data, save_to_csv, get_latest_data, consolidate_csv_files, update_data
from src.data_processing.features import extract_features, get_features_extractor
from src.alphazero.trader import AlphaZeroTrader
from src.alphazero.environment import TradingEnvironment

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='AlphaZero Trader')
    parser.add_argument('--mode', type=str, default='dashboard', choices=['train', 'backtest', 'dashboard'],
                        help='Mode to run: train, backtest, or dashboard (default: dashboard)')
    parser.add_argument('--update', action='store_true',
                        help='Update data before running')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to fetch data for')
    
    args = parser.parse_args()
    
    # Update data if requested
    if args.update:
        print("Updating data...")
        update_data(days=args.days)
    
    # Run the selected mode
    if args.mode == 'train':
        print("Starting training mode...")
        train_mode()
    elif args.mode == 'backtest':
        print("Starting backtest mode...")
        backtest_mode()
    else:  # Default to dashboard
        print("Starting dashboard...")
        launch_dashboard()

def backtest_mode():
    """Run the application in backtest mode"""
    try:
        print("Starting backtest mode...")
        
        # Load data
        try:
            print("Loading data...")
            nifty_df, vix_df = get_latest_data()
            
            if nifty_df is None or vix_df is None:
                print("Error: Failed to load data for backtesting")
                return
                
            # Ensure timezone is set correctly
            if nifty_df.index.tzinfo is None:
                nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
            if vix_df.index.tzinfo is None:
                vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
                
            print(f"Loaded {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records")
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Create a features extractor
        features_extractor = get_features_extractor()
        
        # Determine input shape from data
        try:
            print("Determining input shape...")
            # Create a temporary environment to get the state shape
            temp_env = TradingEnvironment(
                nifty_data=nifty_df,
                vix_data=vix_df,
                features_extractor=features_extractor,
                test_mode=True
            )
            temp_env.reset()
            
            state = temp_env.get_state(0)
            if state is None:
                print("Error: Could not get state from environment")
                input_shape = (1, 1, 10)  # Default fallback
                print(f"Using fallback input shape: {input_shape}")
            else:
                if not isinstance(state, np.ndarray):
                    state = np.array(state)
                input_shape = state.shape
                print(f"Using input shape: {input_shape}")
        except Exception as e:
            print(f"Error determining input shape: {e}")
            import traceback
            traceback.print_exc()
            # Use a default shape as fallback
            input_shape = (1, 1, 10)
            print(f"Using fallback input shape: {input_shape}")
        
        # Initialize trader
        try:
            print("Initializing trader...")
            trader = AlphaZeroTrader(
                input_shape=input_shape,
                n_actions=3,
                features_extractor=features_extractor,
                nifty_data=nifty_df,
                vix_data=vix_df
            )
            
            # Load model
            print("Loading model...")
            model_loaded = trader.load_model()
            
            if not model_loaded:
                print("Warning: Failed to load model. Using untrained model for backtesting.")
        except Exception as e:
            print(f"Error initializing trader: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Run backtest
        try:
            print("\nRunning backtest...")
            results = trader.backtest()
            
            # Print results
            print("\nBacktest Results:")
            print(f"Total trades: {results.get('total_trades', 0)}")
            print(f"Profitable trades: {results.get('profitable_trades', 0)}")
            print(f"Win rate: {results.get('win_rate', 0):.2%}")
            print(f"Total return: {results.get('total_return', 0):.2%}")
            print(f"Max drawdown: {results.get('max_drawdown', 0):.2%}")
            
            # Launch dashboard with backtest results
            print("\nLaunching dashboard with backtest results...")
            launch_dashboard(backtest_results=results)
        except Exception as e:
            print(f"Error during backtesting: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error in backtest mode: {e}")
        import traceback
        traceback.print_exc()

def train_mode():
    """Train the model with self-play."""
    try:
        # Load data
        print("Loading data...")
        nifty_df, vix_df = get_latest_data(load_from_disk=True, fetch_days=30)
        
        if nifty_df is None or vix_df is None:
            raise ValueError("Failed to load Nifty or VIX data")
        
        print(f"Loaded {len(nifty_df)} Nifty records and {len(vix_df)} VIX records.")
        
        # Make sure data has the correct timezone
        if nifty_df.index.tz is None:
            nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
        if vix_df.index.tz is None:
            vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
            
        # Ensure indices are aligned
        common_dates = nifty_df.index.intersection(vix_df.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between NIFTY and VIX data")
            
        nifty_df = nifty_df.loc[common_dates]
        vix_df = vix_df.loc[common_dates]
        
        # Train the model
        trader = train_model(
            nifty_df=nifty_df,
            vix_df=vix_df,
            episodes=10,
            batches=20,
            lot_size=50,
            initial_capital=100000
        )
        
        if trader is not None:
            print("\nTraining completed successfully.")
            return True
        else:
            print("\nTraining failed.")
            return False
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_dashboard(backtest_results=None):
    """
    Launch the dashboard application
    
    Args:
        backtest_results: Optional dictionary with backtest results
    """
    try:
        print("Launching dashboard...")
        
        # Use streamlit run command instead of importing directly
        import subprocess
        import os
        
        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "app.py")
        
        # If we have backtest results, save them to a temporary file
        results_file = None
        if backtest_results is not None:
            results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_results.json")
            with open(results_file, 'w') as f:
                json.dump(backtest_results, f)
            
            # Pass the file path as a parameter
            cmd = ["streamlit", "run", app_path, "--", "--results_file", results_file]
        else:
            cmd = ["streamlit", "run", app_path]
        
        # Run the streamlit app in a subprocess
        process = subprocess.Popen(cmd, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # Print the first few lines of output
        for i, line in enumerate(process.stdout):
            if i < 5:  # Just show first 5 lines
                print(line.strip())
            else:
                break
                
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        import traceback
        traceback.print_exc()

def train_model(nifty_df, vix_df, episodes=5, batches=10, lot_size=50, initial_capital=100000):
    """
    Train the AlphaZero model on the given data
    
    Args:
        nifty_df: DataFrame with NIFTY data
        vix_df: DataFrame with VIX data
        episodes: Number of self-play episodes to run
        batches: Number of training batches
        lot_size: Lot size for trading
        initial_capital: Initial capital for trading
        
    Returns:
        Trained AlphaZeroTrader instance
    """
    try:
        print("Starting model training...")
        
        # Create a features extractor
        features_extractor = get_features_extractor()
        
        # Check if we have data
        if nifty_df is None or vix_df is None:
            print("Error: No data provided for training")
            return None
            
        # Create environment
        try:
            print("Initializing trading environment...")
            env = TradingEnvironment(
                nifty_data=nifty_df,
                vix_data=vix_df,
                features_extractor=features_extractor,
                trade_time='9:15',
                lot_size=lot_size,
                initial_capital=initial_capital,
                test_mode=False
            )
            
            # Reset and process data
            env.reset()
            
            if not env.features_list:
                print("Processing data...")
                env._prepare_data()
                
            if not env.features_list:
                print("Error: No features extracted from data")
                return None
                
            print(f"Processed {len(env.features_list)} trading days")
        except Exception as e:
            print(f"Error initializing environment: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        # Get state shape
        try:
            print("Determining input shape...")
            state = env.get_state(0)
            if state is None:
                print("Error: Could not get state from environment")
                return None
                
            if not isinstance(state, np.ndarray):
                state = np.array(state)
                
            # Print detailed shape info
            print(f"State shape: {state.shape}")
            print(f"State dtype: {state.dtype}")
            
            input_shape = state.shape
            print(f"Using input shape: {input_shape}")
        except Exception as e:
            print(f"Error determining input shape: {e}")
            import traceback
            traceback.print_exc()
            # Use a default shape as fallback
            input_shape = (1, 1, 10)
            print(f"Using fallback input shape: {input_shape}")
        
        # Create trader
        try:
            print("Creating trader...")
            trader = AlphaZeroTrader(
                input_shape=input_shape,
                n_actions=3,
                features_extractor=features_extractor,
                nifty_data=nifty_df,
                vix_data=vix_df
            )
        except Exception as e:
            print(f"Error creating trader: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Run self-play
        try:
            print(f"\nRunning {episodes} self-play episodes...")
            examples = trader.self_play(episodes=episodes)
            
            if not examples:
                print("Warning: No examples generated during self-play")
                if len(trader.replay_buffer) < 10:
                    print("Not enough examples for training. Skipping training phase.")
                    return trader
        except Exception as e:
            print(f"Error during self-play: {e}")
            import traceback
            traceback.print_exc()
            if len(trader.replay_buffer) < 10:
                print("Not enough examples for training. Skipping training phase.")
                return trader
        
        # Train the model
        try:
            print(f"\nTraining model for {batches} batches...")
            training_stats = trader.train(batch_size=64, num_batches=batches)
            
            if 'error' in training_stats:
                print(f"Training error: {training_stats['error']}")
            else:
                print(f"Training completed with average loss: {training_stats['avg_total_loss']:.4f}")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        
        # Save the model
        try:
            print("\nSaving model...")
            trader.save_model()
            print("Model saved to models/alphazero_model.h5")
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
        
        return trader
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
