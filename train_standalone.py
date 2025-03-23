#!/usr/bin/env python
"""
Standalone script to train the AlphaZero model.
This can be run separately to train a model before using the dashboard.
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback
from datetime import datetime

# Add project path to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.data_loader import get_latest_data, load_from_csv
from src.data_processing.features import get_features_extractor
from src.alphazero.trader import AlphaZeroTrader
from src.alphazero.environment import TradingEnvironment

def load_data():
    """Load data from consolidated CSV files or using get_latest_data"""
    # Try to load directly from CSV files
    data_dir = 'data'
    nifty_csv_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
    vix_csv_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
    
    if os.path.exists(nifty_csv_path) and os.path.exists(vix_csv_path):
        try:
            print("Loading data from consolidated CSV files...")
            nifty_df, vix_df = load_from_csv(nifty_csv_path, vix_csv_path)
            print(f"Loaded {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records from CSV")
            return nifty_df, vix_df
        except Exception as e:
            print(f"Error loading from CSV: {e}")
    
    # Fall back to get_latest_data
    print("Loading data using get_latest_data...")
    return get_latest_data(load_from_disk=True, fetch_days=30)

def train_model_robust(nifty_df, vix_df, episodes=5, batches=10):
    """Robust model training with extensive error handling"""
    try:
        print("Starting model training...")
        
        # Create a features extractor
        features_extractor = get_features_extractor()
        
        # Create environment
        env = TradingEnvironment(
            nifty_data=nifty_df,
            vix_data=vix_df,
            features_extractor=features_extractor,
            trade_time='9:15',
            lot_size=50,
            initial_capital=100000,
            test_mode=False
        )
        
        # Reset and process data
        try:
            print("Resetting environment...")
            env.reset()
            
            if not env.features_list:
                print("Processing data...")
                env._prepare_data()
                
            if not env.features_list:
                print("Error: No features extracted from data")
                return None
                
            print(f"Processed {len(env.features_list)} trading days")
        except Exception as e:
            print(f"Error processing data: {e}")
            traceback.print_exc()
            return None
            
        # Get state shape
        try:
            print("Getting state shape...")
            state = env.get_state(0)
            if state is None:
                print("Error: Could not get state from environment")
                return None
                
            if not isinstance(state, np.ndarray):
                state = np.array(state)
                
            # Print detailed shape info
            print(f"Raw state type: {type(state)}")
            print(f"State shape: {state.shape}")
            print(f"State dtype: {state.dtype}")
            print(f"First few values: {state.flatten()[:5]}")
                
            input_shape = state.shape
            print(f"Using input shape: {input_shape}")
        except Exception as e:
            print(f"Error getting state shape: {e}")
            traceback.print_exc()
            return None
        
        # Create trader
        try:
            print("Creating trader...")
            trader = AlphaZeroTrader(
                input_shape=input_shape,
                n_actions=3,
                features_extractor=features_extractor
            )
            
            # Assign environment
            trader.env = env
        except Exception as e:
            print(f"Error creating trader: {e}")
            traceback.print_exc()
            return None
        
        # Run self-play
        try:
            print(f"\nRunning {episodes} self-play episodes...")
            examples = trader.self_play(episodes=episodes)
            print(f"Generated {examples} examples")
            
            if examples == 0:
                print("No examples generated. Skipping training.")
                return None
        except Exception as e:
            print(f"Error in self-play: {e}")
            traceback.print_exc()
            return None
        
        # Train model
        try:
            print(f"\nTraining model for {batches} batches...")
            success = trader.train(num_batches=batches)
            
            if not success:
                print("Training failed.")
                return None
        except Exception as e:
            print(f"Error training model: {e}")
            traceback.print_exc()
            return None
        
        # Save model
        try:
            print("\nSaving model...")
            trader.save_model('alphazero_model.h5')
            print("Model trained and saved successfully")
            return trader
        except Exception as e:
            print(f"Error saving model: {e}")
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Unexpected error in training: {e}")
        traceback.print_exc()
        return None

def main():
    try:
        # Load data
        nifty_df, vix_df = load_data()
        
        if nifty_df is None or vix_df is None:
            print("Error: Failed to load data. Please check if data files exist.")
            return False
        
        # Make sure data has timezone
        if nifty_df.index.tz is None:
            nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
        if vix_df.index.tz is None:
            vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
            
        # Align data indices
        common_dates = nifty_df.index.intersection(vix_df.index)
        if len(common_dates) == 0:
            print("Error: No common dates in NIFTY and VIX data")
            return False
            
        nifty_df = nifty_df.loc[common_dates]
        vix_df = vix_df.loc[common_dates]
        
        print(f"Using {len(nifty_df)} aligned data points")
        
        # Configure training parameters - keep them small for first run
        episodes = 3
        batches = 5
        
        print(f"Training configuration:")
        print(f"- Self-play episodes: {episodes}")
        print(f"- Training batches: {batches}")
        
        # Train model using robust approach
        start_time = datetime.now()
        trader = train_model_robust(nifty_df, vix_df, episodes, batches)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds() / 60
        
        if trader is not None:
            print(f"Training completed in {duration:.1f} minutes")
            print("You can now use the dashboard")
            return True
        else:
            print(f"Training failed after {duration:.1f} minutes")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 