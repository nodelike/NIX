from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import sys
import os
import traceback
from datetime import datetime

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.features import create_market_summary
from src.alphazero.trader import AlphaZeroTrader
from src.alphazero.environment import TradingEnvironment
from app.routes.main import load_data_and_model
from app.routes.settings import load_user_settings

bp = Blueprint('prediction', __name__, url_prefix='/prediction')

@bp.route('/')
def index():
    """Prediction page for real-time trading"""
    # Load user settings
    user_settings = load_user_settings()
    
    # Initialize session variables if not present
    if 'lot_size' not in session:
        session['lot_size'] = user_settings['lot_size']
    if 'initial_capital' not in session:
        session['initial_capital'] = user_settings['initial_capital']
    if 'trade_time' not in session:
        session['trade_time'] = user_settings['trade_time']
    
    # Load data and model
    trader, nifty_df, vix_df = load_data_and_model()
    
    if trader is None or nifty_df is None or vix_df is None:
        flash("No data or model available. Please check data and model.", "error")
        return render_template('prediction.html', has_data=False)
    
    # Get the latest prediction
    prediction = None
    market_summary = None
    
    try:
        # Initialize environment if it doesn't exist
        if trader.env is None:
            trader.env = TradingEnvironment(
                nifty_data=nifty_df, 
                vix_data=vix_df,
                features_extractor=trader.features_extractor,
                window_size=10,
                trade_time=session.get('trade_time', user_settings['trade_time']),
                lot_size=session.get('lot_size', user_settings['lot_size']),
                initial_capital=session.get('initial_capital', user_settings['initial_capital']),
                test_mode=False
            )
        
        # Make sure the environment is properly initialized
        if not trader.env.features_list:
            trader.env._prepare_data()
            
        # Get latest state
        latest_idx = len(trader.env.features_list) - 1
        state = trader.env.get_state(latest_idx)
        
        if state is not None:
            # Get prediction with direct neural network
            direct_pred = trader.predict(state, use_mcts=False)
            
            # Create prediction object
            prediction = {
                'nn_prediction': {
                    'action': get_action_text(direct_pred['action']),
                    'confidence': direct_pred['confidence'],
                    'policy': direct_pred['policy'].tolist() if isinstance(direct_pred['policy'], np.ndarray) else direct_pred['policy']
                },
                'mcts_prediction': None,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'lot_size': session.get('lot_size', user_settings['lot_size'])
            }
            
            # Get window for market summary
            window_idx = latest_idx + trader.env.window_size
            market_window = trader.env.daily_data.iloc[window_idx-trader.env.window_size:window_idx]
            market_summary = create_market_summary(market_window)
            
    except Exception as e:
        flash(f"Error generating prediction: {e}", "error")
        traceback.print_exc()
    
    # Get the latest date
    latest_date = nifty_df.index[-1].date() if nifty_df is not None and len(nifty_df) > 0 else None
    today = datetime.now().date()
    
    return render_template('prediction.html',
                          has_data=True,
                          prediction=prediction,
                          market_summary=market_summary,
                          latest_date=latest_date,
                          today=today)

@bp.route('/mcts-prediction', methods=['POST'])
def mcts_prediction():
    """Run MCTS prediction (more computationally intensive)"""
    # Load user settings
    user_settings = load_user_settings()
    
    # Load data and model
    trader, nifty_df, vix_df = load_data_and_model()
    
    if trader is None or nifty_df is None or vix_df is None:
        return jsonify({'error': 'No data or model available'})
    
    try:
        # Make sure the environment is properly initialized
        if not trader.env.features_list:
            trader.env._prepare_data()
            
        # Get latest state
        latest_idx = len(trader.env.features_list) - 1
        state = trader.env.get_state(latest_idx)
        
        if state is not None:
            # Get prediction with MCTS
            mcts_pred = trader.predict(state, use_mcts=True)
            
            # Get action details
            action_text = get_action_text(mcts_pred['action'])
            
            # Run a test step to get trade info if action is BUY or SELL
            trade_info = None
            if action_text != "HOLD":
                # Create a test environment
                test_env = TradingEnvironment(
                    nifty_data=nifty_df, 
                    vix_data=vix_df,
                    features_extractor=trader.features_extractor,
                    trade_time=session.get('trade_time', user_settings['trade_time']),
                    lot_size=session.get('lot_size', user_settings['lot_size']),
                    initial_capital=session.get('initial_capital', user_settings['initial_capital']),
                    test_mode=False
                )
                test_env.reset()
                
                # Move to latest state
                test_env.current_idx = len(test_env.features_list) - 1
                
                # Take action
                _, _, _, info = test_env.step(mcts_pred['action'])
                
                # Extract trade info
                if 'entry_price' in info:
                    trade_info = {
                        'entry_price': float(info['entry_price']),
                        'lot_size': int(test_env.lot_size)
                    }
                    
                    if 'stop_loss' in info:
                        trade_info['stop_loss'] = float(info['stop_loss'])
                        trade_info['sl_pct'] = float(abs(info['stop_loss'] / info['entry_price'] - 1) * 100)
                    
                    if 'take_profit' in info:
                        trade_info['take_profit'] = float(info['take_profit'])
                        trade_info['tp_pct'] = float(abs(info['take_profit'] / info['entry_price'] - 1) * 100)
            
            # Return prediction
            return jsonify({
                'action': action_text,
                'confidence': float(mcts_pred['confidence']),
                'policy': mcts_pred['policy'].tolist() if isinstance(mcts_pred['policy'], np.ndarray) else mcts_pred['policy'],
                'trade_info': trade_info
            })
        
        return jsonify({'error': 'Could not get state from environment'})
        
    except Exception as e:
        return jsonify({'error': str(e)})

def get_action_text(action):
    """Convert action index to text"""
    action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
    return action_map.get(action, "UNKNOWN") 