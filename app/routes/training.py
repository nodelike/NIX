from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify
import sys
import os
import json
import numpy as np
import pandas as pd
from time import time
import traceback

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.data_loader import get_latest_data
from src.data_processing.features import get_features_extractor
from src.alphazero.trader import AlphaZeroTrader
from src.alphazero.environment import TradingEnvironment
from app.routes.main import load_data_and_model
from app.routes.settings import load_user_settings

bp = Blueprint('training', __name__, url_prefix='/training')

@bp.route('/')
def index():
    """Training page for model training"""
    # Load user settings
    user_settings = load_user_settings()
    
    # Set session values from user settings if not already set
    if 'lot_size' not in session:
        session['lot_size'] = user_settings['lot_size']
    if 'initial_capital' not in session:
        session['initial_capital'] = user_settings['initial_capital']
    if 'trade_time' not in session:
        session['trade_time'] = user_settings['trade_time']
    
    # Load data
    trader, nifty_df, vix_df = load_data_and_model()
    
    # Get training history from session if available
    training_history = session.get('training_history', {
        'rewards': [],
        'actions': [],
        'policy_loss': [],
        'value_loss': [],
        'total_loss': []
    })
    
    # Convert actions to dictionary format if needed
    if 'actions' in training_history and training_history['actions'] and not isinstance(training_history['actions'][0], dict):
        # Convert numeric actions to dict format
        action_names = ["buy", "sell", "hold"]
        converted_actions = []
        for action_list in training_history['actions']:
            action_dict = {}
            for i, count in enumerate(action_list):
                if i < len(action_names):
                    action_dict[action_names[i]] = count
            converted_actions.append(action_dict)
        training_history['actions'] = converted_actions
    
    return render_template('training.html',
                          has_data=(nifty_df is not None and vix_df is not None),
                          training_history=training_history)

@bp.route('/train', methods=['POST'])
def train():
    """Train the model"""
    try:
        # Get user settings
        user_settings = load_user_settings()
        
        # Get training parameters
        episodes = int(request.form.get('episodes', 10))
        simulation_steps = int(request.form.get('simulation_steps', 50))
        epochs = int(request.form.get('epochs', 20))
        batch_size = int(request.form.get('batch_size', 64))
        exploration_rate = float(request.form.get('exploration_rate', 0.25))
        learning_rate = float(request.form.get('learning_rate', 0.001))
        
        # Advanced parameters
        lot_size = int(request.form.get('lot_size', session.get('lot_size', user_settings['lot_size'])))
        initial_capital = int(request.form.get('initial_capital', session.get('initial_capital', user_settings['initial_capital'])))
        discount_factor = float(request.form.get('discount_factor', 0.99))
        dirichlet_alpha = float(request.form.get('dirichlet_alpha', 0.3))
        
        # Save to session
        session['lot_size'] = lot_size
        session['initial_capital'] = initial_capital
        
        # Load data and model
        trader, nifty_df, vix_df = load_data_and_model()
        
        if nifty_df is None or vix_df is None:
            flash("Error: No data available for training", "error")
            return redirect(url_for('training.index'))
        
        # Initialize training history
        training_history = {
            'rewards': [],
            'actions': [],
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
        # Create environment
        try:
            features_extractor = get_features_extractor()
            env = TradingEnvironment(
                nifty_data=nifty_df,
                vix_data=vix_df,
                features_extractor=features_extractor,
                trade_time=session.get('trade_time', user_settings['trade_time']),
                lot_size=lot_size,
                initial_capital=initial_capital,
                test_mode=False
            )
            
            # Reset and process data
            env.reset()
            
            if not env.features_list:
                env._prepare_data()
                
            if not env.features_list:
                flash("Error: No features extracted from data", "error")
                return redirect(url_for('training.index'))
                
            # Get state shape
            state = env.get_state(0)
            if state is None:
                flash("Error: Could not get state from environment", "error")
                return redirect(url_for('training.index'))
                
            if not isinstance(state, np.ndarray):
                state = np.array(state)
                
            input_shape = state.shape
            
            # Create trader
            trader = AlphaZeroTrader(
                input_shape=input_shape,
                n_actions=3,
                features_extractor=features_extractor,
                nifty_data=nifty_df,
                vix_data=vix_df
            )
            
            # Run self-play to generate examples
            examples = trader.self_play(
                episodes=episodes, 
                mcts_simulations=simulation_steps,
                exploration_rate=exploration_rate,
                dirichlet_alpha=dirichlet_alpha,
                discount_factor=discount_factor,
                callback=training_callback
            )
            
            if not examples and len(trader.replay_buffer) < 10:
                flash("Warning: Not enough examples generated for training", "warning")
                return redirect(url_for('training.index'))
            
            # Train the model
            training_stats = trader.train(
                batch_size=batch_size, 
                num_batches=epochs,
                learning_rate=learning_rate,
                callback=training_callback
            )
            
            # Save the model
            trader.save_model()
            
            # Save training history to session
            session['training_history'] = training_history
            
            flash("Model trained and saved successfully!", "success")
            
        except Exception as e:
            flash(f"Error during training: {e}", "error")
            return redirect(url_for('training.index'))
        
        return redirect(url_for('training.index'))
    
    except Exception as e:
        flash(f"Error setting up training: {e}", "error")
        return redirect(url_for('training.index'))

def training_callback(info):
    """Callback function for training updates"""
    # Get training history from session
    training_history = session.get('training_history', {
        'rewards': [],
        'actions': [],
        'policy_loss': [],
        'value_loss': [],
        'total_loss': []
    })
    
    try:
        # Handle self-play episode completion
        if 'episode_complete' in info and info['episode_complete']:
            # Add reward info
            if 'total_reward' in info:
                training_history['rewards'].append(info['total_reward'])
            
            # Add action distribution info
            if 'actions_count' in info:
                # Convert numeric keys to string action names if needed
                actions_count = dict(info['actions_count'])
                action_names = ["buy", "sell", "hold"]
                
                for i, name in enumerate(action_names):
                    if i in actions_count:
                        if name not in actions_count:
                            actions_count[name] = actions_count[i]
                        del actions_count[i]
                
                # Ensure all expected action names are in the dictionary
                for name in action_names:
                    if name not in actions_count:
                        actions_count[name] = 0
                
                training_history['actions'].append(actions_count)
        
        # Handle training batch completion
        if 'batch_complete' in info and info['batch_complete']:
            # Add loss info
            if 'policy_loss' in info:
                training_history['policy_loss'].append(info['policy_loss'])
                
            if 'value_loss' in info:
                training_history['value_loss'].append(info['value_loss'])
                
            if 'total_loss' in info:
                training_history['total_loss'].append(info['total_loss'])
        
        # Update progress information
        session['training_progress'] = info
        session['training_history'] = training_history
        
    except Exception as e:
        print(f"Error in training callback: {e}")
        traceback.print_exc()

@bp.route('/progress')
def progress():
    """Get current training progress for AJAX updates"""
    # Get training progress from session
    training_progress = session.get('training_progress', {
        'episode': 0,
        'step': 0,
        'reward': 0,
        'total_reward': 0,
        'actions_count': [0, 0, 0],
        'total_episodes': 0
    })
    
    # Get training history
    training_history = session.get('training_history', {
        'rewards': [],
        'actions': [],
        'policy_loss': [],
        'value_loss': [],
        'total_loss': []
    })
    
    # Calculate progress percentage
    progress_pct = 0
    if 'episode' in training_progress and 'total_episodes' in training_progress:
        if training_progress['total_episodes'] > 0:
            progress_pct = (training_progress['episode'] / training_progress['total_episodes']) * 100
    
    # Return as JSON
    return jsonify({
        'progress': training_progress,
        'history': training_history,
        'progress_pct': progress_pct
    }) 