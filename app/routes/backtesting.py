from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta
import traceback
import numpy as np

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.alphazero.trader import AlphaZeroTrader
from src.alphazero.environment import TradingEnvironment
from app.routes.main import load_data_and_model
from app.routes.settings import load_user_settings

bp = Blueprint('backtesting', __name__, url_prefix='/backtesting')

@bp.route('/')
def index():
    """Backtesting page for model evaluation"""
    # Load user settings
    user_settings = load_user_settings()
    
    # Initialize session variables if not present
    if 'lot_size' not in session:
        session['lot_size'] = user_settings['lot_size']
    if 'initial_capital' not in session:
        session['initial_capital'] = user_settings['initial_capital']
    if 'trade_time' not in session:
        session['trade_time'] = user_settings['trade_time']
    
    # Get backtesting parameters from query string (if any)
    use_mcts = request.args.get('use_mcts', 'false').lower() == 'true'
    
    # Check if we have backtest results from a previous run
    backtest_results = session.get('backtest_results')
    
    # Load data and model
    trader, nifty_df, vix_df = load_data_and_model()
    
    # Get date range for default values
    start_date = ""
    end_date = ""
    if nifty_df is not None:
        try:
            end_date = nifty_df.index.max().strftime('%Y-%m-%d')
            start_date = (nifty_df.index.max() - timedelta(days=90)).strftime('%Y-%m-%d')
        except:
            pass
    
    return render_template('backtesting.html',
                          has_data=(nifty_df is not None and vix_df is not None),
                          backtest_results=backtest_results,
                          use_mcts=use_mcts,
                          start_date=start_date,
                          end_date=end_date)

@bp.route('/run', methods=['POST'])
def run_backtest():
    """Run a backtest"""
    # Get user settings
    user_settings = load_user_settings()
    
    # Get backtest parameters
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    use_mcts = request.form.get('use_mcts') == 'on'
    lot_size = int(request.form.get('lot_size', session.get('lot_size', user_settings['lot_size'])))
    initial_capital = int(request.form.get('initial_capital', session.get('initial_capital', user_settings['initial_capital'])))
    
    # Save to session
    session['lot_size'] = lot_size
    session['initial_capital'] = initial_capital
    
    # Load data and model
    trader, nifty_df, vix_df = load_data_and_model()
    
    if nifty_df is None or vix_df is None:
        flash("Error: No data available for backtesting", "error")
        return redirect(url_for('backtesting.index'))
    
    try:
        # Convert dates to datetime objects
        try:
            start_date_obj = pd.Timestamp(start_date).tz_localize('Asia/Kolkata')
            end_date_obj = pd.Timestamp(end_date).tz_localize('Asia/Kolkata')
        except TypeError:
            # If already timezone-aware, just use as is
            start_date_obj = pd.Timestamp(start_date)
            end_date_obj = pd.Timestamp(end_date)
        
        # Initialize the environment
        trader.env = TradingEnvironment(
            nifty_data=nifty_df,
            vix_data=vix_df,
            features_extractor=trader.features_extractor,
            trade_time=session.get('trade_time', user_settings['trade_time']),
            lot_size=lot_size,
            initial_capital=initial_capital
        )
        
        # Run backtest
        results = trader.backtest(
            start_date=start_date_obj,
            end_date=end_date_obj,
            use_mcts=use_mcts
        )
        
        # Process results
        if results:
            # Store in session
            session['backtest_results'] = results
            flash("Backtest completed successfully", "success")
        else:
            flash("Backtest failed to produce results", "error")
    
    except Exception as e:
        flash(f"Error during backtesting: {e}", "error")
        traceback.print_exc()
    
    return redirect(url_for('backtesting.index'))

@bp.route('/clear', methods=['POST'])
def clear_results():
    """Clear backtest results"""
    if 'backtest_results' in session:
        session.pop('backtest_results')
        flash("Backtest results cleared", "success")
    
    return redirect(url_for('backtesting.index')) 