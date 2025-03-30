from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.data_loader import get_latest_data, load_from_csv
from src.data_processing.features import extract_features, create_market_summary, get_features_extractor
from src.alphazero.trader import AlphaZeroTrader
from src.alphazero.environment import TradingEnvironment

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Dashboard page with overview - initial load without data"""
    # Initialize session variables if not present
    if 'lot_size' not in session:
        session['lot_size'] = 50
    if 'initial_capital' not in session:
        session['initial_capital'] = 100000
    if 'trade_time' not in session:
        session['trade_time'] = '9:15'
    if 'data_loaded' not in session:
        session['data_loaded'] = False
    if 'model_loaded' not in session:
        session['model_loaded'] = False

    # Get chart type from query params
    chart_type = request.args.get('chart_type', 'line')
    
    # Setup defaults for template variables
    today = datetime.now()
    start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    # Default empty data structures
    nifty_data = {
        'dates': [],
        'open': [],
        'high': [],
        'low': [],
        'close': []
    }
    
    vix_data = {
        'dates': [],
        'open': [],
        'high': [],
        'low': [],
        'close': []
    }
    
    # Render template with no data but all required variables
    return render_template('dashboard.html',
                          chart_type=chart_type,
                          active_tab='nifty',
                          start_date=start_date,
                          end_date=end_date,
                          nifty_data=nifty_data,
                          vix_data=vix_data,
                          nifty_latest_price=0,
                          vix_latest_price=0,
                          nifty_change=0,
                          vix_change=0,
                          latest_date=today.date(),
                          today=today.date(),
                          prediction=None,
                          trade_history=None,
                          trade_metrics=None,
                          data_loaded=session.get('data_loaded', False),
                          model_loaded=session.get('model_loaded', False))

@bp.route('/load_data')
def load_data():
    """Load market data"""
    try:
        # Load data
        nifty_df, vix_df = load_market_data()
        
        if nifty_df is None or vix_df is None or len(nifty_df) == 0 or len(vix_df) == 0:
            flash("No data available. Please check your data files.", "warning")
            # Return to index with default values instead of redirecting
            today = datetime.now()
            return render_template('dashboard.html',
                               chart_type='line',
                               active_tab='nifty',
                               start_date=(today - timedelta(days=90)).strftime('%Y-%m-%d'),
                               end_date=today.strftime('%Y-%m-%d'),
                               nifty_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                               vix_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                               nifty_latest_price=0,
                               vix_latest_price=0,
                               nifty_change=0,
                               vix_change=0,
                               latest_date=today.date(),
                               today=today.date(),
                               prediction=None,
                               trade_history=None,
                               trade_metrics=None,
                               data_loaded=False,
                               model_loaded=False)
        
        # Store basic data in session
        session['nifty_latest'] = float(nifty_df['Close'].iloc[-1])
        session['nifty_prev'] = float(nifty_df['Close'].iloc[-2])
        session['vix_latest'] = float(vix_df['Close'].iloc[-1])
        session['vix_prev'] = float(vix_df['Close'].iloc[-2])
        session['data_loaded'] = True
        
        # Store data paths in session
        data_dir = 'data'
        session['nifty_csv_path'] = os.path.join(data_dir, 'nifty_data_consolidated.csv')
        session['vix_csv_path'] = os.path.join(data_dir, 'vix_data_consolidated.csv')
        
        flash("Data loaded successfully!", "success")
        
        # Instead of redirecting, render the dashboard directly with the loaded data
        nifty_df, vix_df = load_from_csv(session['nifty_csv_path'], session['vix_csv_path'])
        
        # Ensure data has consistent timezone
        if nifty_df.index.tz is None:
            nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
        if vix_df.index.tz is None:
            vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
            
        # Get date range
        today = datetime.now()
        end_date = nifty_df.index.max()
        start_date = end_date - timedelta(days=90)
        
        # Filter data for display
        mask = (nifty_df.index >= start_date) & (nifty_df.index <= end_date)
        filtered_nifty = nifty_df[mask]
        filtered_vix = vix_df[mask]
        
        # Prepare data for template
        nifty_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in filtered_nifty.index],
            'open': filtered_nifty['Open'].tolist(),
            'high': filtered_nifty['High'].tolist(),
            'low': filtered_nifty['Low'].tolist(),
            'close': filtered_nifty['Close'].tolist()
        }
        
        vix_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in filtered_vix.index],
            'open': filtered_vix['Open'].tolist(),
            'high': filtered_vix['High'].tolist(),
            'low': filtered_vix['Low'].tolist(),
            'close': filtered_vix['Close'].tolist()
        }
        
        # Calculate metrics
        latest_nifty = float(nifty_df['Close'].iloc[-1])
        prev_nifty = float(nifty_df['Close'].iloc[-2])
        nifty_change = (latest_nifty / prev_nifty - 1) * 100
        
        latest_vix = float(vix_df['Close'].iloc[-1])
        prev_vix = float(vix_df['Close'].iloc[-2])
        vix_change = (latest_vix / prev_vix - 1) * 100
        
        # Before the return statement, add this final length check
        if len(nifty_data['dates']) != len(nifty_data['close']) or len(vix_data['dates']) != len(vix_data['close']):
            flash("Warning: Data arrays have inconsistent lengths. Fixing...", "warning")
            # Find the minimum length across all arrays
            min_len = min(
                len(nifty_data['dates']), len(nifty_data['open']), len(nifty_data['high']), 
                len(nifty_data['low']), len(nifty_data['close']),
                len(vix_data['dates']), len(vix_data['open']), len(vix_data['high']), 
                len(vix_data['low']), len(vix_data['close'])
            )
            
            # Truncate all arrays to the minimum length
            nifty_data['dates'] = nifty_data['dates'][:min_len]
            nifty_data['open'] = nifty_data['open'][:min_len]
            nifty_data['high'] = nifty_data['high'][:min_len]
            nifty_data['low'] = nifty_data['low'][:min_len]
            nifty_data['close'] = nifty_data['close'][:min_len]
            
            vix_data['dates'] = vix_data['dates'][:min_len]
            vix_data['open'] = vix_data['open'][:min_len]
            vix_data['high'] = vix_data['high'][:min_len]
            vix_data['low'] = vix_data['low'][:min_len]
            vix_data['close'] = vix_data['close'][:min_len]
        
        return render_template('dashboard.html',
                               chart_type='line',
                               active_tab='nifty',
                               start_date=start_date.strftime('%Y-%m-%d'),
                               end_date=end_date.strftime('%Y-%m-%d'),
                               nifty_data=nifty_data,
                               vix_data=vix_data,
                               nifty_latest_price=latest_nifty,
                               vix_latest_price=latest_vix,
                               nifty_change=nifty_change,
                               vix_change=vix_change,
                               latest_date=nifty_df.index[-1].date(),
                               today=today.date(),
                               prediction=None,
                               trade_history=None,
                               trade_metrics=None,
                               data_loaded=True,
                               model_loaded=session.get('model_loaded', False))
                               
    except Exception as e:
        flash(f"Error loading data: {str(e)}", "error")
        # Return to index with default values instead of redirecting
        today = datetime.now()
        return render_template('dashboard.html',
                           chart_type='line', 
                           active_tab='nifty',
                           start_date=(today - timedelta(days=90)).strftime('%Y-%m-%d'),
                           end_date=today.strftime('%Y-%m-%d'),
                           nifty_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                           vix_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                           nifty_latest_price=0,
                           vix_latest_price=0,
                           nifty_change=0,
                           vix_change=0, 
                           latest_date=today.date(),
                           today=today.date(),
                           prediction=None,
                           trade_history=None,
                           trade_metrics=None,
                           data_loaded=False,
                           model_loaded=session.get('model_loaded', False))

@bp.route('/load_model')
def load_model():
    """Load AI model"""
    try:
        if not session.get('data_loaded'):
            flash("Please load data first.", "warning")
            # Return to index with default values instead of redirecting
            today = datetime.now()
            return render_template('dashboard.html',
                               chart_type='line',
                               active_tab='nifty',
                               start_date=(today - timedelta(days=90)).strftime('%Y-%m-%d'),
                               end_date=today.strftime('%Y-%m-%d'),
                               nifty_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                               vix_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                               nifty_latest_price=0,
                               vix_latest_price=0,
                               nifty_change=0,
                               vix_change=0,
                               latest_date=today.date(),
                               today=today.date(),
                               prediction=None,
                               trade_history=None,
                               trade_metrics=None,
                               data_loaded=False,
                               model_loaded=False)
            
        # Load data and model
        trader, nifty_df, vix_df = load_data_and_model(load_feature_extraction=True)
        
        if trader is None:
            flash("Error loading model. Please check the model files.", "error")
            # Return with data but no model
            today = datetime.now()
            # Load data files
            nifty_csv_path = session.get('nifty_csv_path')
            vix_csv_path = session.get('vix_csv_path')
            
            if not nifty_csv_path or not vix_csv_path:
                # If paths not in session, return to index
                return render_template('dashboard.html',
                                   chart_type='line',
                                   active_tab='nifty',
                                   start_date=(today - timedelta(days=90)).strftime('%Y-%m-%d'),
                                   end_date=today.strftime('%Y-%m-%d'),
                                   nifty_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                                   vix_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                                   nifty_latest_price=0,
                                   vix_latest_price=0,
                                   nifty_change=0,
                                   vix_change=0,
                                   latest_date=today.date(),
                                   today=today.date(),
                                   prediction=None,
                                   trade_history=None,
                                   trade_metrics=None,
                                   data_loaded=False,
                                   model_loaded=False)
            
            nifty_df, vix_df = load_from_csv(nifty_csv_path, vix_csv_path)
            
            # Ensure data has consistent timezone
            if nifty_df.index.tz is None:
                nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
            if vix_df.index.tz is None:
                vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
                
            # Get date range
            end_date = nifty_df.index.max()
            start_date = end_date - timedelta(days=90)
            
            # Filter data for display
            mask = (nifty_df.index >= start_date) & (nifty_df.index <= end_date)
            filtered_nifty = nifty_df[mask]
            filtered_vix = vix_df[mask]
            
            # Prepare data for template
            nifty_data = {
                'dates': [d.strftime('%Y-%m-%d') for d in filtered_nifty.index],
                'open': filtered_nifty['Open'].tolist(),
                'high': filtered_nifty['High'].tolist(),
                'low': filtered_nifty['Low'].tolist(),
                'close': filtered_nifty['Close'].tolist()
            }
            
            vix_data = {
                'dates': [d.strftime('%Y-%m-%d') for d in filtered_vix.index],
                'open': filtered_vix['Open'].tolist(),
                'high': filtered_vix['High'].tolist(),
                'low': filtered_vix['Low'].tolist(),
                'close': filtered_vix['Close'].tolist()
            }
            
            # Calculate metrics
            latest_nifty = float(nifty_df['Close'].iloc[-1])
            prev_nifty = float(nifty_df['Close'].iloc[-2])
            nifty_change = (latest_nifty / prev_nifty - 1) * 100
            
            latest_vix = float(vix_df['Close'].iloc[-1])
            prev_vix = float(vix_df['Close'].iloc[-2])
            vix_change = (latest_vix / prev_vix - 1) * 100
            
            # Check data consistency before returning
            if len(nifty_data['dates']) != len(nifty_data['close']) or len(vix_data['dates']) != len(vix_data['close']):
                flash("Warning: Data arrays have inconsistent lengths. Fixing...", "warning")
                # Find the minimum length across all arrays
                min_len = min(
                    len(nifty_data['dates']), len(nifty_data['open']), len(nifty_data['high']), 
                    len(nifty_data['low']), len(nifty_data['close']),
                    len(vix_data['dates']), len(vix_data['open']), len(vix_data['high']), 
                    len(vix_data['low']), len(vix_data['close'])
                )
                
                # Truncate all arrays to the minimum length
                nifty_data['dates'] = nifty_data['dates'][:min_len]
                nifty_data['open'] = nifty_data['open'][:min_len]
                nifty_data['high'] = nifty_data['high'][:min_len]
                nifty_data['low'] = nifty_data['low'][:min_len]
                nifty_data['close'] = nifty_data['close'][:min_len]
                
                vix_data['dates'] = vix_data['dates'][:min_len]
                vix_data['open'] = vix_data['open'][:min_len]
                vix_data['high'] = vix_data['high'][:min_len]
                vix_data['low'] = vix_data['low'][:min_len]
                vix_data['close'] = vix_data['close'][:min_len]
            
            return render_template('dashboard.html',
                           chart_type='line',
                           active_tab='nifty',
                           start_date=start_date.strftime('%Y-%m-%d'),
                           end_date=end_date.strftime('%Y-%m-%d'),
                           nifty_data=nifty_data,
                           vix_data=vix_data,
                           nifty_latest_price=latest_nifty,
                           vix_latest_price=latest_vix,
                           nifty_change=nifty_change,
                           vix_change=vix_change,
                           latest_date=nifty_df.index[-1].date(),
                           today=today.date(),
                           prediction=None,
                           trade_history=None,
                           trade_metrics=None,
                           data_loaded=True,
                           model_loaded=False)
            
        session['model_loaded'] = True
        flash("Model loaded successfully!", "success")
        
        # Generate prediction for the dashboard
        prediction = None
        try:
            # Make sure the environment is properly initialized and data is prepared
            if trader.env and not trader.env.features_list:
                trader.env._prepare_data()
                
            # Get state and prediction
            if trader.env and trader.env.features_list:
                state = trader.env.get_state(len(trader.env.features_list) - 1)
                if state is not None:
                    pred = trader.predict(state, use_mcts=True)
                    
                    action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
                    action_text = action_map.get(pred.get('action', 2), "UNKNOWN")
                    
                    prediction = {
                        'action': action_text,
                        'confidence': pred.get('confidence', 0)
                    }
                    
                    # If action is BUY or SELL, get trade info
                    if action_text != "HOLD":
                        # Run a test step to get the levels
                        test_env = TradingEnvironment(
                            nifty_data=nifty_df, 
                            vix_data=vix_df,
                            features_extractor=trader.features_extractor,
                            trade_time=session.get('trade_time', '9:15'),
                            lot_size=session.get('lot_size', 50),
                            initial_capital=session.get('initial_capital', 100000),
                            test_mode=False
                        )
                        test_env.reset()
                        # Move to last state
                        test_env.current_idx = len(test_env.features_list) - 1
                        # Take action
                        _, _, _, info = test_env.step(pred.get('action', 2))
                        
                        # Extract trade info
                        if 'entry_price' in info:
                            trade_info = {
                                'entry_price': info['entry_price'],
                                'lot_size': test_env.lot_size
                            }
                            
                            if 'stop_loss' in info:
                                trade_info['stop_loss'] = info['stop_loss']
                                trade_info['sl_pct'] = abs(info['stop_loss'] / info['entry_price'] - 1) * 100
                                
                            if 'take_profit' in info:
                                trade_info['take_profit'] = info['take_profit']
                                trade_info['tp_pct'] = abs(info['take_profit'] / info['entry_price'] - 1) * 100
                                
                            prediction['trade_info'] = trade_info
        except Exception as e:
            flash(f"Warning: Could not generate prediction: {e}", "warning")
        
        # Get chart data
        today = datetime.now()
        end_date = nifty_df.index.max()
        start_date = end_date - timedelta(days=90)
        
        # Filter data for display
        mask = (nifty_df.index >= start_date) & (nifty_df.index <= end_date)
        filtered_nifty = nifty_df[mask]
        filtered_vix = vix_df[mask]
        
        # Prepare data for template
        nifty_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in filtered_nifty.index],
            'open': filtered_nifty['Open'].tolist(),
            'high': filtered_nifty['High'].tolist(),
            'low': filtered_nifty['Low'].tolist(),
            'close': filtered_nifty['Close'].tolist()
        }
        
        vix_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in filtered_vix.index],
            'open': filtered_vix['Open'].tolist(),
            'high': filtered_vix['High'].tolist(),
            'low': filtered_vix['Low'].tolist(),
            'close': filtered_vix['Close'].tolist()
        }
        
        # Calculate metrics
        latest_nifty = float(nifty_df['Close'].iloc[-1])
        prev_nifty = float(nifty_df['Close'].iloc[-2])
        nifty_change = (latest_nifty / prev_nifty - 1) * 100
        
        latest_vix = float(vix_df['Close'].iloc[-1])
        prev_vix = float(vix_df['Close'].iloc[-2])
        vix_change = (latest_vix / prev_vix - 1) * 100
        
        # Before the final return statement when loading the model succeeds
        if len(nifty_data['dates']) != len(nifty_data['close']) or len(vix_data['dates']) != len(vix_data['close']):
            flash("Warning: Data arrays have inconsistent lengths. Fixing...", "warning")
            # Find the minimum length across all arrays
            min_len = min(
                len(nifty_data['dates']), len(nifty_data['open']), len(nifty_data['high']), 
                len(nifty_data['low']), len(nifty_data['close']),
                len(vix_data['dates']), len(vix_data['open']), len(vix_data['high']), 
                len(vix_data['low']), len(vix_data['close'])
            )
            
            # Truncate all arrays to the minimum length
            nifty_data['dates'] = nifty_data['dates'][:min_len]
            nifty_data['open'] = nifty_data['open'][:min_len]
            nifty_data['high'] = nifty_data['high'][:min_len]
            nifty_data['low'] = nifty_data['low'][:min_len]
            nifty_data['close'] = nifty_data['close'][:min_len]
            
            vix_data['dates'] = vix_data['dates'][:min_len]
            vix_data['open'] = vix_data['open'][:min_len]
            vix_data['high'] = vix_data['high'][:min_len]
            vix_data['low'] = vix_data['low'][:min_len]
            vix_data['close'] = vix_data['close'][:min_len]
        
        return render_template('dashboard.html',
                           chart_type='line',
                           active_tab='nifty',
                           start_date=start_date.strftime('%Y-%m-%d'),
                           end_date=end_date.strftime('%Y-%m-%d'),
                           nifty_data=nifty_data,
                           vix_data=vix_data,
                           nifty_latest_price=latest_nifty,
                           vix_latest_price=latest_vix,
                           nifty_change=nifty_change,
                           vix_change=vix_change,
                           latest_date=nifty_df.index[-1].date(),
                           today=today.date(),
                           prediction=prediction,
                           trade_history=None,
                           trade_metrics=None,
                           data_loaded=True,
                           model_loaded=True)
                               
    except Exception as e:
        flash(f"Error loading model: {str(e)}", "error")
        # Return to dashboard with data but no model
        today = datetime.now()
        
        # Try to load data
        nifty_csv_path = session.get('nifty_csv_path')
        vix_csv_path = session.get('vix_csv_path')
        
        if not nifty_csv_path or not vix_csv_path:
            # If paths not in session, return to index with defaults
            return render_template('dashboard.html',
                               chart_type='line',
                               active_tab='nifty',
                               start_date=(today - timedelta(days=90)).strftime('%Y-%m-%d'),
                               end_date=today.strftime('%Y-%m-%d'),
                               nifty_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                               vix_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                               nifty_latest_price=0,
                               vix_latest_price=0,
                               nifty_change=0,
                               vix_change=0,
                               latest_date=today.date(),
                               today=today.date(),
                               prediction=None,
                               trade_history=None,
                               trade_metrics=None,
                               data_loaded=session.get('data_loaded', False),
                               model_loaded=False)
        
        try:
            nifty_df, vix_df = load_from_csv(nifty_csv_path, vix_csv_path)
            
            # Ensure data has consistent timezone
            if nifty_df.index.tz is None:
                nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
            if vix_df.index.tz is None:
                vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
                
            # Get date range
            end_date = nifty_df.index.max()
            start_date = end_date - timedelta(days=90)
            
            # Filter data for display
            mask = (nifty_df.index >= start_date) & (nifty_df.index <= end_date)
            filtered_nifty = nifty_df[mask]
            filtered_vix = vix_df[mask]
            
            # Prepare data for template
            nifty_data = {
                'dates': [d.strftime('%Y-%m-%d') for d in filtered_nifty.index],
                'open': filtered_nifty['Open'].tolist(),
                'high': filtered_nifty['High'].tolist(),
                'low': filtered_nifty['Low'].tolist(),
                'close': filtered_nifty['Close'].tolist()
            }
            
            vix_data = {
                'dates': [d.strftime('%Y-%m-%d') for d in filtered_vix.index],
                'open': filtered_vix['Open'].tolist(),
                'high': filtered_vix['High'].tolist(),
                'low': filtered_vix['Low'].tolist(),
                'close': filtered_vix['Close'].tolist()
            }
            
            # Calculate metrics
            latest_nifty = float(nifty_df['Close'].iloc[-1])
            prev_nifty = float(nifty_df['Close'].iloc[-2])
            nifty_change = (latest_nifty / prev_nifty - 1) * 100
            
            latest_vix = float(vix_df['Close'].iloc[-1])
            prev_vix = float(vix_df['Close'].iloc[-2])
            vix_change = (latest_vix / prev_vix - 1) * 100
            
            # Check data consistency before returning
            if len(nifty_data['dates']) != len(nifty_data['close']) or len(vix_data['dates']) != len(vix_data['close']):
                flash("Warning: Data arrays have inconsistent lengths. Fixing...", "warning")
                # Find the minimum length across all arrays
                min_len = min(
                    len(nifty_data['dates']), len(nifty_data['open']), len(nifty_data['high']), 
                    len(nifty_data['low']), len(nifty_data['close']),
                    len(vix_data['dates']), len(vix_data['open']), len(vix_data['high']), 
                    len(vix_data['low']), len(vix_data['close'])
                )
                
                # Truncate all arrays to the minimum length
                nifty_data['dates'] = nifty_data['dates'][:min_len]
                nifty_data['open'] = nifty_data['open'][:min_len]
                nifty_data['high'] = nifty_data['high'][:min_len]
                nifty_data['low'] = nifty_data['low'][:min_len]
                nifty_data['close'] = nifty_data['close'][:min_len]
                
                vix_data['dates'] = vix_data['dates'][:min_len]
                vix_data['open'] = vix_data['open'][:min_len]
                vix_data['high'] = vix_data['high'][:min_len]
                vix_data['low'] = vix_data['low'][:min_len]
                vix_data['close'] = vix_data['close'][:min_len]
            
            return render_template('dashboard.html',
                            chart_type='line',
                            active_tab='nifty',
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            nifty_data=nifty_data,
                            vix_data=vix_data,
                            nifty_latest_price=latest_nifty,
                            vix_latest_price=latest_vix,
                            nifty_change=nifty_change,
                            vix_change=vix_change,
                            latest_date=nifty_df.index[-1].date(),
                            today=today.date(),
                            prediction=None,
                            trade_history=None,
                            trade_metrics=None,
                            data_loaded=True,
                            model_loaded=False)
        except Exception as inner_e:
            flash(f"Error loading data: {str(inner_e)}", "error")
            return render_template('dashboard.html',
                              chart_type='line',
                              active_tab='nifty',
                              start_date=(today - timedelta(days=90)).strftime('%Y-%m-%d'),
                              end_date=today.strftime('%Y-%m-%d'),
                              nifty_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                              vix_data={'dates': [], 'open': [], 'high': [], 'low': [], 'close': []},
                              nifty_latest_price=0,
                              vix_latest_price=0,
                              nifty_change=0,
                              vix_change=0,
                              latest_date=today.date(),
                              today=today.date(),
                              prediction=None,
                              trade_history=None,
                              trade_metrics=None,
                              data_loaded=False,
                              model_loaded=False)

@bp.route('/dashboard')
def dashboard():
    """Dashboard with loaded data"""
    # If data not loaded, redirect to index with a flag to prevent redirect loops
    if not session.get('data_loaded'):
        # Instead of redirecting, we'll just render the index page directly with all required variables
        flash("Please load data first.", "warning")
        
        # Setup defaults for template variables
        today = datetime.now()
        start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        # Default empty data structures
        nifty_data = {
            'dates': [],
            'open': [],
            'high': [],
            'low': [],
            'close': []
        }
        
        vix_data = {
            'dates': [],
            'open': [],
            'high': [],
            'low': [],
            'close': []
        }
        
        return render_template('dashboard.html',
                           chart_type=request.args.get('chart_type', 'line'),
                           active_tab='nifty',
                           start_date=start_date,
                           end_date=end_date,
                           nifty_data=nifty_data,
                           vix_data=vix_data,
                           nifty_latest_price=0,
                           vix_latest_price=0,
                           nifty_change=0,
                           vix_change=0,
                           latest_date=today.date(),
                           today=today.date(),
                           prediction=None,
                           trade_history=None,
                           trade_metrics=None,
                           data_loaded=False,
                           model_loaded=False)
        
    # Get chart type and date range from query params
    chart_type = request.args.get('chart_type', 'line')
    
    # Load data and prepare for display
    try:
        # Load data files
        nifty_csv_path = session.get('nifty_csv_path')
        vix_csv_path = session.get('vix_csv_path')
        
        nifty_df, vix_df = load_from_csv(nifty_csv_path, vix_csv_path)
        
        # Ensure data has consistent timezone
        if nifty_df.index.tz is None:
            nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
        if vix_df.index.tz is None:
            vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
        
        # Get the date range for the chart
        today = datetime.now()
        end_date = request.args.get('end_date', nifty_df.index.max().strftime('%Y-%m-%d'))
        try:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            # Localize end_date to Asia/Kolkata timezone
            end_date = pd.Timestamp(end_date).tz_localize('Asia/Kolkata')
        except ValueError:
            end_date = nifty_df.index.max()
        
        start_date = request.args.get('start_date', (end_date - timedelta(days=90)).strftime('%Y-%m-%d'))
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            # Localize start_date to Asia/Kolkata timezone
            start_date = pd.Timestamp(start_date).tz_localize('Asia/Kolkata')
        except ValueError:
            start_date = (end_date - timedelta(days=90))
        
        # Filter data based on date range
        mask = (nifty_df.index >= start_date) & (nifty_df.index <= end_date)
        filtered_nifty = nifty_df[mask]
        filtered_vix = vix_df[mask]
        
        # Prepare data for the template
        nifty_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in filtered_nifty.index],
            'open': filtered_nifty['Open'].tolist(),
            'high': filtered_nifty['High'].tolist(),
            'low': filtered_nifty['Low'].tolist(),
            'close': filtered_nifty['Close'].tolist()
        }
        
        vix_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in filtered_vix.index],
            'open': filtered_vix['Open'].tolist(),
            'high': filtered_vix['High'].tolist(),
            'low': filtered_vix['Low'].tolist(),
            'close': filtered_vix['Close'].tolist()
        }
        
        # Calculate metrics for summary display
        latest_nifty = nifty_df['Close'].iloc[-1]
        prev_nifty = nifty_df['Close'].iloc[-2]
        nifty_change = (latest_nifty / prev_nifty - 1) * 100
        
        latest_vix = vix_df['Close'].iloc[-1]
        prev_vix = vix_df['Close'].iloc[-2]
        vix_change = (latest_vix / prev_vix - 1) * 100
        
        # Get prediction if model is loaded
        prediction = None
        latest_date = nifty_df.index[-1].date()
        today = datetime.now().date()
        
        if session.get('model_loaded'):
            try:
                trader, _, _ = load_data_and_model(load_feature_extraction=True)
                if trader is not None:
                    # Initialize environment if it doesn't exist
                    if trader.env is None:
                        try:
                            trader.env = TradingEnvironment(
                                nifty_data=nifty_df, 
                                vix_data=vix_df,
                                features_extractor=trader.features_extractor,
                                window_size=10,
                                trade_time=session.get('trade_time', '9:15'),
                                lot_size=session.get('lot_size', 50),
                                initial_capital=session.get('initial_capital', 100000),
                                test_mode=False
                            )
                        except Exception as e:
                            flash(f"Error initializing environment: {e}", "error")
                            
                    # Make sure the environment is properly initialized and data is prepared
                    if trader.env and not trader.env.features_list:
                        trader.env._prepare_data()
                        
                    # Get state and prediction
                    if trader.env and trader.env.features_list:
                        state = trader.env.get_state(len(trader.env.features_list) - 1)
                        if state is not None:
                            pred = trader.predict(state, use_mcts=True)
                            
                            action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
                            action_text = action_map.get(pred.get('action', 2), "UNKNOWN")
                            
                            prediction = {
                                'action': action_text,
                                'confidence': pred.get('confidence', 0)
                            }
                            
                            # If action is BUY or SELL, get trade info
                            if action_text != "HOLD":
                                # Run a test step to get the levels
                                test_env = TradingEnvironment(
                                    nifty_data=nifty_df, 
                                    vix_data=vix_df,
                                    features_extractor=trader.features_extractor,
                                    trade_time=session.get('trade_time', '9:15'),
                                    lot_size=session.get('lot_size', 50),
                                    initial_capital=session.get('initial_capital', 100000),
                                    test_mode=False
                                )
                                test_env.reset()
                                # Move to last state
                                test_env.current_idx = len(test_env.features_list) - 1
                                # Take action
                                _, _, _, info = test_env.step(pred.get('action', 2))
                                
                                # Extract trade info
                                if 'entry_price' in info:
                                    trade_info = {
                                        'entry_price': info['entry_price'],
                                        'lot_size': test_env.lot_size
                                    }
                                    
                                    if 'stop_loss' in info:
                                        trade_info['stop_loss'] = info['stop_loss']
                                        trade_info['sl_pct'] = abs(info['stop_loss'] / info['entry_price'] - 1) * 100
                                        
                                    if 'take_profit' in info:
                                        trade_info['take_profit'] = info['take_profit']
                                        trade_info['tp_pct'] = abs(info['take_profit'] / info['entry_price'] - 1) * 100
                                        
                                    prediction['trade_info'] = trade_info
            except Exception as e:
                flash(f"Error generating prediction: {e}", "error")
        
        # Get trading history if available
        trade_history = None
        trade_metrics = None
        
        # TODO: We'll implement trade history later once backtesting is in place
        
        # Before the final return statement in the dashboard route, add this check
        if len(nifty_data['dates']) != len(nifty_data['close']) or len(vix_data['dates']) != len(vix_data['close']):
            flash("Warning: Data arrays have inconsistent lengths. Fixing...", "warning")
            # Find the minimum length across all arrays
            min_len = min(
                len(nifty_data['dates']), len(nifty_data['open']), len(nifty_data['high']), 
                len(nifty_data['low']), len(nifty_data['close']),
                len(vix_data['dates']), len(vix_data['open']), len(vix_data['high']), 
                len(vix_data['low']), len(vix_data['close'])
            )
            
            # Truncate all arrays to the minimum length
            nifty_data['dates'] = nifty_data['dates'][:min_len]
            nifty_data['open'] = nifty_data['open'][:min_len]
            nifty_data['high'] = nifty_data['high'][:min_len]
            nifty_data['low'] = nifty_data['low'][:min_len]
            nifty_data['close'] = nifty_data['close'][:min_len]
            
            vix_data['dates'] = vix_data['dates'][:min_len]
            vix_data['open'] = vix_data['open'][:min_len]
            vix_data['high'] = vix_data['high'][:min_len]
            vix_data['low'] = vix_data['low'][:min_len]
            vix_data['close'] = vix_data['close'][:min_len]
        
        return render_template('dashboard.html',
                               chart_type=chart_type,
                               active_tab='nifty',
                               start_date=start_date.strftime('%Y-%m-%d'),
                               end_date=end_date.strftime('%Y-%m-%d'),
                               nifty_data=nifty_data,
                               vix_data=vix_data,
                               nifty_latest_price=latest_nifty,
                               vix_latest_price=latest_vix,
                               nifty_change=nifty_change,
                               vix_change=vix_change,
                               latest_date=latest_date,
                               today=today,
                               prediction=prediction,
                               trade_history=trade_history,
                               trade_metrics=trade_metrics,
                               data_loaded=session.get('data_loaded', False),
                               model_loaded=session.get('model_loaded', False))
                               
    except Exception as e:
        flash(f"Error displaying dashboard: {str(e)}", "error")
        return redirect(url_for('main.index'))

def load_market_data():
    """Load market data from CSV files"""
    try:
        # Try to load data directly from consolidated CSV files
        data_dir = os.environ.get('DATA_DIR', 'data')
        nifty_csv_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
        vix_csv_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
        
        if os.path.exists(nifty_csv_path) and os.path.exists(vix_csv_path):
            # Load directly from CSV files using the improved data_loader
            from src.data_processing.data_loader import load_from_csv
            nifty_df, vix_df = load_from_csv(nifty_csv_path, vix_csv_path)
            print(f"Loaded market data from CSV files: {len(nifty_df) if nifty_df is not None else 0} NIFTY records")
        else:
            # Fall back to get_latest_data which will handle fetching if needed
            from src.data_processing.data_loader import get_latest_data
            nifty_df, vix_df = get_latest_data(data_dir=data_dir, load_from_disk=True)
            print(f"Loaded market data using get_latest_data: {len(nifty_df) if nifty_df is not None else 0} NIFTY records")
        
        # The new load_from_csv and get_latest_data functions handle all the alignment and processing
        # so we no longer need the explicit timezone and alignment code here
        
        if nifty_df is None or vix_df is None:
            flash("Error loading data. Please check the data files.", "error")
            return None, None
            
        return nifty_df, vix_df
            
    except Exception as e:
        flash(f"Error loading market data: {e}", "error")
        import traceback
        traceback.print_exc()
        return None, None

def load_data_and_model(load_feature_extraction=False):
    """Load data and model for the app"""
    try:
        # Load market data using our dedicated function
        nifty_df, vix_df = load_market_data()
        
        if nifty_df is None or vix_df is None:
            flash("Error loading data. Please check the data files.", "error")
            return None, None, None
        
        # Skip feature extraction unless needed
        if not load_feature_extraction:
            return None, nifty_df, vix_df
        
        # Load features extractor
        features_extractor = get_features_extractor()
        
        # Create a temporary environment to determine input shape
        temp_env = TradingEnvironment(
            nifty_data=nifty_df,
            vix_data=vix_df,
            features_extractor=features_extractor,
            window_size=int(os.environ.get('WINDOW_SIZE', 10)),
            trade_time=os.environ.get('TRADE_TIME', '9:15'),
            test_mode=False
        )
        
        # Get the shape from the first valid state
        if not temp_env.features_list:
            temp_env._prepare_data()
            
        if not temp_env.features_list:
            flash("No features extracted from data. Using default shape.", "warning")
            input_shape = (1, 1, 19)
        else:
            test_state = temp_env.get_state(0)
            if test_state is not None:
                input_shape = test_state.shape
            else:
                flash("Could not get state from environment. Using default shape.", "warning")
                input_shape = (1, 1, 19)
            
        # Initialize the trader
        trader = AlphaZeroTrader(
            input_shape=input_shape,
            n_actions=3,
            features_extractor=features_extractor,
            model_dir=os.environ.get('MODEL_DIR', 'models')
        )
        
        try:
            # Load the model
            model_loaded = trader.load_model(os.environ.get('MODEL_FILE', 'alphazero_model'))
            if not model_loaded:
                flash("Could not load model. Will try to initialize a new one.", "warning")
            
            # Make sure trader has an environment
            trader.env = TradingEnvironment(
                nifty_data=nifty_df,
                vix_data=vix_df,
                features_extractor=features_extractor,
                window_size=int(os.environ.get('WINDOW_SIZE', 10)),
                trade_time=session.get('trade_time', '9:15'),
                lot_size=int(session.get('lot_size', 50)),
                initial_capital=int(session.get('initial_capital', 100000)),
                test_mode=False
            )
            
            # Make sure the environment is properly initialized
            if not trader.env.features_list:
                trader.env._prepare_data()
                
            return trader, nifty_df, vix_df
            
        except Exception as e:
            flash(f"Error loading data and model: {e}", "error")
            import traceback
            traceback.print_exc()
            return None, nifty_df, vix_df
    
    except Exception as e:
        flash(f"Error loading data and model: {e}", "error")
        import traceback
        traceback.print_exc()
        return None, None, None 