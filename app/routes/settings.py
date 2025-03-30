from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify
import sys
import os
import json
import shutil
from datetime import datetime
import configparser

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import AppConfig

bp = Blueprint('settings', __name__, url_prefix='/settings')

def load_user_settings():
    """Load user settings from user_config.ini if it exists, otherwise use defaults"""
    # Default settings from AppConfig
    settings = {
        'lot_size': AppConfig.DEFAULT_LOT_SIZE,
        'initial_capital': AppConfig.DEFAULT_INITIAL_CAPITAL,
        'trade_time': AppConfig.TRADE_TIME,
    }
    
    # Check if user_config.ini exists in the instance directory
    instance_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'instance')
    config_path = os.path.join(instance_path, 'user_config.ini')
    
    if os.path.exists(config_path):
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            if 'Trading' in config:
                if 'lot_size' in config['Trading']:
                    settings['lot_size'] = int(config['Trading']['lot_size'])
                if 'initial_capital' in config['Trading']:
                    settings['initial_capital'] = int(config['Trading']['initial_capital'])
                if 'trade_time' in config['Trading']:
                    settings['trade_time'] = config['Trading']['trade_time']
        except Exception as e:
            print(f"Error loading user settings: {e}")
    
    return settings

def save_user_settings(settings):
    """Save user settings to user_config.ini"""
    try:
        # Create instance directory if it doesn't exist
        instance_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'instance')
        os.makedirs(instance_path, exist_ok=True)
        
        config_path = os.path.join(instance_path, 'user_config.ini')
        
        config = configparser.ConfigParser()
        
        # Load existing config if it exists
        if os.path.exists(config_path):
            config.read(config_path)
        
        # Create Trading section if it doesn't exist
        if 'Trading' not in config:
            config['Trading'] = {}
        
        # Update settings
        config['Trading']['lot_size'] = str(settings['lot_size'])
        config['Trading']['initial_capital'] = str(settings['initial_capital'])
        config['Trading']['trade_time'] = settings['trade_time']
        
        # Save config
        with open(config_path, 'w') as f:
            config.write(f)
            
        return True
    except Exception as e:
        print(f"Error saving user settings: {e}")
        return False

@bp.route('/')
def index():
    """Settings page"""
    # First load settings from user_config.ini or defaults
    default_settings = load_user_settings()
    
    # Then override with session values if they exist
    settings = {
        'lot_size': session.get('lot_size', default_settings['lot_size']),
        'initial_capital': session.get('initial_capital', default_settings['initial_capital']),
        'trade_time': session.get('trade_time', default_settings['trade_time']),
    }
    
    # Additional settings from AppConfig for display
    config_info = {
        'env': os.environ.get('FLASK_ENV', 'development'),
        'data_dir': AppConfig.DATA_DIR,
        'model_dir': AppConfig.MODEL_DIR,
        'window_size': AppConfig.WINDOW_SIZE,
    }
    
    return render_template('settings.html', settings=settings, config_info=config_info)

@bp.route('/save', methods=['POST'])
def save_settings():
    """Save settings"""
    try:
        # Get settings from form
        lot_size = int(request.form.get('lot_size', 50))
        initial_capital = int(request.form.get('initial_capital', 100000))
        trade_time = request.form.get('trade_time', '9:15')
        
        # Validate
        if lot_size <= 0:
            flash("Lot size must be positive", "error")
            return redirect(url_for('settings.index'))
            
        if initial_capital <= 0:
            flash("Initial capital must be positive", "error")
            return redirect(url_for('settings.index'))
        
        # Save settings to session
        session['lot_size'] = lot_size
        session['initial_capital'] = initial_capital
        session['trade_time'] = trade_time
        
        # Save settings to user_config.ini
        settings = {
            'lot_size': lot_size,
            'initial_capital': initial_capital,
            'trade_time': trade_time
        }
        
        if save_user_settings(settings):
            flash("Settings saved successfully", "success")
        else:
            flash("Settings saved to session but could not be saved to file", "warning")
            
    except Exception as e:
        flash(f"Error saving settings: {e}", "error")
    
    return redirect(url_for('settings.index'))

@bp.route('/reset', methods=['POST'])
def reset_settings():
    """Reset settings to default values"""
    try:
        # Clear specific settings from session
        if 'lot_size' in session:
            session.pop('lot_size')
        if 'initial_capital' in session:
            session.pop('initial_capital')
        if 'trade_time' in session:
            session.pop('trade_time')
        if 'data_loaded' in session:
            session.pop('data_loaded')
        if 'model_loaded' in session:
            session.pop('model_loaded')
        
        # Save default settings to user_config.ini
        default_settings = {
            'lot_size': AppConfig.DEFAULT_LOT_SIZE,
            'initial_capital': AppConfig.DEFAULT_INITIAL_CAPITAL,
            'trade_time': AppConfig.TRADE_TIME
        }
        
        if save_user_settings(default_settings):
            flash("Settings reset to defaults successfully", "success")
        else:
            flash("Settings reset in session but could not update configuration file", "warning")
            
    except Exception as e:
        flash(f"Error resetting settings: {e}", "error")
    
    return redirect(url_for('settings.index')) 