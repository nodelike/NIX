import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def extract_features(window):
    """
    Extract features from price data window
    
    Args:
        window: DataFrame with price data window (multiple days)
        
    Returns:
        numpy array of features
    """
    # Ensure we have enough data
    if len(window) < 5:
        raise ValueError(f"Not enough data points in window: {len(window)} (need at least 5)")
    
    # Handle missing values
    window = window.ffill().bfill()
    
    # Last row contains the current day
    current_day = window.iloc[-1]
    
    # Calculate technical indicators with safety checks
    # Add a small epsilon to denominators to avoid division by zero
    epsilon = 1e-8
    
    # Calculate returns safely
    nifty_returns = window['nifty_close'].pct_change().fillna(0)
    vix_returns = window['vix_close'].pct_change().fillna(0)
    
    # Moving averages
    nifty_ma5 = window['nifty_close'].rolling(5).mean().bfill()
    nifty_ma10 = window['nifty_close'].rolling(10).mean().bfill()
    vix_ma5 = window['vix_close'].rolling(5).mean().bfill()
    
    # Volatility measures
    nifty_volatility = nifty_returns.rolling(5).std().fillna(0)
    vix_volatility = vix_returns.rolling(5).std().fillna(0)
    
    # Price ratios with safety checks
    price_to_ma5 = window['nifty_close'] / (nifty_ma5 + epsilon)
    price_to_ma10 = window['nifty_close'] / (nifty_ma10 + epsilon)
    
    # VIX features with safety checks
    vix_ratio = window['vix_close'] / (vix_ma5 + epsilon)
    vix_percentile = window['vix_close'].rank(pct=True)
    
    # Get default value for volume if missing
    volume_mean = window['nifty_volume'].replace(0, np.nan).mean()
    if np.isnan(volume_mean) or volume_mean == 0:
        volume_mean = 1.0  # Default value to avoid division by zero
        
    # Create feature vector (use the last day in the window)
    features = np.array([
        nifty_returns.iloc[-1],  # 1-day return
        nifty_returns.iloc[-5:].mean(),  # 5-day mean return
        nifty_volatility.iloc[-1],  # Recent volatility
        price_to_ma5.iloc[-1] - 1,  # Deviation from 5-day MA
        price_to_ma10.iloc[-1] - 1,  # Deviation from 10-day MA
        vix_returns.iloc[-1],  # 1-day VIX return
        vix_ratio.iloc[-1] - 1,  # VIX deviation from 5-day MA
        vix_percentile.iloc[-1],  # VIX rank percentile
        current_day['vix_close'],  # Current VIX level
        current_day['nifty_volume'] / volume_mean,  # Volume ratio
    ])
    
    # Replace any NaN or inf values with zeros
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Reshape to be compatible with Conv1D input (samples, timesteps, features)
    # Return as (1, 1, n_features) - batch size, time steps, features
    return features.reshape(1, 1, -1)
    
def extract_additional_features(window):
    """
    Extract extended features with more technical indicators
    
    Args:
        window: DataFrame with price data window
        
    Returns:
        numpy array of features
    """
    # Basic features
    basic_features = extract_features(window).flatten()
    
    # Last row contains the current day
    current_day = window.iloc[-1]
    prev_day = window.iloc[-2] if len(window) > 1 else current_day
    
    # Calculate additional technical indicators
    nifty_returns = window['nifty_close'].pct_change().fillna(0)
    
    # Momentum indicators
    rsi_period = 14
    delta = window['nifty_close'].diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean().fillna(0)
    loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean().fillna(0)
    
    # Calculate RSI (avoid division by zero)
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Trend indicators
    nifty_ma20 = window['nifty_close'].rolling(20).mean().bfill()
    nifty_ma50 = window['nifty_close'].rolling(min(50, len(window))).mean().bfill()
    
    # Gap analysis
    gap = (window['nifty_open'] - window['nifty_close'].shift(1)) / window['nifty_close'].shift(1)
    gap = gap.fillna(0)
    
    # VIX momentum
    vix_rsi_period = 5
    vix_delta = window['vix_close'].diff().fillna(0)
    vix_gain = vix_delta.where(vix_delta > 0, 0).rolling(window=vix_rsi_period).mean().fillna(0)
    vix_loss = -vix_delta.where(vix_delta < 0, 0).rolling(window=vix_rsi_period).mean().fillna(0)
    vix_rs = vix_gain / (vix_loss + 1e-10)
    vix_rsi = 100 - (100 / (1 + vix_rs))
    
    # Additional features
    additional_features = np.array([
        rsi.iloc[-1] / 100,                                       # RSI (normalized)
        (current_day['nifty_close'] / nifty_ma20.iloc[-1]) - 1,   # Distance from 20-day MA
        (current_day['nifty_close'] / nifty_ma50.iloc[-1]) - 1,   # Distance from 50-day MA
        gap.iloc[-1],                                             # Today's gap
        vix_rsi.iloc[-1] / 100,                                   # VIX RSI (normalized)
        current_day['nifty_high'] / current_day['nifty_low'] - 1, # Day's range
        (current_day['nifty_close'] - current_day['nifty_open']) / current_day['nifty_open'] # Day's return
    ])
    
    # Combine features
    combined_features = np.concatenate([basic_features, additional_features])
    
    # Reshape for Conv1D
    return combined_features.reshape(1, -1)
    
def create_market_summary(window):
    """
    Create a human-readable market summary from features
    
    Args:
        window: DataFrame with price data window
        
    Returns:
        String with market summary
    """
    # Extract key statistics
    current_day = window.iloc[-1]
    prev_day = window.iloc[-2] if len(window) > 1 else current_day
    
    nifty_return = (current_day['nifty_close'] / prev_day['nifty_close'] - 1) * 100
    vix_change = (current_day['vix_close'] / prev_day['vix_close'] - 1) * 100
    
    # Calculate technical indicators
    nifty_ma5 = window['nifty_close'].rolling(5).mean().iloc[-1]
    nifty_ma20 = window['nifty_close'].rolling(min(20, len(window))).mean().iloc[-1]
    
    above_ma5 = current_day['nifty_close'] > nifty_ma5
    above_ma20 = current_day['nifty_close'] > nifty_ma20
    
    # Determine trend
    trend = "NEUTRAL"
    if above_ma5 and above_ma20 and nifty_return > 0:
        trend = "BULLISH"
    elif not above_ma5 and not above_ma20 and nifty_return < 0:
        trend = "BEARISH"
    
    # Generate summary
    summary = (
        f"NIFTY: {current_day['nifty_close']:.2f} ({nifty_return:+.2f}%) | "
        f"INDIA VIX: {current_day['vix_close']:.2f} ({vix_change:+.2f}%)\n"
        f"Market Trend: {trend}\n"
        f"NIFTY vs 5-day MA: {'Above' if above_ma5 else 'Below'} | "
        f"NIFTY vs 20-day MA: {'Above' if above_ma20 else 'Below'}\n"
        f"Today's Range: {current_day['nifty_low']:.2f} - {current_day['nifty_high']:.2f}"
    )
    
    return summary 

def extract_trading_time_features(window, nifty_df, vix_df, trade_time='09:05'):
    """
    Extract features focused on the 9:05 AM trading decision, including
    previous day's last 30 minutes and 9:00-9:05 data.
    
    Args:
        window: DataFrame with daily data
        nifty_df: Complete NIFTY DataFrame with minute-level data
        vix_df: Complete VIX DataFrame with minute-level data
        trade_time: Time of day to trade (format: HH:MM)
        
    Returns:
        numpy array of features
    """
    # Get the current day from the window
    current_day = window.iloc[-1]
    current_date = current_day.name.date() if hasattr(current_day.name, 'date') else current_day['date']
    
    # Get previous day
    if len(window) > 1:
        prev_day = window.iloc[-2]
        prev_date = prev_day.name.date() if hasattr(prev_day.name, 'date') else prev_day['date']
    else:
        # If no previous day data available, use current day as fallback
        prev_date = current_date
    
    # Convert dates to string for filtering
    prev_date_str = str(prev_date)
    current_date_str = str(current_date)
    
    # Extract features from standard daily data
    daily_features = extract_features(window).reshape(-1)
    
    # --- Previous day's last 30 minutes ---
    # Filter nifty data for previous day's last 30 minutes (15:00-15:30)
    prev_day_last_30min_nifty = nifty_df[
        (nifty_df.index.date == prev_date) & 
        (nifty_df.index.time >= pd.Timestamp('15:00').time()) &
        (nifty_df.index.time <= pd.Timestamp('15:30').time())
    ]
    
    # Filter vix data for previous day's last 30 minutes
    prev_day_last_30min_vix = vix_df[
        (vix_df.index.date == prev_date) & 
        (vix_df.index.time >= pd.Timestamp('15:00').time()) &
        (vix_df.index.time <= pd.Timestamp('15:30').time())
    ]
    
    # --- Current day's 9:00-9:05 data ---
    # Filter nifty data for current day's 9:00-9:05
    current_day_early_nifty = nifty_df[
        (nifty_df.index.date == current_date) & 
        (nifty_df.index.time >= pd.Timestamp('09:00').time()) &
        (nifty_df.index.time <= pd.Timestamp('09:05').time())
    ]
    
    # Filter vix data for current day's 9:00-9:05
    current_day_early_vix = vix_df[
        (vix_df.index.date == current_date) & 
        (vix_df.index.time >= pd.Timestamp('09:00').time()) &
        (vix_df.index.time <= pd.Timestamp('09:05').time())
    ]
    
    # --- Extract features from the filtered data ---
    # Features from previous day's last 30 minutes
    prev_day_last_30min_features = []
    if len(prev_day_last_30min_nifty) > 0 and len(prev_day_last_30min_vix) > 0:
        # Get return
        if prev_day_last_30min_nifty['Open'].iloc[0] > 0:
            prev_day_nifty_return = (prev_day_last_30min_nifty['Close'].iloc[-1] / prev_day_last_30min_nifty['Open'].iloc[0] - 1)
        else:
            prev_day_nifty_return = 0
            
        if prev_day_last_30min_vix['Open'].iloc[0] > 0:
            prev_day_vix_return = (prev_day_last_30min_vix['Close'].iloc[-1] / prev_day_last_30min_vix['Open'].iloc[0] - 1)
        else:
            prev_day_vix_return = 0
        
        # Calculate volatility
        prev_day_nifty_volatility = prev_day_last_30min_nifty['Close'].pct_change().std()
        
        # Calculate volume trend (handle division by zero)
        if prev_day_last_30min_nifty['Volume'].iloc[0] > 0:
            prev_day_volume_trend = prev_day_last_30min_nifty['Volume'].iloc[-1] / prev_day_last_30min_nifty['Volume'].iloc[0]
        else:
            prev_day_volume_trend = 1.0  # Default to no change if first volume is zero
            
        prev_day_last_30min_features = [
            prev_day_nifty_return,
            prev_day_vix_return,
            prev_day_nifty_volatility,
            prev_day_volume_trend
        ]
    else:
        # Default values if data not available
        prev_day_last_30min_features = [0, 0, 0, 0]
    
    # Features from current day's 9:00-9:05
    early_morning_features = []
    if len(current_day_early_nifty) > 0 and len(current_day_early_vix) > 0:
        # Get open to 9:05 return with safety checks
        if current_day_early_nifty['Open'].iloc[0] > 0:
            morning_nifty_return = (current_day_early_nifty['Close'].iloc[-1] / current_day_early_nifty['Open'].iloc[0] - 1)
        else:
            morning_nifty_return = 0
            
        if current_day_early_vix['Open'].iloc[0] > 0:
            morning_vix_return = (current_day_early_vix['Close'].iloc[-1] / current_day_early_vix['Open'].iloc[0] - 1)
        else:
            morning_vix_return = 0
        
        # Calculate first 5 minutes volatility
        morning_nifty_volatility = current_day_early_nifty['Close'].pct_change().std()
        
        # Calculate first 5 minutes volume relative to previous day's last 30 min
        morning_volume_ratio = 0
        if len(prev_day_last_30min_nifty) > 0:
            prev_avg_volume = prev_day_last_30min_nifty['Volume'].mean()
            morning_volume = current_day_early_nifty['Volume'].mean()
            if prev_avg_volume > 0:
                morning_volume_ratio = morning_volume / prev_avg_volume
        
        # Check for gap up/down
        gap = 0
        if len(prev_day_last_30min_nifty) > 0:
            prev_close = prev_day_last_30min_nifty['Close'].iloc[-1]
            today_open = current_day_early_nifty['Open'].iloc[0]
            if prev_close > 0:
                gap = (today_open / prev_close - 1)
        
        early_morning_features = [
            morning_nifty_return,
            morning_vix_return,
            morning_nifty_volatility,
            morning_volume_ratio,
            gap
        ]
    else:
        # Default values if data not available
        early_morning_features = [0, 0, 0, 0, 0]
    
    # Combine all features
    all_features = np.concatenate([
        daily_features,
        prev_day_last_30min_features,
        early_morning_features
    ])
    
    # Replace any NaN or inf values with zeros
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure we have the right shape for the model (1, 1, num_features)
    return all_features.reshape(1, 1, -1) 

def get_features_extractor():
    """Return the extract_features function for use by other modules."""
    return extract_features 