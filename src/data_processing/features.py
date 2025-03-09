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
    # Last row contains the current day
    current_day = window.iloc[-1]
    
    # Calculate technical indicators
    nifty_returns = window['nifty_close'].pct_change().fillna(0)
    vix_returns = window['vix_close'].pct_change().fillna(0)
    
    # Moving averages
    nifty_ma5 = window['nifty_close'].rolling(5).mean().fillna(method='bfill')
    nifty_ma10 = window['nifty_close'].rolling(10).mean().fillna(method='bfill')
    vix_ma5 = window['vix_close'].rolling(5).mean().fillna(method='bfill')
    
    # Volatility measures
    nifty_volatility = nifty_returns.rolling(5).std().fillna(0)
    vix_volatility = vix_returns.rolling(5).std().fillna(0)
    
    # Price ratios
    price_to_ma5 = window['nifty_close'] / nifty_ma5
    price_to_ma10 = window['nifty_close'] / nifty_ma10
    
    # VIX features
    vix_ratio = window['vix_close'] / vix_ma5
    vix_percentile = window['vix_close'].rank(pct=True)
    
    # Create feature vector (use the last day in the window)
    features = np.array([
        nifty_returns.iloc[-1],                                       # 1-day return
        nifty_returns.iloc[-5:].mean(),                               # 5-day mean return
        nifty_volatility.iloc[-1],                                    # Recent volatility
        price_to_ma5.iloc[-1] - 1,                                    # Deviation from 5-day MA
        price_to_ma10.iloc[-1] - 1,                                   # Deviation from 10-day MA
        vix_returns.iloc[-1],                                         # 1-day VIX return
        vix_ratio.iloc[-1] - 1,                                       # VIX deviation from 5-day MA
        vix_percentile.iloc[-1],                                      # VIX rank percentile
        current_day['vix_close'],                                     # Current VIX level
        current_day['nifty_volume'] / window['nifty_volume'].mean(),  # Volume ratio
    ])
    
    # Reshape to be compatible with Conv1D input (samples, timesteps, features)
    return features.reshape(1, -1)
    
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
    nifty_ma20 = window['nifty_close'].rolling(20).mean().fillna(method='bfill')
    nifty_ma50 = window['nifty_close'].rolling(min(50, len(window))).mean().fillna(method='bfill')
    
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