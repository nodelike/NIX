import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from tslearn.shapelets import ShapeletModel
from imblearn.over_sampling import SMOTE
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def monte_carlo_split(data_length, n_splits=30, test_size=5):
    """Monte Carlo Time Series Validation."""
    if data_length < 10:  # Too small for meaningful splits
        return [(np.arange(data_length-2), np.arange(data_length-2, data_length))]
    
    indices = np.arange(data_length)
    splits = []
    min_train = max(5, data_length // 2)  # Ensure minimum training size
    
    for _ in range(n_splits):
        # Ensure we have enough samples for both train and test
        valid_starts = np.arange(min_train, data_length - test_size + 1)
        if len(valid_starts) == 0:
            start = min_train
        else:
            start = np.random.choice(valid_starts)
        
        train_idx = indices[:start]
        test_idx = indices[start:start + test_size]
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    # If no valid splits were created, use a simple split
    if not splits:
        split_point = data_length - test_size
        return [(indices[:split_point], indices[split_point:])]
    
    return splits

def calculate_dynamic_levels(vix_data):
    """Calculate key percentage levels from recent volatility"""
    recent_vix = vix_data.groupby('date').agg({
        'High': 'max',
        'Low': 'min'
    }).last('5D')
    
    return {
        'aggressive_call': recent_vix['High'].quantile(0.8),
        'conservative_call': recent_vix['High'].quantile(0.6),
        'aggressive_put': recent_vix['Low'].quantile(0.2),
        'conservative_put': recent_vix['Low'].quantile(0.4)
    }

def calculate_position_size(entry_time, probability):
    """Dynamic position sizing based on remaining trading window and probability"""
    entry_time = pd.Timestamp(entry_time)
    remaining_time = (pd.Timestamp(entry_time.date()) + pd.Timedelta('09:45:00') - entry_time).total_seconds() / 60
    
    # Base position size on time
    if remaining_time > 25:
        base_size = 0.15  # Max risk
    elif remaining_time > 15:
        base_size = 0.10
    else:
        base_size = 0.05  # Minimum position after 09:30
    
    # Adjust by probability confidence
    confidence = abs(probability - 0.5) * 2
    return base_size * confidence

def get_first_5min_features(df):
    """Extract enhanced features from first 5 minutes and previous day's last 30 minutes."""
    df['date'] = df.index.date
    daily_groups = df.groupby('date')
    
    features_list = []
    for date, group in daily_groups:
        # Get first 5 minutes of current day
        day_data = group.between_time('09:15', '09:20')
        if len(day_data) >= 5:
            first_5_min = day_data.iloc[:5]
            first_candle = day_data.iloc[0]  # First minute candle
            
            # Calculate current day features
            features = {
                'vix_open': first_5_min['Open'].iloc[0],
                'vix_high': first_5_min['High'].max(),
                'vix_low': first_5_min['Low'].min(),
                'vix_close': first_5_min['Close'].iloc[-1],
                'vix_volatility': first_5_min['Close'].std(),
                'vix_pct_change': (first_5_min['Close'].iloc[-1] / first_5_min['Close'].iloc[0] - 1) * 100,
                'vix_range': (first_5_min['High'].max() - first_5_min['Low'].min()) / first_5_min['Open'].iloc[0] * 100,
                'vix_oc_ratio': (first_5_min['Close'].iloc[-1] - first_5_min['Open'].iloc[0]) / first_5_min['Open'].iloc[0],
                'first_candle_range': (first_candle['High'] - first_candle['Low']) / first_candle['Open'] * 100,
                'first_candle_body': abs(first_candle['Close'] - first_candle['Open']) / first_candle['Open'] * 100,
                'first_candle_direction': 1 if first_candle['Close'] > first_candle['Open'] else -1
            }
            
            # Get previous day's data
            prev_date = pd.Timestamp(date) - pd.Timedelta(days=1)
            if prev_date in daily_groups.groups:
                prev_day_data = daily_groups.get_group(prev_date).between_time('15:00', '15:30')
                if len(prev_day_data) > 0:
                    prev_close = prev_day_data['Close'].iloc[-1]
                    prev_high = prev_day_data['High'].max()
                    prev_low = prev_day_data['Low'].min()
                    
                    features.update({
                        'prev_day_close': prev_close,
                        'prev_day_high': prev_high,
                        'prev_day_low': prev_low,
                        'prev_day_volatility': prev_day_data['Close'].std(),
                        'prev_day_trend': (prev_day_data['Close'].iloc[-1] / prev_day_data['Close'].iloc[0] - 1) * 100,
                        'prev_day_range': (prev_high - prev_low) / prev_day_data['Open'].iloc[0] * 100,
                        'gap_pct': (first_5_min['Open'].iloc[0] / prev_close - 1) * 100,
                        'gap_above_high': (first_5_min['Open'].iloc[0] > prev_high) * 1,
                        'gap_below_low': (first_5_min['Open'].iloc[0] < prev_low) * 1,
                        'prev_day_ATR': prev_high - prev_low
                    })
                else:
                    features.update({
                        'prev_day_close': first_5_min['Open'].iloc[0],
                        'prev_day_high': first_5_min['Open'].iloc[0],
                        'prev_day_low': first_5_min['Open'].iloc[0],
                        'prev_day_volatility': 0,
                        'prev_day_trend': 0,
                        'prev_day_range': 0,
                        'gap_pct': 0,
                        'gap_above_high': 0,
                        'gap_below_low': 0,
                        'prev_day_ATR': 0
                    })
            else:
                features.update({
                    'prev_day_close': first_5_min['Open'].iloc[0],
                    'prev_day_high': first_5_min['Open'].iloc[0],
                    'prev_day_low': first_5_min['Open'].iloc[0],
                    'prev_day_volatility': 0,
                    'prev_day_trend': 0,
                    'prev_day_range': 0,
                    'gap_pct': 0,
                    'gap_above_high': 0,
                    'gap_below_low': 0,
                    'prev_day_ATR': 0
                })
            
            features_list.append((date, features))
    
    if not features_list:
        logger.warning("No valid trading days found!")
        return pd.DataFrame()
    
    result = pd.DataFrame([f[1] for f in features_list], index=[pd.Timestamp(d) for d, _ in features_list])
    logger.info(f"Generated features for {len(result)} trading days out of {len(daily_groups)}")
    return result

def get_nifty_target(df):
    """Calculate 15-minute return for NIFTY after first 5 minutes"""
    df['date'] = df.index.date
    targets = []
    
    for date, group in df.groupby('date'):
        entry_period = group.between_time('09:15', '09:20')
        target_period = group.between_time('09:20', '09:45')
        
        if len(entry_period) >= 5 and len(target_period) >= 15:
            entry_price = entry_period['Close'].iloc[-1]  # End of 5th minute
            high_target = target_period['High'].max()
            low_target = target_period['Low'].min()
            
            # Calculate max potential in both directions
            upside_pct = (high_target/entry_price - 1) * 100
            downside_pct = (1 - low_target/entry_price) * 100
            
            # Determine trend based on closing price
            final_price = target_period['Close'].iloc[-1]
            trend = 1 if final_price > entry_price else 0
            
            targets.append((date, upside_pct, downside_pct, trend))
    
    if not targets:
        logger.warning("No valid target days found!")
        return pd.DataFrame()
    
    result = pd.DataFrame(targets, 
                         columns=['date', 'upside_pct', 'downside_pct', 'trend']
                        ).set_index('date')
    return result

def generate_trading_signals(prediction_proba, current_vix, prev_day_data):
    """Generate entry triggers with stop-loss and targets"""
    prev_high = prev_day_data['High'].max()
    prev_low = prev_day_data['Low'].min()
    current_price = current_vix['Close'].iloc[-1]
    
    # Calculate probabilistic ranges
    call_level = prev_high + (0.3 * prediction_proba)
    put_level = prev_low - (0.3 * (1 - prediction_proba))
    
    signals = []
    
    # CALL strategy parameters
    if current_price > prev_high:
        sl = prev_low
        target = current_price + (2 * (current_price - sl))
        signals.append(('CALL', target, sl))
    
    # PUT strategy parameters    
    elif current_price < prev_low:
        sl = prev_high
        target = current_price - (2 * (sl - current_price))
        signals.append(('PUT', target, sl))
        
    # Mean-reversion play
    elif (current_price - prev_low)/(prev_high - prev_low) > 0.7:
        signals.append(('PUT', prev_low, prev_high))
        
    elif (current_price - prev_low)/(prev_high - prev_low) < 0.3:
        signals.append(('CALL', prev_high, prev_low))
    
    return signals

def vectorized_backtest(df):
    """Fast backtesting using vectorized operations"""
    df['signal'] = 0
    df['returns'] = 0.0
    
    # Adjust thresholds for signal generation
    vix_std = df['vix_volatility'].mean()
    
    # CALL signals - More aggressive thresholds
    call_mask = ((df['vix_open'] > df['prev_day_high']) | 
                 ((df['gap_pct'] > 0.5) & (df['first_candle_direction'] == 1)))
    df.loc[call_mask, 'signal'] = 1
    df.loc[call_mask, 'returns'] = df['upside_pct'] * 0.75
    
    # PUT signals - More aggressive thresholds
    put_mask = ((df['vix_open'] < df['prev_day_low']) | 
                ((df['gap_pct'] < -0.5) & (df['first_candle_direction'] == -1)))
    df.loc[put_mask, 'signal'] = -1
    df.loc[put_mask, 'returns'] = df['downside_pct'] * 0.75
    
    # Mean reversion signals with adjusted thresholds
    high_zone = ((df['vix_open'] - df['prev_day_low'])/(df['prev_day_high'] - df['prev_day_low']) > 0.6)
    low_zone = ((df['vix_open'] - df['prev_day_low'])/(df['prev_day_high'] - df['prev_day_low']) < 0.4)
    
    # Only take mean reversion trades with strong momentum
    df.loc[high_zone & (df['first_candle_direction'] == -1), 'signal'] = -1
    df.loc[high_zone & (df['first_candle_direction'] == -1), 'returns'] = df['downside_pct'] * 0.5
    
    df.loc[low_zone & (df['first_candle_direction'] == 1), 'signal'] = 1
    df.loc[low_zone & (df['first_candle_direction'] == 1), 'returns'] = df['upside_pct'] * 0.5
    
    # Count number of trades
    n_trades = len(df[df['signal'] != 0])
    
    # If no trades were taken, return default metrics
    if n_trades == 0:
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'avg_return': 0.0,
            'std_return': 0.0
        }
    
    # Calculate metrics
    win_rate = (df['returns'] > 0).mean()
    
    # Calculate profit factor (handle case where there are no losses)
    total_gains = df.loc[df['returns'] > 0, 'returns'].sum()
    total_losses = abs(df.loc[df['returns'] < 0, 'returns'].sum())
    profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
    
    # Calculate Sharpe ratio (annualized)
    returns_mean = df['returns'].mean()
    returns_std = df['returns'].std()
    sharpe = (returns_mean / returns_std * np.sqrt(252)) if returns_std != 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = df['returns'].cumsum()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns - rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'total_trades': n_trades,
        'avg_return': returns_mean,
        'std_return': returns_std
    }

try:
    logger.info("Loading and preparing data...")
    # Read the data
    nifty_df = pd.read_csv('nifty_1min_data.csv')
    vix_df = pd.read_csv('vix_1min_data.csv')

    # Data Preparation & Cleaning
    for df in [nifty_df, vix_df]:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)
        
    # Filter market hours only (9:15 AM to 3:30 PM IST)
    nifty_df = nifty_df.between_time('09:15', '15:30')
    vix_df = vix_df.between_time('09:15', '15:30')

    # Get unique dates using pandas datetime index
    unique_dates = pd.Series(nifty_df.index.date).nunique()
    logger.info(f"Total trading days in data: {unique_dates}")

    if unique_dates < 10:
        raise ValueError("Insufficient data: Need at least 10 trading days")

    logger.info("Generating features and targets...")
    # Generate VIX features
    vix_features = get_first_5min_features(vix_df)
    logger.info(f"Generated features for {len(vix_features)} trading days")

    # Generate NIFTY target
    nifty_target = get_nifty_target(nifty_df)
    logger.info(f"Generated targets for {len(nifty_target)} trading days")

    # Ensure data alignment
    common_dates = vix_features.index.intersection(nifty_target.index)
    vix_features = vix_features.loc[common_dates]
    nifty_target = nifty_target.loc[common_dates]

    # Merge features and target
    merged_df = pd.concat([vix_features, nifty_target], axis=1).dropna()
    logger.info(f"Final dataset size: {len(merged_df)} trading days")

    # Run backtest on raw signals
    logger.info("\nRunning backtest on raw signals...")
    backtest_results = vectorized_backtest(merged_df)
    logger.info("\nBacktest Results:")
    for metric, value in backtest_results.items():
        logger.info(f"{metric:15}: {value:0.4f}")

    # Enhanced Model Training
    logger.info("\nTraining enhanced model...")
    features = [col for col in merged_df.columns if col not in ['trend', 'upside_pct', 'downside_pct']]
    X = merged_df[features].values
    y = merged_df[['upside_pct', 'downside_pct']].values

    # Create multi-output regressor
    model = MultiOutputRegressor(
        TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )
        )
    )

    # Use walk-forward validation
    n_splits = min(5, max(2, len(merged_df) // 5))
    splits = monte_carlo_split(len(X), n_splits=n_splits, test_size=max(2, len(X) // 5))

    logger.info(f"Using {len(splits)} folds with {len(X)} samples")

    # Model Evaluation
    logger.info("\nModel Performance Across Monte Carlo Splits:")
    logger.info("-" * 50)
    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(splits, 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics for each output separately
        mse = np.mean((y_test - y_pred) ** 2, axis=0)
        rmse = np.sqrt(mse)
        
        # Calculate R² for each output separately
        r2_upside = model.estimators_[0].score(X_test, y_test[:, 0])
        r2_downside = model.estimators_[1].score(X_test, y_test[:, 1])
        
        fold_metrics.append({
            'fold': fold,
            'rmse_upside': rmse[0],
            'rmse_downside': rmse[1],
            'r2_upside': r2_upside,
            'r2_downside': r2_downside
        })
        
        logger.info(f"\nFold {fold}:")
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"RMSE Upside: {rmse[0]:.4f}")
        logger.info(f"RMSE Downside: {rmse[1]:.4f}")
        logger.info(f"R² Score Upside: {r2_upside:.4f}")
        logger.info(f"R² Score Downside: {r2_downside:.4f}")

    # Calculate average metrics
    avg_metrics = pd.DataFrame(fold_metrics).mean()
    logger.info("\nAverage Model Performance:")
    for metric, value in avg_metrics.items():
        if metric != 'fold':
            logger.info(f"{metric:15}: {value:0.4f}")

    # Feature Importance Analysis
    logger.info("\nAnalyzing feature importance...")
    importance_scores = []
    feature_importance_dict = {}
    
    # Calculate permutation importance
    for i, feature_name in enumerate(features):
        scores = []
        for fold, (train_index, test_index) in enumerate(splits):
            X_test_permuted = X[test_index].copy()
            baseline_score = model.estimators_[0].score(X[test_index], y[test_index, 0])
            
            # Permute the feature
            np.random.shuffle(X_test_permuted[:, i])
            permuted_score = model.estimators_[0].score(X_test_permuted, y[test_index, 0])
            
            # Importance is the decrease in score
            importance = baseline_score - permuted_score
            scores.append(importance)
        
        feature_importance_dict[feature_name] = np.mean(scores)
    
    # Convert to Series for plotting
    feature_importance = pd.Series(feature_importance_dict)
    
    plt.figure(figsize=(12, 8))
    feature_importance.sort_values(ascending=True).plot(kind='barh')
    plt.title('Feature Importance (Permutation Method)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # Display feature importance
    logger.info("\nTop 10 Most Important Features:")
    for feature, importance in feature_importance.nlargest(10).items():
        logger.info(f"{feature:20} : {importance:.3f}")

    def daily_prediction(live_vix, prev_day_data=None):
        """
        Make predictions using live VIX data with enhanced risk management.
        
        Args:
            live_vix (pd.DataFrame): First 5 minutes of live VIX data
            prev_day_data (pd.DataFrame): Previous day's last 30 minutes of data (optional)
            
        Returns:
            dict: Trading decision with entry, targets, and risk parameters
        """
        features = create_features(live_vix, prev_day_data)
        predictions = model.predict([features])[0]
        upside_potential, downside_potential = predictions
        
        current_time = live_vix.index[-1]
        position_size = calculate_position_size(current_time, max(upside_potential, downside_potential)/100)
        
        # Generate trading signals
        signals = generate_trading_signals(max(upside_potential, downside_potential)/100, 
                                        live_vix, 
                                        prev_day_data)
        
        if not signals:
            return {
                'action': 'NEUTRAL',
                'position_size': 0,
                'target': None,
                'stop_loss': None,
                'upside_potential': upside_potential,
                'downside_potential': downside_potential
            }
            
        direction, target, sl = signals[0]
        return {
            'action': direction,
            'position_size': position_size,
            'target': target,
            'stop_loss': sl,
            'upside_potential': upside_potential,
            'downside_potential': downside_potential
        }

    logger.info("\nAnalysis complete! The following files have been generated:")
    logger.info("1. correlation_heatmap.png - Shows feature correlations with target")
    logger.info("2. pair_plots.png - Visualizes relationships between key variables")
    logger.info("3. feature_importance.png - Shows relative importance of each feature")

    # Final warnings and recommendations
    if len(merged_df) < 100:
        logger.warning("\nWARNING: Current dataset is small. Use conservative position sizing.")
        logger.warning("Recommended: Collect more historical data if possible.")
    
    if backtest_results['win_rate'] < 0.6:
        logger.warning("\nWARNING: Win rate below 60%. Consider adjusting entry criteria.")
    
    if backtest_results['profit_factor'] < 1.5:
        logger.warning("\nWARNING: Profit factor below 1.5. Review risk management parameters.")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    logger.error("Analysis failed to complete. Please check the error message above.")
    raise