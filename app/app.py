import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
import base64

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import get_latest_data, consolidate_csv_files, update_data, get_data_summary, load_from_csv, merge_dataframes, save_to_csv
from src.data_processing.features import extract_features, create_market_summary, get_features_extractor
from src.alphazero.trader import AlphaZeroTrader
from src.alphazero.environment import TradingEnvironment

# Constants
INPUT_SHAPE = (1, 10)
DEFAULT_TRADE_TIME = "09:05"
PAGES = ["Dashboard", "Data Management", "Training", "Backtesting", "Prediction", "Settings"]

# Set page configuration
st.set_page_config(
    page_title="AlphaZero Trader for India VIX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for global variables
if 'trader' not in st.session_state:
    st.session_state.trader = None
if 'nifty_data' not in st.session_state:
    st.session_state.nifty_data = None
if 'vix_data' not in st.session_state:
    st.session_state.vix_data = None
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = {
        'episode': 0,
        'step': 0,
        'reward': 0,
        'total_reward': 0,
        'actions_count': [0, 0, 0],
        'total_episodes': 0
    }
if 'training_history' not in st.session_state:
    st.session_state.training_history = {
        'rewards': [],
        'actions': [],
        'policy_loss': [],
        'value_loss': [],
        'total_loss': []
    }
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Dashboard"
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None

# Define callback functions
def training_callback(info):
    """Callback function for training updates"""
    try:
        # Initialize training history if not already done
        if 'training_history' not in st.session_state:
            st.session_state.training_history = {
                'rewards': [],
                'actions': [],
                'policy_loss': [],
                'value_loss': [],
                'total_loss': []
            }
        
        # Handle self-play episode completion
        if 'episode_complete' in info and info['episode_complete']:
            # Add reward info
            if 'total_reward' in info:
                st.session_state.training_history['rewards'].append(info['total_reward'])
            
            # Add action distribution info
            if 'actions_count' in info:
                # Copy the actions_count to avoid modifying the original
                actions_count = dict(info['actions_count'])
                
                # Ensure actions are using string keys, not numeric indexes
                # Convert numeric keys to string action names if needed
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
                
                st.session_state.training_history['actions'].append(actions_count)
        
        # Handle training batch completion
        if 'batch_complete' in info and info['batch_complete']:
            # Add loss info
            if 'policy_loss' in info:
                if 'policy_loss' not in st.session_state.training_history:
                    st.session_state.training_history['policy_loss'] = []
                st.session_state.training_history['policy_loss'].append(info['policy_loss'])
                
            if 'value_loss' in info:
                if 'value_loss' not in st.session_state.training_history:
                    st.session_state.training_history['value_loss'] = []
                st.session_state.training_history['value_loss'].append(info['value_loss'])
                
            if 'total_loss' in info:
                if 'total_loss' not in st.session_state.training_history:
                    st.session_state.training_history['total_loss'] = []
                st.session_state.training_history['total_loss'].append(info['total_loss'])
        
        # Update progress information
        st.session_state.training_progress = info
        
        # Log for debugging
        print(f"Training callback processed: {info.keys()}")
        
    except Exception as e:
        print(f"Error in training callback: {e}")
        # Don't propagate exceptions from callbacks to avoid breaking the main process

def backtest_callback(info):
    """Callback function for backtest updates"""
    try:
        if 'complete' in info and info['complete']:
            # Backtest completed
            st.session_state.backtest_results = {
                'metrics': info['metrics'],
                'results_df': info['results_df']
            }
        else:
            # Update progress
            st.session_state.backtest_progress = info
        
        # We don't need to call rerun here as it can interfere with the backtesting process
    except Exception as e:
        print(f"Error in backtest callback: {e}")
        # Don't propagate exceptions from callbacks to avoid breaking the main process

# Helper functions
def load_data_and_model():
    """Load data and model for the app"""
    try:
        # Try to load data directly from consolidated CSV files
        data_dir = 'data'
        nifty_csv_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
        vix_csv_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
        
        if os.path.exists(nifty_csv_path) and os.path.exists(vix_csv_path):
            # Load directly from CSV files
            from src.data_processing.data_loader import load_from_csv
            nifty_df, vix_df = load_from_csv(nifty_csv_path, vix_csv_path)
            st.info(f"Loaded data from consolidated CSV files: {len(nifty_df)} NIFTY records, {len(vix_df)} VIX records")
        else:
            # Fall back to get_latest_data
            from src.data_processing.data_loader import get_latest_data
            nifty_df, vix_df = get_latest_data(load_from_disk=True)
            st.info(f"Loaded data using get_latest_data: {len(nifty_df)} NIFTY records, {len(vix_df)} VIX records")
        
        if nifty_df is None or vix_df is None:
            st.error("Error loading data. Please check the data files.")
            return None, None, None
        
        # Ensure data has consistent timezone
        if nifty_df.index.tz is None:
            nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
        if vix_df.index.tz is None:
            vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
            
        # Ensure indices are aligned
        common_dates = nifty_df.index.intersection(vix_df.index)
        if len(common_dates) == 0:
            st.error("No common dates between NIFTY and VIX data.")
            return None, None, None
            
        # Align data to common dates
        nifty_df = nifty_df.loc[common_dates]
        vix_df = vix_df.loc[common_dates]
        
        # Load features extractor
        from src.data_processing.features import get_features_extractor
        features_extractor = get_features_extractor()
        
        # Create a temporary environment to determine input shape
        from src.alphazero.environment import TradingEnvironment
        temp_env = TradingEnvironment(
            nifty_data=nifty_df,
            vix_data=vix_df,
            features_extractor=features_extractor,
            window_size=10,
            trade_time='9:15',
            test_mode=False
        )
        
        # Get the shape from the first valid state
        if not temp_env.features_list:
            temp_env._prepare_data()
            
        if not temp_env.features_list:
            st.warning("No features extracted from data. Using default shape.")
            input_shape = (1, 1, 19)
        else:
            test_state = temp_env.get_state(0)
            if test_state is not None:
                input_shape = test_state.shape
            else:
                st.warning("Could not get state from environment. Using default shape.")
                input_shape = (1, 1, 19)
            
        # Initialize the trader
        from src.alphazero.trader import AlphaZeroTrader
        trader = AlphaZeroTrader(
            input_shape=input_shape,
            n_actions=3,
            features_extractor=features_extractor
        )
        
        try:
            # Load the model
            model_loaded = trader.load_model('alphazero_model.h5')
            if not model_loaded:
                st.warning("Could not load model. Will try to initialize a new one.")
            
            # Make sure trader has an environment
            trader.env = TradingEnvironment(
                nifty_data=nifty_df,
                vix_data=vix_df,
                features_extractor=features_extractor,
                window_size=10,
                trade_time='9:15',
                lot_size=int(st.session_state.get('lot_size', 50)),
                initial_capital=int(st.session_state.get('initial_capital', 100000)),
                test_mode=False
            )
            
            # Make sure the environment is properly initialized
            if not trader.env.features_list:
                trader.env._prepare_data()
                
            return trader, nifty_df, vix_df
            
        except Exception as e:
            st.error(f"Error loading data and model: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    except Exception as e:
        st.error(f"Error loading data and model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def plot_training_history():
    """Plot training history"""
    if 'training_history' not in st.session_state or not st.session_state.training_history:
        st.info("No training history available. Train the model first.")
        return
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # Plot rewards
    if 'rewards' in st.session_state.training_history and len(st.session_state.training_history['rewards']) > 0:
        ax1 = fig.add_subplot(2, 2, 1)
        rewards = st.session_state.training_history['rewards']
        ax1.plot(range(1, len(rewards) + 1), rewards, 'b-')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
    
    # Plot action distribution
    if 'actions' in st.session_state.training_history and len(st.session_state.training_history['actions']) > 0:
        ax2 = fig.add_subplot(2, 2, 2)
        actions = st.session_state.training_history['actions']
        
        if len(actions) > 0 and isinstance(actions[0], dict):
            # Extract counts
            buy_counts = [a.get('buy', 0) for a in actions]
            sell_counts = [a.get('sell', 0) for a in actions]
            hold_counts = [a.get('hold', 0) for a in actions]
            
            episodes = range(1, len(actions) + 1)
            ax2.bar(episodes, buy_counts, label='Buy', alpha=0.7, color='green')
            ax2.bar(episodes, sell_counts, bottom=buy_counts, label='Sell', alpha=0.7, color='red')
            ax2.bar(episodes, hold_counts, bottom=[b+s for b, s in zip(buy_counts, sell_counts)], 
                    label='Hold', alpha=0.7, color='blue')
            
            ax2.set_title('Action Distribution')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Count')
            ax2.legend()
    
    # Plot loss values
    losses_available = any(
        k in st.session_state.training_history and len(st.session_state.training_history[k]) > 0 
        for k in ['policy_loss', 'value_loss', 'total_loss']
    )
    
    if losses_available:
        ax3 = fig.add_subplot(2, 2, 3)
        
        if 'policy_loss' in st.session_state.training_history and len(st.session_state.training_history['policy_loss']) > 0:
            policy_loss = st.session_state.training_history['policy_loss']
            ax3.plot(range(1, len(policy_loss) + 1), policy_loss, 'r-', label='Policy Loss')
        
        if 'value_loss' in st.session_state.training_history and len(st.session_state.training_history['value_loss']) > 0:
            value_loss = st.session_state.training_history['value_loss']
            ax3.plot(range(1, len(value_loss) + 1), value_loss, 'g-', label='Value Loss')
        
        ax3.set_title('Training Loss')
        ax3.set_xlabel('Batch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
    
    # Add latest metrics as text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    text_info = []
    
    # Get latest metrics
    if ('rewards' in st.session_state.training_history and 
        len(st.session_state.training_history['rewards']) > 0):
        text_info.append(f"Latest Reward: {st.session_state.training_history['rewards'][-1]:.4f}")
    
    if ('policy_loss' in st.session_state.training_history and 
        len(st.session_state.training_history['policy_loss']) > 0):
        text_info.append(f"Latest Policy Loss: {st.session_state.training_history['policy_loss'][-1]:.4f}")
    
    if ('value_loss' in st.session_state.training_history and 
        len(st.session_state.training_history['value_loss']) > 0):
        text_info.append(f"Latest Value Loss: {st.session_state.training_history['value_loss'][-1]:.4f}")
    
    if text_info:
        ax4.text(0.1, 0.5, '\n'.join(text_info), fontsize=12)
    else:
        ax4.text(0.1, 0.5, "Training in progress...", fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_backtest_results():
    """Plot the backtest results with enhanced trade visualization including stop loss and take profit levels"""
    if st.session_state.backtest_results is None:
        st.info("No backtest results available. Run a backtest first.")
        return
    
    try:
        # Get data from session state
        if 'results_df' not in st.session_state.backtest_results or 'metrics' not in st.session_state.backtest_results:
            st.warning("Incomplete backtest results found.")
            return
    
        results_df = st.session_state.backtest_results['results_df']
        metrics = st.session_state.backtest_results['metrics']
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{metrics['total_return']:.2%}")
        col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        col4.metric("Total Trades", metrics['total_trades'])
        
        # Create backtest chart
        fig = go.Figure()
        
        # Check if 'capital' is in the DataFrame
        if 'capital' in results_df.columns:
            # Add equity curve
            fig.add_trace(
                go.Scatter(
                    x=results_df['date'],
                    y=results_df['capital'],
                    mode='lines',
                    name='Capital',
                    line=dict(color='blue')
                )
            )
            
            # Add price chart on secondary axis if available
            if 'price' in results_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=results_df['date'],
                        y=results_df['price'],
                        mode='lines',
                        name='NIFTY',
                        line=dict(color='gray', width=1),
                        yaxis='y2'
                    )
                )
            
            # Update layout with second y-axis
            fig.update_layout(
                title='Backtest Results',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Capital (₹)', side='left'),
                yaxis2=dict(title='NIFTY', overlaying='y', side='right'),
                legend=dict(x=0, y=1, orientation='h'),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display trade history if available
            if 'trade_history' in st.session_state.backtest_results and st.session_state.backtest_results['trade_history']:
                trades = st.session_state.backtest_results['trade_history']
                trades_df = pd.DataFrame(trades)
                
                # Display trade statistics
                st.subheader("Trade Statistics")
                
                # Create simple trade dashboard
                if len(trades_df) > 0:
                    # Calculate trade results if necessary fields exist
                    if all(col in trades_df.columns for col in ['action', 'date', 'price']):
                        # Track actions distribution
                        action_counts = trades_df['action'].value_counts()
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Display action distribution pie chart
                            fig_actions = px.pie(
                                values=action_counts.values,
                                names=action_counts.index,
                                title="Trading Action Distribution"
                            )
                            st.plotly_chart(fig_actions, use_container_width=True)
                        
                        with col2:
                            # Calculate basic stats if capital column exists
                            if 'return' in trades_df.columns:
                                winning_trades = len(trades_df[trades_df['return'] > 0])
                                total_trades = len(trades_df)
                                win_ratio = winning_trades / total_trades if total_trades > 0 else 0
                                avg_return = trades_df['return'].mean() if 'return' in trades_df.columns else 0
                                
                                st.metric("Win Ratio", f"{win_ratio:.2%}")
                                st.metric("Average Return", f"{avg_return:.2%}")
                    
                    # Display the trade table
                    st.subheader("Trade Details")
                    st.dataframe(trades_df)
            else:
                st.info("No detailed trade history available in the backtest results.")
        else:
            st.warning("No capital data available in backtest results.")
            return
    except Exception as e:
        st.error(f"Error displaying backtest results: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")

# Sidebar for navigation
def sidebar():
    """Sidebar with navigation and settings"""
    with st.sidebar:
        st.title("AlphaZero Trader")
    
        # Navigation
        page = st.radio("Navigation", [
            "Dashboard", 
            "Training",
            "Backtesting",
            "Prediction",
            "Data Management",
            "Settings"
        ])
        
        st.markdown("---")
        
        # Model and trading parameters
        st.subheader("Trading Parameters")
        
        # Lot size for trading
        st.session_state.lot_size = st.number_input(
            "Lot Size", 
            min_value=1, 
            max_value=1000, 
            value=st.session_state.get('lot_size', 50),
            help="Number of shares per trade"
        )
        
        # Initial capital
        st.session_state.initial_capital = st.number_input(
            "Initial Capital",
            min_value=10000,
            max_value=10000000,
            value=st.session_state.get('initial_capital', 100000),
            step=10000,
            help="Starting capital for backtesting"
        )
        
        # Trade time
        trade_time_options = {'9:15 AM': '9:15', '9:30 AM': '9:30', '10:00 AM': '10:00'}
        selected_time = st.selectbox(
            "Trade Time",
            list(trade_time_options.keys()),
            index=0,
            help="Time of day to place trades"
        )
        st.session_state.trade_time = trade_time_options[selected_time]
        
        st.markdown("---")
        
        # Data stats
        st.subheader("Data Summary")
        
        try:
            summary = get_data_summary()
            if summary:
                st.markdown(f"""
                **NIFTY Records:** {summary['nifty_count']}  
                **VIX Records:** {summary['vix_count']}  
                **Date Range:** {summary['start_date']} to {summary['end_date']}
                """)
            else:
                st.warning("No data available")
        except:
            st.warning("Error loading data summary")
            
        # Refresh data button
        if st.button("Refresh Data"):
            # Clear session state to reload everything
            st.session_state.trader = None
            st.session_state.nifty_data = None
            st.session_state.vix_data = None
            st.rerun()
    
        # Footer
        st.markdown("---")
        st.markdown("### AlphaZero Trader v1.0")
        st.markdown("Implementation of AlphaZero for trading")
        
        return page

# Pages implementation
def dashboard_page():
    """Dashboard page with overview"""
    st.title("AlphaZero Trader Dashboard")
    
    # Ensure data and model are loaded
    if 'trader' not in st.session_state or st.session_state.trader is None or \
       'nifty_data' not in st.session_state or st.session_state.nifty_data is None:
        try:
            with st.spinner("Loading data and model..."):
                trader, nifty_df, vix_df = load_data_and_model()
                st.session_state.trader = trader
                st.session_state.nifty_data = nifty_df
                st.session_state.vix_data = vix_df
        except Exception as e:
            st.error(f"Error loading data or model: {e}")
            st.info("No trained model found. You need to train a model first.")
            
            # Button to go to training page
            if st.button("Go to Training Page"):
                st.session_state.current_page = "Training"
                st.rerun()
            return
    else:
        trader = st.session_state.trader
        nifty_df = st.session_state.nifty_data
        vix_df = st.session_state.vix_data
    
    # Check if we have valid data
    if nifty_df is None or vix_df is None or len(nifty_df) == 0 or len(vix_df) == 0:
        st.warning("No data available. Please fetch data first.")
        if st.button("Fetch Data"):
            st.session_state.trader = None
            st.session_state.nifty_data = None
            st.session_state.vix_data = None
            st.rerun()
        return
        
    # Ensure the two dataframes have the same index to avoid length mismatches
    common_dates = nifty_df.index.intersection(vix_df.index)
    if len(common_dates) < len(nifty_df) or len(common_dates) < len(vix_df):
        st.warning(f"Aligning Nifty and VIX data to {len(common_dates)} common dates.")
        nifty_df = nifty_df.loc[common_dates]
        vix_df = vix_df.loc[common_dates]
        # Update session state with aligned data
        st.session_state.nifty_data = nifty_df
        st.session_state.vix_data = vix_df
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        latest_nifty = nifty_df['Close'].iloc[-1]
        prev_nifty = nifty_df['Close'].iloc[-2]
        nifty_change = (latest_nifty / prev_nifty - 1) * 100
        nifty_color = "green" if nifty_change >= 0 else "red"
        st.markdown(f"""
        ### NIFTY
        <p style='font-size:24px'>{latest_nifty:.2f} <span style='color:{nifty_color}'>({nifty_change:+.2f}%)</span></p>
        """, unsafe_allow_html=True)
    
    with col2:
        latest_vix = vix_df['Close'].iloc[-1]
        prev_vix = vix_df['Close'].iloc[-2]
        vix_change = (latest_vix / prev_vix - 1) * 100
        vix_color = "red" if vix_change >= 0 else "green"  # Inverse relationship with market
        st.markdown(f"""
        ### India VIX
        <p style='font-size:24px'>{latest_vix:.2f} <span style='color:{vix_color}'>({vix_change:+.2f}%)</span></p>
        """, unsafe_allow_html=True)
    
    with col3:
        # Get model prediction
        try:
            # Get the latest state
            latest_date = nifty_df.index[-1].date()
            today = pd.Timestamp.now(tz='Asia/Kolkata').date()
            
            # If using today's data for prediction
            if latest_date == today:
                st.markdown("### Today's Prediction")
            else:
                st.markdown(f"### Next Day Prediction ({latest_date})")
            
            # Get the latest prediction from the model
            if trader.env is None:
                # Initialize environment if it doesn't exist
                try:
                    trader.env = TradingEnvironment(
                        nifty_data=nifty_df, 
                        vix_data=vix_df,
                        features_extractor=trader.features_extractor,
                        window_size=10,
                        trade_time=st.session_state.trade_time,
                        lot_size=st.session_state.lot_size,
                        initial_capital=st.session_state.initial_capital,
                        test_mode=False
                    )
                except Exception as e:
                    st.error(f"Error initializing environment: {e}")
                    import traceback
                    traceback.print_exc()
                    return
                
                # Make sure the environment is properly initialized
                if not trader.env.features_list:
                    trader.env._prepare_data()
                    
                # Make sure environment is reset and features are processed
                try:
                    trader.env.reset()
                    if not trader.env.features_list:
                        st.warning("No features available. Processing data...")
                        trader.env._prepare_data()
                        
                    if not trader.env.features_list:
                        st.error("Failed to extract features from data.")
                        return
                        
                    state = trader.env.get_state(len(trader.env.features_list) - 1)
                    if state is None:
                        st.error("Failed to get state from environment.")
                        return
                        
                    prediction = trader.predict(state, use_mcts=True)
                    
                    # Convert action to text
                    action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
                    action_text = action_map.get(prediction['action'], "UNKNOWN")
                    
                    # Set color based on action
                    action_color = "green" if action_text == "BUY" else "red" if action_text == "SELL" else "gray"
                    
                    # Display prediction
                    st.markdown(f"""
                    <p style='font-size:24px; color:{action_color};'>{action_text}</p>
                    <p>Confidence: {prediction['confidence']*100:.1f}%</p>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error getting prediction: {e}")
                    import traceback
                    traceback.print_exc()
                    return
            
            # Simulate the action to get entry, stop loss, and take profit levels
            if action_text != "HOLD":
                # Run a test step to get the levels
                test_env = TradingEnvironment(
                    nifty_data=nifty_df, 
                    vix_data=vix_df,
                    features_extractor=trader.features_extractor,
                    trade_time=st.session_state.trade_time,
                    lot_size=st.session_state.lot_size,
                    initial_capital=st.session_state.initial_capital,
                    test_mode=False
                )
                test_env.reset()
                # Move to last state
                test_env.current_idx = len(test_env.features_list) - 1
                # Take action
                _, _, _, info = test_env.step(prediction['action'])
                
                # Extract trade info
                if 'entry_price' in info:
                    st.markdown(f"**Entry Price:** {info['entry_price']:.2f}")
                if 'stop_loss' in info:
                    sl_pct = abs(info['stop_loss'] / info['entry_price'] - 1) * 100
                    st.markdown(f"**Stop Loss:** {info['stop_loss']:.2f} ({sl_pct:.1f}%)")
                if 'take_profit' in info:
                    tp_pct = abs(info['take_profit'] / info['entry_price'] - 1) * 100
                    st.markdown(f"**Take Profit:** {info['take_profit']:.2f} ({tp_pct:.1f}%)")
                
                # Show lot size
                st.markdown(f"**Lot Size:** {test_env.lot_size} shares")
        except Exception as e:
            st.error(f"Error getting prediction: {e}")
            import traceback
            traceback.print_exc()
    
    # Create a section for market data visualization
    st.markdown("## Market Data")
    
    # Checkbox to toggle between regular and candlestick chart
    chart_type = st.radio("Chart Type", ["Line", "Candlestick", "OHLC"], horizontal=True)
    
    # Date range slider
    end_date = nifty_df.index.max()
    start_date = end_date - pd.Timedelta(days=90)
    
    # Convert to datetime for the slider
    if isinstance(end_date, pd.Timestamp):
        end_date_dt = end_date.to_pydatetime()
        start_date_dt = start_date.to_pydatetime()
    else:
        end_date_dt = end_date
        start_date_dt = start_date
        
    date_range = st.slider(
        "Select Date Range",
        min_value=nifty_df.index.min().to_pydatetime(),
        max_value=nifty_df.index.max().to_pydatetime(),
        value=(start_date_dt, end_date_dt),
    )
    
    # Filter data based on date range
    mask = (nifty_df.index >= date_range[0]) & (nifty_df.index <= date_range[1])
    
    filtered_nifty = nifty_df[mask]
    filtered_vix = vix_df[mask]
    
    # Tab selection for charts
    tab1, tab2, tab3 = st.tabs(["NIFTY", "India VIX", "Comparison"])
    
    with tab1:
        fig = go.Figure()
        
        if chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=filtered_nifty.index,
                y=filtered_nifty['Close'],
                mode='lines',
                name='NIFTY Close'
            ))
        elif chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=filtered_nifty.index,
                open=filtered_nifty['Open'],
                high=filtered_nifty['High'],
                low=filtered_nifty['Low'],
                close=filtered_nifty['Close'],
                name='NIFTY'
            ))
        elif chart_type == "OHLC":
            fig.add_trace(go.Ohlc(
                x=filtered_nifty.index,
                open=filtered_nifty['Open'],
                high=filtered_nifty['High'],
                low=filtered_nifty['Low'],
                close=filtered_nifty['Close'],
                name='NIFTY'
            ))
            
        fig.update_layout(
            title='NIFTY Price Action',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price (₹)'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_vix.index,
            y=filtered_vix['Close'],
                mode='lines',
            name='India VIX',
                line=dict(color='orange', width=2)
        ))
        fig.update_layout(
            title='India VIX',
            xaxis=dict(title='Date'),
            yaxis=dict(title='VIX'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        
        # Normalize for comparison
        if len(filtered_nifty) > 0 and len(filtered_vix) > 0:
            nifty_norm = filtered_nifty['Close'] / filtered_nifty['Close'].iloc[0] * 100
            vix_norm = filtered_vix['Close'] / filtered_vix['Close'].iloc[0] * 100
            
            fig.add_trace(go.Scatter(
                x=filtered_nifty.index,
                y=nifty_norm,
                mode='lines',
                name='NIFTY',
                line=dict(color='blue', width=2)
            ))
        
            fig.add_trace(go.Scatter(
                x=filtered_vix.index,
                y=vix_norm,
                mode='lines',
                name='India VIX',
                line=dict(color='orange', width=2)
            ))
        
            fig.update_layout(
                title='NIFTY vs India VIX (Normalized)',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Normalized Value (%)'),
            height=500
        )
            st.plotly_chart(fig, use_container_width=True)
    
    # Trading History Section
    st.markdown("## Trading History")
    
    # Check if backtest data is available
    if hasattr(trader, 'env') and hasattr(trader.env, 'trade_history') and trader.env.trade_history:
        trade_history = trader.env.trade_history
        
        # Convert trade history to DataFrame
        if isinstance(trade_history, list) and len(trade_history) > 0:
            trade_df = pd.DataFrame(trade_history)
            
            # Calculate performance metrics
            total_trades = len(trade_df)
            if 'return' in trade_df.columns:
                profitable_trades = len(trade_df[trade_df['return'] > 0])
                win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
                total_return = trade_df['return'].sum()
                
                # Calculate drawdown if capital column exists
                if 'capital' in trade_df.columns:
                    capital_series = trade_df['capital']
                    cummax = capital_series.cummax()
                    drawdown = (capital_series - cummax) / cummax * 100
                    max_drawdown = drawdown.min()
                else:
                    max_drawdown = 0
                
                # Display performance metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", f"{total_trades}")
                with col2:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with col3:
                    st.metric("Total Return", f"{total_return:.2f}%")
                with col4:
                    st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                
                # Show capital growth chart
                if 'capital' in trade_df.columns:
                    st.subheader("Capital Growth")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trade_df['date'],
                        y=trade_df['capital'],
                        mode='lines+markers',
                        name='Capital',
                        line=dict(color='green', width=2)
                    ))
                    fig.update_layout(
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='Capital (₹)'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show trades table
            st.subheader("Trades")
            st.dataframe(trade_df, use_container_width=True)
        else:
            st.warning("No trade history available. Try running a backtest first.")
    else:
        st.info("No trade history available. Try running a backtest first.")

def data_management_page():
    """Data Management page for handling data files"""
    st.title("Data Management")
    
    # Ensure data directory exists
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Create tabs for different data operations
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Update Data", "Consolidate Files", "Export/Import"])
    
    # Get data summary
    data_summary = get_data_summary(data_dir=data_dir)
    st.session_state.data_summary = data_summary
    
    with tab1:
        st.subheader("Current Data Status")
        
        # Display consolidated data information
        if data_summary['has_consolidated_files']:
            st.success("✅ Consolidated data files are available")
            
            # Create metrics for NIFTY and VIX
            col1, col2 = st.columns(2)
            with col1:
                st.metric("NIFTY Records", f"{data_summary['nifty_records']:,}")
                if data_summary['nifty_date_range']:
                    st.write(f"Date Range: {data_summary['nifty_date_range'][0].date()} to {data_summary['nifty_date_range'][1].date()}")
            
            with col2:
                st.metric("VIX Records", f"{data_summary['vix_records']:,}")
                if data_summary['vix_date_range']:
                    st.write(f"Date Range: {data_summary['vix_date_range'][0].date()} to {data_summary['vix_date_range'][1].date()}")
                    
            # Show data availability timeline
            if data_summary['nifty_date_range'] and data_summary['vix_date_range']:
                st.subheader("Data Availability")
                
                # Calculate date range for visualization
                min_date = min(data_summary['nifty_date_range'][0], data_summary['vix_date_range'][0])
                max_date = max(data_summary['nifty_date_range'][1], data_summary['vix_date_range'][1])
                
                # Create date range dataframe
                date_range = pd.date_range(start=min_date, end=max_date)
                dates_df = pd.DataFrame({
                    'date': date_range,
                    'has_data': 1  # Just a placeholder value
                })
                
                # Create timeline chart
                fig = px.timeline(
                    dates_df,
                    x_start='date',
                    x_end='date',
                    y='has_data',
                    title="Data Coverage Timeline"
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_title="Date",
                    yaxis_title="",
                    yaxis_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No consolidated data files found. Use the 'Consolidate Files' tab to create them.")
        
        # Show other data files
        st.subheader("Other Data Files")
        if data_summary['other_files']:
            file_data = data_summary['other_files']
            # Convert to dataframe for better display
            files_df = pd.DataFrame(file_data)
            files_df['size'] = files_df['size'].apply(lambda x: f"{x / 1024:.1f} KB")
            files_df['modified'] = files_df['modified'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
            
            st.dataframe(files_df)
        else:
            st.info("No additional data files found.")
    
    with tab2:
        st.subheader("Update Data")
        
        # Option to update data
        days = st.slider("Days to fetch", min_value=1, max_value=60, value=30, 
                         help="Number of days to fetch from Yahoo Finance")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Fetch & Update Data"):
                with st.spinner("Fetching and merging data..."):
                    try:
                        nifty_df, vix_df = update_data(days=days, data_dir=data_dir)
                        
                        # Update session state with new data
                        st.session_state.nifty_data = nifty_df
                        st.session_state.vix_data = vix_df
                        
                        # Update data summary
                        st.session_state.data_summary = get_data_summary(data_dir=data_dir)
                        
                        st.success(f"Successfully updated data with {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records")
                    except Exception as e:
                        st.error(f"Error updating data: {e}")
        
        with col2:
            st.info("""
            This will fetch the most recent data and merge it with your existing data.
            The process will:
            1. Load your existing consolidated data
            2. Fetch new data from Yahoo Finance
            3. Merge them together without duplicates
            4. Save the consolidated data file
            """)
        
        # Show last update time
        if data_summary['has_consolidated_files']:
            last_updated = None
            for file in data_summary['other_files']:
                if file['name'] == 'nifty_data_consolidated.csv':
                    last_updated = file['modified']
                    break
            
            if last_updated:
                st.write(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with tab3:
        st.subheader("Consolidate Data Files")
        
        st.write("""
        If you have multiple data files (from different fetches), you can consolidate them into a single file.
        This will merge all data, remove duplicates, and create a clean consolidated file.
        """)
        
        if st.button("Consolidate All Files"):
            with st.spinner("Consolidating files..."):
                try:
                    nifty_df, vix_df = consolidate_csv_files(data_dir=data_dir)
                    
                    # Update session state
                    st.session_state.nifty_data = nifty_df
                    st.session_state.vix_data = vix_df
                    
                    # Update data summary
                    st.session_state.data_summary = get_data_summary(data_dir=data_dir)
                    
                    st.success(f"Successfully consolidated data with {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records")
                except Exception as e:
                    st.error(f"Error consolidating files: {e}")
        
        # Option to delete old files after consolidation
        if data_summary['has_consolidated_files'] and data_summary['other_files']:
            if st.button("Delete Old Files (Keep Consolidated Only)"):
                with st.spinner("Deleting old files..."):
                    try:
                        # Only delete non-consolidated files
                        for file_info in data_summary['other_files']:
                            if file_info['name'] not in ['nifty_data_consolidated.csv', 'vix_data_consolidated.csv']:
                                file_path = os.path.join(data_dir, file_info['name'])
                                os.remove(file_path)
                        
                        # Update data summary
                        st.session_state.data_summary = get_data_summary(data_dir=data_dir)
                        
                        st.success("Successfully deleted old files")
                    except Exception as e:
                        st.error(f"Error deleting files: {e}")
    
    with tab4:
        st.subheader("Export/Import Data")
        
        # Export data
        st.write("#### Export Data")
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
        
        if st.button("Export Consolidated Data"):
            with st.spinner("Preparing data for export..."):
                try:
                    # Load consolidated data
                    nifty_consolidated_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
                    vix_consolidated_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
                    
                    if os.path.exists(nifty_consolidated_path) and os.path.exists(vix_consolidated_path):
                        nifty_df, vix_df = load_from_csv(nifty_consolidated_path, vix_consolidated_path)
                        
                        # Create a zip file with both datasets
                        import zipfile
                        import io
                        
                        # Create in-memory zip file
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
                            if export_format == "CSV":
                                # Write CSVs to zip
                                nifty_buffer = io.StringIO()
                                vix_buffer = io.StringIO()
                                
                                nifty_df.to_csv(nifty_buffer)
                                vix_df.to_csv(vix_buffer)
                                
                                zip_file.writestr("nifty_data.csv", nifty_buffer.getvalue())
                                zip_file.writestr("vix_data.csv", vix_buffer.getvalue())
                            
                            elif export_format == "Excel":
                                # Write Excel to zip
                                excel_buffer = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer) as writer:
                                    nifty_df.to_excel(writer, sheet_name="NIFTY")
                                    vix_df.to_excel(writer, sheet_name="VIX")
                                
                                zip_file.writestr("nifty_vix_data.xlsx", excel_buffer.getvalue())
                            
                            elif export_format == "JSON":
                                # Write JSONs to zip
                                nifty_json = nifty_df.reset_index().to_json(orient="records", date_format="iso")
                                vix_json = vix_df.reset_index().to_json(orient="records", date_format="iso")
                                
                                zip_file.writestr("nifty_data.json", nifty_json)
                                zip_file.writestr("vix_data.json", vix_json)
                        
                        # Download button for the zip file
                        zip_buffer.seek(0)
                        timestamp = datetime.now().strftime("%Y%m%d")
                        
                        # Use st.download_button to enable downloading the file
                        st.download_button(
                            label="Download Data",
                            data=zip_buffer,
                            file_name=f"nifty_vix_data_{timestamp}.zip",
                            mime="application/zip"
                        )
                    else:
                        st.warning("No consolidated data files found. Please consolidate your data first.")
                except Exception as e:
                    st.error(f"Error exporting data: {e}")
        
        # Import data
        st.write("#### Import Data")
        st.write("Upload CSV files to import data.")
        
        col1, col2 = st.columns(2)
        with col1:
            nifty_file = st.file_uploader("Upload NIFTY Data", type=["csv"])
        with col2:
            vix_file = st.file_uploader("Upload VIX Data", type=["csv"])
        
        if nifty_file is not None and vix_file is not None:
            if st.button("Import Data"):
                with st.spinner("Processing uploaded files..."):
                    try:
                        # Read uploaded files
                        nifty_df = pd.read_csv(nifty_file, index_col=0, parse_dates=True)
                        vix_df = pd.read_csv(vix_file, index_col=0, parse_dates=True)
                        
                        # Merge with existing data
                        nifty_consolidated_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
                        vix_consolidated_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
                        
                        if os.path.exists(nifty_consolidated_path) and os.path.exists(vix_consolidated_path):
                            existing_nifty, existing_vix = load_from_csv(nifty_consolidated_path, vix_consolidated_path)
                            
                            # Merge
                            nifty_df = merge_dataframes(existing_nifty, nifty_df)
                            vix_df = merge_dataframes(existing_vix, vix_df)
                        
                        # Save merged data
                        save_to_csv(nifty_df, vix_df, data_dir=data_dir)
                        
                        # Update session state
                        st.session_state.nifty_data = nifty_df
                        st.session_state.vix_data = vix_df
                        
                        # Update data summary
                        st.session_state.data_summary = get_data_summary(data_dir=data_dir)
                        
                        st.success(f"Successfully imported data with {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records")
                    except Exception as e:
                        st.error(f"Error importing data: {e}")

def training_page():
    """Training page for model training"""
    st.title("Model Training")
    
    # Add info box with guidance
    with st.expander("Training Information", expanded=True):
        st.info(
            "This page allows you to train the AlphaZero model for trading. "
            "Training involves two steps:\n\n"
            "1. **Self-play**: The model plays against itself to generate training examples\n"
            "2. **Neural Network Training**: The collected examples are used to train the neural network\n\n"
            "Training can take some time, especially with more episodes and batches. "
            "If you encounter errors during training, you can try running the standalone training script "
            "by executing `python train_standalone.py` in your terminal."
        )
    
    # Ensure data is loaded
    if 'nifty_data' not in st.session_state or st.session_state.nifty_data is None or \
       'vix_data' not in st.session_state or st.session_state.vix_data is None:
        try:
            with st.spinner("Loading data..."):
                trader, nifty_df, vix_df = load_data_and_model()
                st.session_state.nifty_data = nifty_df
                st.session_state.vix_data = vix_df
                if trader is not None:
                    st.session_state.trader = trader
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please make sure you have data available. Go to Data Management to fetch data.")
            
            # Button to go to data management
            if st.button("Go to Data Management"):
                st.session_state.current_page = "Data Management"
                st.rerun()
            return
    else:
        nifty_df = st.session_state.nifty_data
        vix_df = st.session_state.vix_data
    
    # Training parameters
    st.subheader("Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        episodes = st.slider("Self-play episodes", min_value=1, max_value=50, value=10)
        simulation_steps = st.slider("MCTS simulation steps", min_value=10, max_value=200, value=50)
        epochs = st.slider("Training epochs", min_value=5, max_value=100, value=20)
        
    with col2:
        batch_size = st.slider("Batch size", min_value=16, max_value=256, value=64)
        exploration_rate = st.slider("Exploration rate", min_value=0.0, max_value=1.0, value=0.25)
        learning_rate = st.slider("Learning rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
    
    # Advanced parameters
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            lot_size = st.number_input("Lot Size", min_value=1, max_value=1000, 
                                      value=int(st.session_state.get('lot_size', 50)))
            st.session_state.lot_size = lot_size
            
            initial_capital = st.number_input("Initial Capital", min_value=10000, max_value=10000000, 
                                             value=int(st.session_state.get('initial_capital', 100000)), 
                                             step=10000)
            st.session_state.initial_capital = initial_capital
            
        with col2:
            discount_factor = st.slider("Discount factor", min_value=0.9, max_value=0.999, value=0.99)
            dirichlet_alpha = st.slider("Dirichlet noise alpha", min_value=0.03, max_value=1.0, value=0.3)
    
    # Training button
    if st.button("Start Training"):
        try:
            # Import train_model function from main.py
            from main import train_model
            
            with st.spinner("Training in progress... This may take several minutes."):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create a container for the training metrics
                metrics_container = st.container()
                
                # Status update function for progress
                def update_status(message, progress=None):
                    status_text.text(message)
                    if progress is not None:
                        progress_bar.progress(progress)
                
                # Create a callback to update the UI
                def training_callback(info):
                    if 'progress' in info:
                        progress_bar.progress(info['progress'])
                        
                    if 'status' in info:
                        status_text.text(info['status'])
                    elif 'batch' in info:
                        status_text.text(f"Training batch {info['batch']}/{info['total_batches']}")
                    elif 'episode' in info:
                        status_text.text(f"Self-play episode {info['episode']}/{info['total_episodes']}")
                        
                    if 'metrics' in info:
                        with metrics_container:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Policy Loss", f"{info['metrics'].get('policy_loss', 0):.4f}")
                            with col2:
                                st.metric("Value Loss", f"{info['metrics'].get('value_loss', 0):.4f}")
                            with col3:
                                st.metric("Total Loss", f"{info['metrics'].get('total_loss', 0):.4f}")
                
                # Try to perform the training
                try:
                    # Run training with callback
                    trader = train_model(
                        nifty_df=nifty_df,
                        vix_df=vix_df,
                        episodes=episodes, 
                        batches=epochs
                    )
                    
                    if trader is not None:
                        st.session_state.trader = trader
                        progress_bar.progress(1.0)
                        status_text.text("Training completed successfully!")
                        st.success("Model trained and saved successfully!")
                    else:
                        st.error("Training failed. Check the logs for details.")
                        
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    st.info("You can try running the standalone training script with: `python train_standalone.py`")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
        
        except Exception as e:
            st.error(f"Error setting up training: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")
                
    # Training history
    st.subheader("Training History")
    if 'trader' in st.session_state and st.session_state.trader is not None:
        try:
            plot_training_history()
        except Exception as e:
            st.warning(f"Could not display training history: {e}")
    else:
        st.info("No training history available yet. Train a model to see metrics.")

def backtesting_page():
    """Backtesting page for model evaluation"""
    st.title("Model Backtesting")
    
    # Check if we have backtest results from a previous run
    if 'backtest_results' in st.session_state and st.session_state.backtest_results is not None:
        st.success("Backtest results loaded successfully!")
        results = st.session_state.backtest_results
        
        # Display backtest results
        st.subheader("Backtest Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Trades", results.get('total_trades', 0))
            st.metric("Win Rate", f"{results.get('win_rate', 0):.2%}")
            st.metric("Total Return", f"{results.get('total_return', 0):.2%}")
        
        with col2:
            st.metric("Profitable Trades", results.get('profitable_trades', 0))
            st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
            
        # Plot trade history if available
        if 'trade_history' in results and results['trade_history']:
            st.subheader("Trade History")
            plot_backtest_results()
            
        # Clear results after displaying
        if st.button("Clear Results"):
            st.session_state.backtest_results = None
            st.rerun()
            
        return
    
    # Ensure data and model are loaded
    if 'trader' not in st.session_state or st.session_state.trader is None or \
       'nifty_data' not in st.session_state or st.session_state.nifty_data is None:
        try:
            with st.spinner("Loading data and model..."):
                trader, nifty_df, vix_df = load_data_and_model()
                st.session_state.trader = trader
                st.session_state.nifty_data = nifty_df
                st.session_state.vix_data = vix_df
        except Exception as e:
            st.error(f"Error loading data or model: {e}")
            st.info("Please make sure you have data available and a trained model.")
            return
    else:
        trader = st.session_state.trader
        nifty_df = st.session_state.nifty_data
        vix_df = st.session_state.vix_data
    
    # Backtest parameters
    st.subheader("Backtest Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date range for backtest
        backtest_range = st.slider(
            "Backtest Date Range",
            min_value=nifty_df.index.min().to_pydatetime(),
            max_value=nifty_df.index.max().to_pydatetime(),
            value=(
                (nifty_df.index.max() - pd.Timedelta(days=90)).to_pydatetime(),
                nifty_df.index.max().to_pydatetime()
            )
        )
        start_date, end_date = backtest_range
        
        # Use MCTS for decisions
        use_mcts = st.checkbox("Use MCTS", value=False, 
                              help="Use Monte Carlo Tree Search for better predictions (slower)")
    
    with col2:
        # Trading parameters
        lot_size = st.number_input("Lot Size", min_value=1, max_value=1000, 
                                  value=int(st.session_state.get('lot_size', 50)))
        st.session_state.lot_size = lot_size
        
        initial_capital = st.number_input("Initial Capital", min_value=10000, max_value=10000000, 
                                         value=int(st.session_state.get('initial_capital', 100000)), 
                                         step=10000)
        st.session_state.initial_capital = initial_capital
    
    # Start backtest button
    if st.button("Start Backtest"):
        with st.spinner("Backtesting in progress..."):
            try:
                # Make sure the environment is properly initialized
                trader.env = TradingEnvironment(
                    nifty_data=nifty_df,
                    vix_data=vix_df,
                    features_extractor=trader.features_extractor,
                    trade_time=st.session_state.trade_time,
                    lot_size=st.session_state.lot_size,
                    initial_capital=st.session_state.initial_capital
                )
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Convert start_date and end_date to timezone-aware datetime objects
                try:
                    # Convert to timezone-aware timestamps with Asia/Kolkata timezone
                    start_date_obj = pd.Timestamp(start_date).tz_localize('Asia/Kolkata')
                    end_date_obj = pd.Timestamp(end_date).tz_localize('Asia/Kolkata')
                except TypeError:
                    # If already timezone-aware, just use as is
                    start_date_obj = pd.Timestamp(start_date)
                    end_date_obj = pd.Timestamp(end_date)
                
                # Run backtest
                status_text.text(f"Running backtest from {start_date_obj.date()} to {end_date_obj.date()}...")
                results = trader.backtest(start_date=start_date_obj, end_date=end_date_obj)
                progress_bar.progress(1.0)
                
                # Display results
                if results:
                    # Show backtest summary
                    st.subheader("Backtest Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Trades", results.get('total_trades', 0))
                        st.metric("Win Rate", f"{results.get('win_rate', 0):.2%}")
                        st.metric("Total Return", f"{results.get('total_return', 0):.2%}")
                    
                    with col2:
                        st.metric("Profitable Trades", results.get('profitable_trades', 0))
                        st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
                        st.metric("Final Capital", f"₹{results.get('final_capital', 0):,.2f}")
                    
                    # Store the results in session state
                    st.session_state.backtest_results = results
                    
                    # Display trade history if available
                    if 'trade_history' in results and results['trade_history']:
                        st.session_state.backtest_complete = True
                        plot_backtest_results()
                    else:
                        st.warning("No trades were executed during the backtest period.")
                else:
                    st.error("Backtest failed to produce results.")
                
            except Exception as e:
                st.error(f"Error during backtesting: {e}")
                import traceback
                traceback.print_exc()

def prediction_page():
    """Real-time prediction page"""
    st.title("Trade Prediction")
    
    # Ensure data and model are loaded
    if 'trader' not in st.session_state or st.session_state.trader is None or \
       'nifty_data' not in st.session_state or st.session_state.nifty_data is None:
        try:
            with st.spinner("Loading data and model..."):
                trader, nifty_df, vix_df = load_data_and_model()
                st.session_state.trader = trader
                st.session_state.nifty_data = nifty_df
                st.session_state.vix_data = vix_df
        except Exception as e:
            st.error(f"Error loading data or model: {e}")
            st.info("Before making predictions, you need to train a model first.")
            return
    else:
        trader = st.session_state.trader
        nifty_df = st.session_state.nifty_data
        vix_df = st.session_state.vix_data
        
    # Check if model was loaded successfully
    if not hasattr(trader, 'model') or trader.model is None:
        st.warning("No trained model found. Please train a model first.")
        if st.button("Go to Training"):
            st.session_state.active_page = "Training"
            st.rerun()
        return
    
    # Current prediction
    st.subheader("Current Prediction")
    
    # Get latest state
    try:
        latest_idx = len(trader.env.features_list) - 1
        state = trader.env.get_state(latest_idx)
        
        # Make prediction with and without MCTS
        direct_prediction = trader.predict(state, use_mcts=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display neural network prediction
            action = direct_prediction['action']
            confidence = direct_prediction['confidence'] * 100
            
            action_color = "green" if action == "buy" else ("red" if action == "sell" else "gray")
            st.markdown(f"""
            ### Neural Network Prediction
            <p style='font-size:42px; color:{action_color}'>{action.upper()}</p>
            <p style='font-size:24px'>Confidence: {confidence:.1f}%</p>
            """, unsafe_allow_html=True)
            
            # Show policy distribution
            policy = direct_prediction['policy']
            policy_df = pd.DataFrame({
                'Action': ['Buy', 'Sell', 'Hold'],
                'Probability': policy
            })
            
            fig = px.bar(policy_df, x='Action', y='Probability', color='Action',
                       color_discrete_map={'Buy': 'green', 'Sell': 'red', 'Hold': 'gray'})
            fig.update_layout(title='Policy Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Option to run MCTS prediction
            run_mcts = st.button("Run MCTS Prediction (Slower but more accurate)")
            
            if run_mcts:
                with st.spinner("Running MCTS search..."):
                    mcts_prediction = trader.predict(state, use_mcts=True)
                    
                    action = mcts_prediction['action']
                    confidence = mcts_prediction['confidence'] * 100
                    
                    action_color = "green" if action == "buy" else ("red" if action == "sell" else "gray")
                    st.markdown(f"""
                    ### MCTS Prediction
                    <p style='font-size:42px; color:{action_color}'>{action.upper()}</p>
                    <p style='font-size:24px'>Confidence: {confidence:.1f}%</p>
                    """, unsafe_allow_html=True)
                    
                    # Show policy distribution
                    policy = mcts_prediction['policy']
                    policy_df = pd.DataFrame({
                        'Action': ['Buy', 'Sell', 'Hold'],
                        'Probability': policy
                    })
                    
                    fig = px.bar(policy_df, x='Action', y='Probability', color='Action',
                               color_discrete_map={'Buy': 'green', 'Sell': 'red', 'Hold': 'gray'})
                    fig.update_layout(title='MCTS Policy Distribution')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display market context
        st.subheader("Market Context")
        
        # Get the window for the current state
        window_idx = latest_idx + trader.env.window_size
        market_window = trader.env.daily_data.iloc[window_idx-trader.env.window_size:window_idx]
        
        # Create market summary
        market_summary = create_market_summary(market_window)
        st.markdown(f"""
        ```
        {market_summary}
        ```
        """)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.info("Make sure the model is trained and data is loaded correctly.")

def settings_page():
    """Settings page for the application"""
    st.title("Settings")
    
    # Model settings
    st.subheader("Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trade_time = st.text_input("Trading Time (HH:MM)", value=DEFAULT_TRADE_TIME)
    
    with col2:
        window_size = st.number_input("Window Size (Days)", min_value=5, max_value=50, value=20)
    
    # Data settings moved to Data Management
    st.subheader("Data Management")
    st.info("Data management has been moved to a dedicated page. Click the button below to go there.")
    
    if st.button("Go to Data Management"):
        st.session_state.active_page = "Data Management"
        st.rerun()
    
    # Apply settings
    if st.button("Apply Settings"):
        # Store settings
        st.session_state.trade_time = trade_time
        st.session_state.window_size = window_size
        
        # Reload model with new settings
        if st.session_state.trader is not None:
            with st.spinner("Applying settings and reloading model..."):
                # Get existing data
                nifty_df = st.session_state.nifty_data
                vix_df = st.session_state.vix_data
                
                if nifty_df is not None and vix_df is not None:
                    # Reinitialize trader with new settings
                    trader = AlphaZeroTrader(
                        nifty_data=nifty_df,
                        vix_data=vix_df,
                        features_extractor=extract_features,
                        input_shape=INPUT_SHAPE,
                        trade_time=trade_time
                    )
                    
                    # Try to load existing model
                    trader.load_model()
                    
                    st.session_state.trader = trader
                    
                    st.success("Settings applied successfully")
                else:
                    st.warning("No data available. Please go to Data Management to load or fetch data.")
        else:
            st.warning("Model not initialized. Please go to Data Management to load or fetch data first.")
    
    # Advanced settings
    st.subheader("Advanced Settings")
    
    # Reset application
    if st.button("Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main function
def main(backtest_results=None):
    """Main entry point for the Streamlit app"""
    # Parse command line arguments for results file
    import sys
    import json
    import os
    
    # Check if results file was passed as command line argument
    results_file = None
    if len(sys.argv) > 2 and sys.argv[1] == "--results_file":
        results_file = sys.argv[2]
        
    # Load results from file if provided
    if results_file and os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                backtest_results = json.load(f)
            # Remove the temp file after loading
            os.remove(results_file)
        except Exception as e:
            st.error(f"Error loading backtest results: {e}")
    
    # Initialize session state for navigation and data
    if 'trader' not in st.session_state:
        st.session_state.trader = None
    
    if 'nifty_data' not in st.session_state:
        st.session_state.nifty_data = None
        
    if 'vix_data' not in st.session_state:
        st.session_state.vix_data = None
    
    if 'lot_size' not in st.session_state:
        st.session_state.lot_size = 50
        
    if 'initial_capital' not in st.session_state:
        st.session_state.initial_capital = 100000
        
    if 'trade_time' not in st.session_state:
        st.session_state.trade_time = '9:15'
        
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Store backtest results if provided
    if backtest_results is not None:
        st.session_state.backtest_results = backtest_results
        st.session_state.current_page = "Backtesting"
    
    # Display the sidebar and get selected page
    selected_page = sidebar()
    
    # If sidebar selection is different from current page, update
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()
    
    # Render the selected page
    if st.session_state.current_page == "Dashboard":
        dashboard_page()
    elif st.session_state.current_page == "Training":
        training_page()
    elif st.session_state.current_page == "Backtesting":
        backtesting_page()
    elif st.session_state.current_page == "Prediction":
        prediction_page()
    elif st.session_state.current_page == "Data Management":
        data_management_page()
    elif st.session_state.current_page == "Settings":
        settings_page()
    else:
        st.error(f"Unknown page: {st.session_state.current_page}")

if __name__ == "__main__":
    main() 