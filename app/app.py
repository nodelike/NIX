import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import json
from PIL import Image
from io import BytesIO
import base64

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import get_latest_data
from src.data_processing.features import extract_features, create_market_summary
from src.alphazero.trader import AlphaZeroTrader

# Constants
INPUT_SHAPE = (1, 10)
DEFAULT_TRADE_TIME = "09:05"
PAGES = ["Dashboard", "Training", "Backtesting", "Prediction", "Settings"]

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

# Define callback functions
def training_callback(info):
    """Callback function for training updates"""
    if 'episode_complete' in info and info['episode_complete']:
        # New episode completed
        if 'rewards' not in st.session_state.training_history:
            st.session_state.training_history['rewards'] = []
        st.session_state.training_history['rewards'].append(info['total_reward'])
        
        if 'actions' not in st.session_state.training_history:
            st.session_state.training_history['actions'] = []
        st.session_state.training_history['actions'].append(info['actions_count'])
    
    # Update progress information
    st.session_state.training_progress = info
    
    # Force streamlit to update
    st.experimental_rerun()

def backtest_callback(info):
    """Callback function for backtest updates"""
    if 'complete' in info and info['complete']:
        # Backtest completed
        st.session_state.backtest_results = {
            'metrics': info['metrics'],
            'results_df': info['results_df']
        }
    else:
        # Update progress
        st.session_state.backtest_progress = info
    
    # Force streamlit to update
    st.experimental_rerun()

# Helper functions
def load_data_and_model():
    """Load data and initialize model"""
    with st.spinner("Loading data and initializing model..."):
        # Get data
        nifty_df, vix_df = get_latest_data(load_from_disk=True, fetch_days=30)
        
        # Initialize trader
        trader = AlphaZeroTrader(
            nifty_data=nifty_df,
            vix_data=vix_df,
            features_extractor=extract_features,
            input_shape=INPUT_SHAPE,
            trade_time=DEFAULT_TRADE_TIME
        )
        
        # Try to load existing model
        trader.load_model()
        
        st.session_state.trader = trader
        st.session_state.nifty_data = nifty_df
        st.session_state.vix_data = vix_df
        
        return trader, nifty_df, vix_df

def plot_training_history():
    """Plot the training history"""
    if not st.session_state.training_history['rewards']:
        st.info("No training history available. Train the model first.")
        return
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot rewards
        rewards = st.session_state.training_history['rewards']
        fig_rewards = px.line(
            x=list(range(1, len(rewards) + 1)),
            y=rewards,
            title="Reward per Episode",
            labels={"x": "Episode", "y": "Reward"}
        )
        st.plotly_chart(fig_rewards, use_container_width=True)
        
    with col2:
        # Plot action distribution
        if st.session_state.training_history['actions']:
            actions = np.array(st.session_state.training_history['actions'])
            buy_counts = actions[:, 0]
            sell_counts = actions[:, 1]
            hold_counts = actions[:, 2]
            
            actions_df = pd.DataFrame({
                'Episode': list(range(1, len(actions) + 1)),
                'Buy': buy_counts,
                'Sell': sell_counts,
                'Hold': hold_counts
            })
            
            fig_actions = px.bar(
                actions_df, 
                x='Episode', 
                y=['Buy', 'Sell', 'Hold'],
                title="Actions per Episode",
                barmode='stack'
            )
            st.plotly_chart(fig_actions, use_container_width=True)
    
    # Plot losses if available
    if st.session_state.training_history.get('policy_loss') and st.session_state.training_history.get('value_loss'):
        # Create a DataFrame for losses
        losses_df = pd.DataFrame({
            'Batch': list(range(1, len(st.session_state.training_history['policy_loss']) + 1)),
            'Policy Loss': st.session_state.training_history['policy_loss'],
            'Value Loss': st.session_state.training_history['value_loss'],
            'Total Loss': st.session_state.training_history['total_loss']
        })
        
        # Plot all losses
        fig_losses = px.line(
            losses_df, 
            x='Batch', 
            y=['Policy Loss', 'Value Loss', 'Total Loss'],
            title="Training Losses",
            labels={"x": "Batch", "value": "Loss"}
        )
        st.plotly_chart(fig_losses, use_container_width=True)

def plot_backtest_results():
    """Plot the backtest results"""
    if st.session_state.backtest_results is None:
        st.info("No backtest results available. Run a backtest first.")
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
    
    # Mark Buy and Sell actions
    buy_points = results_df[results_df['action'] == 'buy']
    sell_points = results_df[results_df['action'] == 'sell']
    
    fig.add_trace(
        go.Scatter(
            x=buy_points['date'],
            y=buy_points['capital'],
            mode='markers',
            name='Buy',
            marker=dict(color='green', size=10, symbol='triangle-up')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=sell_points['date'],
            y=sell_points['capital'],
            mode='markers',
            name='Sell',
            marker=dict(color='red', size=10, symbol='triangle-down')
        )
    )
    
    # Add NIFTY price on secondary axis
    fig.add_trace(
        go.Scatter(
            x=results_df['date'],
            y=results_df['price'],
            mode='lines',
            name='NIFTY',
            line=dict(color='gray', width=1, dash='dot'),
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
    
    # Display trade distribution
    action_counts = results_df['action'].value_counts()
    fig_actions = px.pie(
        values=action_counts.values,
        names=action_counts.index,
        title="Trading Action Distribution"
    )
    st.plotly_chart(fig_actions, use_container_width=True)
    
    # Display results table
    st.dataframe(results_df)

# Sidebar for navigation
def sidebar():
    """Create sidebar for navigation"""
    st.sidebar.title("AlphaZero Trader")
    st.sidebar.caption("Powered by AlphaZero-inspired RL")
    
    # Navigation
    st.sidebar.header("Navigation")
    selected_page = st.sidebar.radio("Go to", PAGES)
    if selected_page != st.session_state.active_page:
        st.session_state.active_page = selected_page
        st.experimental_rerun()
    
    # Load or reload data
    if st.sidebar.button("Reload Data"):
        st.session_state.trader = None
        st.session_state.nifty_data = None
        st.session_state.vix_data = None
        load_data_and_model()
        st.experimental_rerun()
    
    # Custom styles
    st.sidebar.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Status indicators
    if st.session_state.trader is not None:
        st.sidebar.success("✅ Model loaded")
    else:
        st.sidebar.error("❌ Model not loaded")
        
    if st.session_state.nifty_data is not None and st.session_state.vix_data is not None:
        data_timestamp = st.session_state.nifty_data.index[-1].strftime("%Y-%m-%d %H:%M")
        st.sidebar.success(f"✅ Data loaded (last: {data_timestamp})")
    else:
        st.sidebar.error("❌ Data not loaded")
    
    # Show project info
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2023 AlphaZero Trader")

# Pages implementation
def dashboard_page():
    """Dashboard page with overview"""
    st.title("AlphaZero Trader Dashboard")
    
    # Ensure data and model are loaded
    if st.session_state.trader is None or st.session_state.nifty_data is None:
        trader, nifty_df, vix_df = load_data_and_model()
    else:
        trader = st.session_state.trader
        nifty_df = st.session_state.nifty_data
        vix_df = st.session_state.vix_data
    
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
        vix_color = "red" if vix_change >= 0 else "green"  # Higher VIX is typically "bad"
        st.markdown(f"""
        ### INDIA VIX
        <p style='font-size:24px'>{latest_vix:.2f} <span style='color:{vix_color}'>({vix_change:+.2f}%)</span></p>
        """, unsafe_allow_html=True)
    
    with col3:
        # Latest prediction
        try:
            last_window = trader.env.daily_data.iloc[-trader.env.window_size:]
            state = trader.env.get_state(len(trader.env.features_list) - 1)
            if state is not None:
                prediction = trader.predict(state, use_mcts=False)
                action = prediction['action']
                confidence = prediction['confidence'] * 100
                action_color = "green" if action == "buy" else ("red" if action == "sell" else "gray")
                st.markdown(f"""
                ### Today's Prediction
                <p style='font-size:24px; color:{action_color}'>{action.upper()} <span style='font-size:18px'>({confidence:.1f}%)</span></p>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            ### Today's Prediction
            <p style='font-size:24px; color:gray'>Not Available</p>
            """, unsafe_allow_html=True)
    
    # Market charts
    st.subheader("Market Overview")
    tab1, tab2, tab3 = st.tabs(["NIFTY", "INDIA VIX", "Comparison"])
    
    with tab1:
        # NIFTY chart
        fig_nifty = go.Figure()
        fig_nifty.add_trace(
            go.Candlestick(
                x=nifty_df.index[-30:],
                open=nifty_df['Open'][-30:],
                high=nifty_df['High'][-30:],
                low=nifty_df['Low'][-30:],
                close=nifty_df['Close'][-30:],
                name='NIFTY'
            )
        )
        fig_nifty.update_layout(
            title='NIFTY Price Action (Last 30 Days)',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price (₹)'),
            height=500
        )
        st.plotly_chart(fig_nifty, use_container_width=True)
    
    with tab2:
        # VIX chart
        fig_vix = go.Figure()
        fig_vix.add_trace(
            go.Scatter(
                x=vix_df.index[-30:],
                y=vix_df['Close'][-30:],
                mode='lines',
                name='INDIA VIX',
                line=dict(color='orange', width=2)
            )
        )
        fig_vix.update_layout(
            title='INDIA VIX (Last 30 Days)',
            xaxis=dict(title='Date'),
            yaxis=dict(title='VIX'),
            height=500
        )
        st.plotly_chart(fig_vix, use_container_width=True)
    
    with tab3:
        # Comparison chart
        fig_comp = go.Figure()
        
        # Normalize NIFTY for comparison
        nifty_norm = nifty_df['Close'][-30:] / nifty_df['Close'].iloc[-30] * 100
        vix_norm = vix_df['Close'][-30:] / vix_df['Close'].iloc[-30] * 100
        
        fig_comp.add_trace(
            go.Scatter(
                x=nifty_df.index[-30:],
                y=nifty_norm,
                mode='lines',
                name='NIFTY',
                line=dict(color='blue', width=2)
            )
        )
        
        fig_comp.add_trace(
            go.Scatter(
                x=vix_df.index[-30:],
                y=vix_norm,
                mode='lines',
                name='INDIA VIX',
                line=dict(color='orange', width=2)
            )
        )
        
        fig_comp.update_layout(
            title='NIFTY vs INDIA VIX (Normalized, Last 30 Days)',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Normalized Value (%)'),
            height=500
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    
    # Market summary
    st.subheader("Market Analysis")
    try:
        last_window = trader.env.daily_data.iloc[-trader.env.window_size:]
        market_summary = create_market_summary(last_window)
        st.markdown(f"""
        ```
        {market_summary}
        ```
        """)
    except:
        st.info("Market analysis not available. Make sure the model is trained.")

def training_page():
    """Training page for the model"""
    st.title("AlphaZero Model Training")
    
    # Ensure data and model are loaded
    if st.session_state.trader is None or st.session_state.nifty_data is None:
        trader, nifty_df, vix_df = load_data_and_model()
    else:
        trader = st.session_state.trader
        nifty_df = st.session_state.nifty_data
        vix_df = st.session_state.vix_data
    
    # Training parameters
    st.subheader("Training Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        self_play_episodes = st.number_input("Self-Play Episodes", min_value=1, max_value=100, value=5)
    
    with col2:
        batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32, step=8)
    
    with col3:
        num_batches = st.number_input("Training Batches", min_value=1, max_value=100, value=10)
    
    # Training process
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Start Self-Play"):
            with st.spinner("Running self-play episodes..."):
                result = trader.self_play(episodes=self_play_episodes, callback=training_callback)
                
                # Update session state with results
                st.session_state.training_history['rewards'] = result['rewards']
                actions_counts = result['actions']
                st.session_state.training_history['actions'] = actions_counts
                
                # Save trader to session state
                st.session_state.trader = trader
                
                st.success(f"Self-play completed with {self_play_episodes} episodes")
    
    with col2:
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                metrics = trader.train(batch_size=batch_size, num_batches=num_batches)
                
                if metrics:
                    # Update session state with metrics
                    st.session_state.training_history['policy_loss'] = metrics['policy_loss']
                    st.session_state.training_history['value_loss'] = metrics['value_loss']
                    st.session_state.training_history['total_loss'] = metrics['total_loss']
                    
                    # Save trader to session state
                    st.session_state.trader = trader
                    
                    st.success("Model training completed")
                else:
                    st.error("Training failed. Make sure to run self-play first to collect training data.")
    
    # Self-play progress
    if st.session_state.training_progress['total_episodes'] > 0:
        progress = st.session_state.training_progress['episode'] / st.session_state.training_progress['total_episodes']
        st.progress(progress)
        
        # Display episode info
        st.info(f"Episode {st.session_state.training_progress['episode']}/{st.session_state.training_progress['total_episodes']} - "
               f"Total Reward: {st.session_state.training_progress['total_reward']:.4f}")
        
        # Show action counts
        actions_count = st.session_state.training_progress['actions_count']
        st.text(f"Actions: Buy: {actions_count[0]}, Sell: {actions_count[1]}, Hold: {actions_count[2]}")
    
    # Save/load model
    st.subheader("Model Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Model"):
            with st.spinner("Saving model..."):
                trader.save_model()
                st.success("Model saved successfully")
    
    with col2:
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                success = trader.load_model()
                if success:
                    # Update session state with loaded model
                    st.session_state.trader = trader
                    
                    # Get training history
                    if trader.training_history:
                        st.session_state.training_history = trader.training_history
                    
                    st.success("Model loaded successfully")
                else:
                    st.error("Failed to load model")
    
    # Display training history
    st.subheader("Training History")
    plot_training_history()

def backtesting_page():
    """Backtesting page for the model"""
    st.title("Model Backtesting")
    
    # Ensure data and model are loaded
    if st.session_state.trader is None or st.session_state.nifty_data is None:
        trader, nifty_df, vix_df = load_data_and_model()
    else:
        trader = st.session_state.trader
        nifty_df = st.session_state.nifty_data
        vix_df = st.session_state.vix_data
    
    # Backtesting parameters
    st.subheader("Backtesting Parameters")
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        min_date = nifty_df.index[0].date()
        max_date = nifty_df.index[-1].date()
        start_date = st.date_input("Start Date", 
                                  value=max_date - timedelta(days=90),
                                  min_value=min_date,
                                  max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", 
                                value=max_date,
                                min_value=start_date,
                                max_value=max_date)
    
    # MCTS usage
    use_mcts = st.checkbox("Use MCTS for predictions", value=False, 
                          help="Using MCTS improves prediction quality but is much slower")
    
    # Run backtest
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            results_df = trader.backtest(
                start_date=start_date,
                end_date=end_date,
                use_mcts=use_mcts,
                callback=backtest_callback
            )
            
            if results_df is not None and len(results_df) > 0:
                st.success("Backtest completed successfully")
    
    # Display backtest results
    st.subheader("Backtest Results")
    plot_backtest_results()

def prediction_page():
    """Real-time prediction page"""
    st.title("Trade Prediction")
    
    # Ensure data and model are loaded
    if st.session_state.trader is None or st.session_state.nifty_data is None:
        trader, nifty_df, vix_df = load_data_and_model()
    else:
        trader = st.session_state.trader
        nifty_df = st.session_state.nifty_data
        vix_df = st.session_state.vix_data
    
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
    
    # Data settings
    st.subheader("Data Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days_to_fetch = st.number_input("Days to Fetch", min_value=5, max_value=365, value=30)
    
    with col2:
        load_from_disk = st.checkbox("Load Data from Disk", value=True)
    
    # Apply settings
    if st.button("Apply Settings"):
        # Store settings
        st.session_state.trade_time = trade_time
        st.session_state.window_size = window_size
        st.session_state.days_to_fetch = days_to_fetch
        st.session_state.load_from_disk = load_from_disk
        
        # Reload data and model with new settings
        st.session_state.trader = None
        st.session_state.nifty_data = None
        st.session_state.vix_data = None
        
        with st.spinner("Applying settings and reloading..."):
            # Get data
            nifty_df, vix_df = get_latest_data(
                load_from_disk=load_from_disk, 
                fetch_days=days_to_fetch
            )
            
            # Initialize trader
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
            st.session_state.nifty_data = nifty_df
            st.session_state.vix_data = vix_df
        
        st.success("Settings applied successfully")
    
    # Advanced settings
    st.subheader("Advanced Settings")
    
    # Reset application
    if st.button("Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# Main function
def main():
    # Create sidebar
    sidebar()
    
    # Render active page
    if st.session_state.active_page == "Dashboard":
        dashboard_page()
    elif st.session_state.active_page == "Training":
        training_page()
    elif st.session_state.active_page == "Backtesting":
        backtesting_page()
    elif st.session_state.active_page == "Prediction":
        prediction_page()
    elif st.session_state.active_page == "Settings":
        settings_page()

if __name__ == "__main__":
    main() 