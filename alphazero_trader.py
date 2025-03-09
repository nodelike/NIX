import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import random
from collections import deque
import json
import time
from zoneinfo import ZoneInfo

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Define constants
ACTIONS = ["buy", "sell", "hold"]
NUM_ACTIONS = len(ACTIONS)
MAX_EPISODES = 1000
MCTS_SIMULATIONS = 100
C_PUCT = 1.0  # Controls exploration vs exploitation
BATCH_SIZE = 32
TRAIN_EPOCHS = 10
LR = 0.001
MAX_MEMORY_SIZE = 10000
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

class MarketState:
    """Represents the state of the market at a specific time"""
    
    def __init__(self, features, timestamp=None, normalized=False):
        """
        Initialize market state
        
        Args:
            features: numpy array of market features
            timestamp: timestamp of the state
            normalized: whether the features are already normalized
        """
        self.features = features
        self.timestamp = timestamp
        self.normalized = normalized
        
    def normalize(self, feature_means, feature_stds):
        """Normalize features"""
        if not self.normalized:
            self.features = (self.features - feature_means) / feature_stds
            self.normalized = True
        return self
    
    def to_input_array(self):
        """Convert state to input array for neural network"""
        return np.expand_dims(self.features, axis=0)

class Node:
    """Node in the Monte Carlo Tree Search"""
    
    def __init__(self, state, prior=0, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # Action -> Node
        self.visits = 0
        self.value_sum = 0
        self.prior = prior
        self.is_expanded = False
        
    def value(self):
        """Get average value of node"""
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    def upper_confidence_bound(self, total_visits):
        """UCB formula from AlphaZero paper"""
        if self.visits == 0:
            return float('inf')
        
        # Balance exploration vs exploitation
        exploitation = self.value()
        exploration = C_PUCT * self.prior * np.sqrt(total_visits) / (1 + self.visits)
        
        return exploitation + exploration
    
    def select_child(self):
        """Select child with highest UCB"""
        total_visits = sum(child.visits for child in self.children.values())
        return max(self.children.items(), 
                  key=lambda item: item[1].upper_confidence_bound(total_visits))
    
    def expand(self, action_priors):
        """Expand node with action probabilities from policy network"""
        self.is_expanded = True
        
        for action, prob in enumerate(action_priors):
            if prob > 0:
                self.children[action] = Node(None, prob, self)

class MCTS:
    """Monte Carlo Tree Search as used in AlphaZero"""
    
    def __init__(self, model, env, simulations=MCTS_SIMULATIONS):
        self.model = model
        self.env = env
        self.simulations = simulations
        
    def search(self, state):
        """Run MCTS simulations starting from current state"""
        root = Node(state)
        
        # Initial expansion of root node
        policy, value = self.model.predict(state)
        root.expand(policy)
        
        # Run simulations
        for _ in range(self.simulations):
            node = root
            search_path = [node]
            
            # Selection - Travel down the tree until we reach a leaf
            while node.is_expanded and node.children:
                action, node = node.select_child()
                search_path.append(node)
            
            # If node is not expanded, it's a leaf node
            parent = search_path[-2]
            state = self.env.get_next_state(parent.state, action)
            node.state = state
            
            # Expansion - if leaf is not terminal, expand with policy
            if state is not None:  # Not terminal
                policy, value = self.model.predict(state)
                node.expand(policy)
            else:  # Terminal state
                value = self.env.get_reward(parent.state, action)
            
            # Backpropagation - Update values up the search path
            for node in reversed(search_path):
                node.value_sum += value
                node.visits += 1
        
        # Get visit counts for each action from root
        visits = np.zeros(NUM_ACTIONS)
        for action, child in root.children.items():
            visits[action] = child.visits
            
        # Convert to policy
        policy = visits / np.sum(visits)
        
        return policy

class AlphaZeroModel:
    """Neural network model for AlphaZero"""
    
    def __init__(self, input_shape, learning_rate=LR):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """Build neural network architecture"""
        inputs = Input(shape=self.input_shape)
        
        # Shared representation
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        shared = Dense(128, activation='relu')(x)
        
        # Policy head
        policy_head = Dense(64, activation='relu')(shared)
        policy_output = Dense(NUM_ACTIONS, activation='softmax', name='policy')(policy_head)
        
        # Value head
        value_head = Dense(64, activation='relu')(shared)
        value_output = Dense(1, activation='tanh', name='value')(value_head)  # tanh for [-1, 1] value
        
        # Create model with two outputs
        model = Model(inputs=inputs, outputs=[policy_output, value_output])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            }
        )
        
        return model
    
    def predict(self, state):
        """Predict policy and value for given state"""
        inputs = state.to_input_array()
        policy, value = self.model.predict(inputs, verbose=0)
        return policy[0], value[0][0]
    
    def train(self, states, policies, values):
        """Train model on batch of examples"""
        inputs = np.vstack([state.to_input_array()[0] for state in states])
        
        self.model.fit(
            inputs,
            {'policy': np.array(policies), 'value': np.array(values)},
            batch_size=BATCH_SIZE,
            epochs=TRAIN_EPOCHS,
            verbose=0
        )
        
    def save(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
        
    def load(self, filepath):
        """Load model from file"""
        self.model.load_weights(filepath)

class TradingEnvironment:
    """Trading environment for the AlphaZero agent"""
    
    def __init__(self, nifty_data, vix_data, features_extractor, 
                 trade_time='09:05', commission=0.0003, slippage=0.0002,
                 window_size=20, test_mode=False):
        """
        Initialize trading environment
        
        Args:
            nifty_data: DataFrame with NIFTY data
            vix_data: DataFrame with VIX data
            features_extractor: Function to extract features
            trade_time: Time to trade every day (string HH:MM)
            commission: Commission percentage
            slippage: Slippage percentage for market orders
            window_size: Number of days to use for feature creation
            test_mode: Whether to run in test mode
        """
        self.nifty_data = nifty_data
        self.vix_data = vix_data
        self.features_extractor = features_extractor
        self.trade_time = trade_time
        self.commission = commission
        self.slippage = slippage
        self.window_size = window_size
        self.test_mode = test_mode
        
        # Prepare data and normalize features
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and align data"""
        # Ensure both dataframes have DatetimeIndex
        if not isinstance(self.nifty_data.index, pd.DatetimeIndex):
            self.nifty_data.index = pd.to_datetime(self.nifty_data.index)
            
        if not isinstance(self.vix_data.index, pd.DatetimeIndex):
            self.vix_data.index = pd.to_datetime(self.vix_data.index)
        
        # Combine indices data at the specified trade time
        self.nifty_data['date'] = self.nifty_data.index.date
        self.vix_data['date'] = self.vix_data.index.date
        
        daily_data = []
        
        for date in sorted(set(self.nifty_data['date'])):
            date_str = str(date)
            trade_dt = pd.to_datetime(f"{date_str} {self.trade_time}")
            
            # Get closest data points to trade time
            nifty_slice = self.nifty_data.loc[self.nifty_data['date'] == date]
            vix_slice = self.vix_data.loc[self.vix_data['date'] == date]
            
            if len(nifty_slice) > 0 and len(vix_slice) > 0:
                nifty_at_time = nifty_slice.iloc[nifty_slice.index.get_indexer([trade_dt], method='nearest')[0]]
                vix_at_time = vix_slice.iloc[vix_slice.index.get_indexer([trade_dt], method='nearest')[0]]
                
                daily_data.append({
                    'date': date,
                    'datetime': trade_dt,
                    'nifty_open': nifty_at_time['Open'],
                    'nifty_high': nifty_at_time['High'],
                    'nifty_low': nifty_at_time['Low'],
                    'nifty_close': nifty_at_time['Close'],
                    'nifty_volume': nifty_at_time['Volume'],
                    'vix_open': vix_at_time['Open'],
                    'vix_high': vix_at_time['High'],
                    'vix_low': vix_at_time['Low'],
                    'vix_close': vix_at_time['Close'],
                    'vix_volume': vix_at_time.get('Volume', 0),
                })
        
        self.daily_data = pd.DataFrame(daily_data)
        self.daily_data.set_index('datetime', inplace=True)
        
        # Extract features
        self.features_list = []
        self.feature_means = None
        self.feature_stds = None
        
        for i in range(self.window_size, len(self.daily_data)):
            window = self.daily_data.iloc[i-self.window_size:i]
            features = self.features_extractor(window)
            self.features_list.append({
                'features': features,
                'date': self.daily_data.iloc[i]['date'],
                'timestamp': self.daily_data.index[i]
            })
        
        # Calculate feature statistics for normalization
        features_array = np.array([f['features'] for f in self.features_list])
        self.feature_means = np.mean(features_array, axis=0)
        self.feature_stds = np.std(features_array, axis=0) + 1e-8  # Avoid division by zero
        
    def get_state(self, idx):
        """Get state at given index"""
        if idx < 0 or idx >= len(self.features_list):
            return None
        
        features = self.features_list[idx]['features']
        timestamp = self.features_list[idx]['timestamp']
        
        state = MarketState(features, timestamp)
        return state.normalize(self.feature_means, self.feature_stds)
    
    def get_next_state(self, state, action):
        """Get next state after action"""
        # Find current state index
        current_idx = -1
        for i, f in enumerate(self.features_list):
            if f['timestamp'] == state.timestamp:
                current_idx = i
                break
        
        if current_idx == -1 or current_idx >= len(self.features_list) - 1:
            return None  # Terminal state
            
        return self.get_state(current_idx + 1)
    
    def get_reward(self, state, action):
        """Calculate reward for taking action from state"""
        if state is None:
            return 0
            
        # Find state in data
        state_idx = -1
        for i, f in enumerate(self.features_list):
            if f['timestamp'] == state.timestamp:
                state_idx = i
                break
                
        if state_idx == -1 or state_idx >= len(self.features_list) - 1:
            return 0  # No reward at terminal state
            
        # Get current and next day's prices
        current_price = self.daily_data.iloc[state_idx + self.window_size]['nifty_close']
        next_price = self.daily_data.iloc[state_idx + self.window_size + 1]['nifty_close']
        
        # Calculate price change percentage
        price_change = (next_price - current_price) / current_price
        
        # Calculate reward based on action
        if action == 0:  # Buy
            reward = price_change - self.commission - self.slippage
        elif action == 1:  # Sell
            reward = -price_change - self.commission - self.slippage
        else:  # Hold
            reward = 0
            
        return reward

class ReplayBuffer:
    """Experience replay buffer to store training data"""
    
    def __init__(self, max_size=MAX_MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, policy, value):
        """Add experience to buffer"""
        self.buffer.append((state, policy, value))
        
    def sample(self, batch_size):
        """Sample batch of experiences from buffer"""
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
            
        states, policies, values = zip(*batch)
        return states, policies, values
        
    def __len__(self):
        return len(self.buffer)

class AlphaZeroTrader:
    """AlphaZero-based trading system"""
    
    def __init__(self, nifty_data, vix_data, features_extractor, 
                 input_shape, trade_time='09:05', model_dir='models'):
        """
        Initialize AlphaZero trader
        
        Args:
            nifty_data: DataFrame with NIFTY data
            vix_data: DataFrame with VIX data
            features_extractor: Function to extract features
            input_shape: Shape of input features
            trade_time: Time to trade every day
            model_dir: Directory to save models
        """
        self.nifty_data = nifty_data
        self.vix_data = vix_data
        self.features_extractor = features_extractor
        self.input_shape = input_shape
        self.trade_time = trade_time
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model and environment
        self.model = AlphaZeroModel(input_shape)
        self.env = TradingEnvironment(nifty_data, vix_data, features_extractor, trade_time)
        self.mcts = MCTS(self.model, self.env)
        self.replay_buffer = ReplayBuffer()
        
        # Training metrics
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'rewards': [],
            'actions': [[] for _ in range(NUM_ACTIONS)]
        }
        
    def self_play(self, episodes=1):
        """Execute self-play to generate training data"""
        for episode in range(episodes):
            episode_memory = []
            total_reward = 0
            
            # Start from a random valid state
            start_idx = np.random.randint(0, len(self.env.features_list))
            state = self.env.get_state(start_idx)
            
            while state is not None:
                # Add Dirichlet noise to root for exploration
                policy = self.mcts.search(state)
                
                # Add exploration noise at the root node
                noise = np.random.dirichlet([DIRICHLET_ALPHA] * NUM_ACTIONS)
                policy = (1 - DIRICHLET_EPSILON) * policy + DIRICHLET_EPSILON * noise
                
                # Select action based on the MCTS search
                if np.random.random() < 0.05:  # 5% random exploration
                    action = np.random.choice(NUM_ACTIONS)
                else:
                    action = np.argmax(policy)
                
                # Get reward and next state
                reward = self.env.get_reward(state, action)
                next_state = self.env.get_next_state(state, action)
                
                # Store experience
                episode_memory.append((state, policy, None))  # Value will be updated later
                total_reward += reward
                
                # Update training history for UI
                self.training_history['actions'][action].append(1)
                for a in range(NUM_ACTIONS):
                    if a != action:
                        self.training_history['actions'][a].append(0)
                
                # Move to next state
                state = next_state
            
            # Calculate returns with bootstrapped values
            running_value = 0
            for i in reversed(range(len(episode_memory))):
                state, policy, _ = episode_memory[i]
                
                # Use model's value prediction to bootstrap
                _, value = self.model.predict(state)
                running_value = value  # Use model's value estimate
                
                # Update experience with final value
                episode_memory[i] = (state, policy, running_value)
            
            # Add episode memory to replay buffer
            for experience in episode_memory:
                self.replay_buffer.add(*experience)
            
            # Update training history
            self.training_history['rewards'].append(total_reward)
            
            # Print episode information
            print(f"Episode {episode+1}/{episodes} - Reward: {total_reward:.4f}")
    
    def train(self, batch_size=BATCH_SIZE, num_batches=10):
        """Train the model on collected experiences"""
        if len(self.replay_buffer) < batch_size:
            print(f"Not enough samples in replay buffer ({len(self.replay_buffer)}/{batch_size})")
            return
        
        for i in range(num_batches):
            states, policies, values = self.replay_buffer.sample(batch_size)
            
            # Train model
            inputs = np.vstack([state.to_input_array()[0] for state in states])
            
            history = self.model.model.fit(
                inputs,
                {'policy': np.array(policies), 'value': np.array(values)},
                batch_size=batch_size,
                epochs=1,
                verbose=0
            )
            
            # Update training history
            self.training_history['policy_loss'].append(history.history['policy_loss'][0])
            self.training_history['value_loss'].append(history.history['value_loss'][0])
            self.training_history['total_loss'].append(history.history['loss'][0])
            
            if i % 5 == 0:
                print(f"Batch {i+1}/{num_batches} - "
                     f"Loss: {history.history['loss'][0]:.4f}, "
                     f"Policy Loss: {history.history['policy_loss'][0]:.4f}, "
                     f"Value Loss: {history.history['value_loss'][0]:.4f}")
    
    def save_model(self, filename='alphazero_model'):
        """Save model to file"""
        filepath = os.path.join(self.model_dir, f"{filename}.h5")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save training history
        history_path = os.path.join(self.model_dir, f"{filename}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)
            
    def load_model(self, filename='alphazero_model'):
        """Load model from file"""
        filepath = os.path.join(self.model_dir, f"{filename}.h5")
        if os.path.exists(filepath):
            self.model.load(filepath)
            print(f"Model loaded from {filepath}")
            
            # Load training history if available
            history_path = os.path.join(self.model_dir, f"{filename}_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
            
            return True
        else:
            print(f"Model file not found: {filepath}")
            return False
    
    def predict(self, state, use_mcts=True):
        """Make a trading decision for the given state"""
        if use_mcts:
            # Use MCTS for better predictions
            policy = self.mcts.search(state)
            action = np.argmax(policy)
            confidence = policy[action]
        else:
            # Use neural network directly (faster but less accurate)
            policy, value = self.model.predict(state)
            action = np.argmax(policy)
            confidence = policy[action]
            
        return {
            'action': ACTIONS[action],
            'action_index': action,
            'confidence': float(confidence),
            'policy': policy.tolist(),
            'timestamp': state.timestamp
        }
            
    def backtest(self, start_date=None, end_date=None, use_mcts=False):
        """Backtest the trading strategy"""
        print("Starting backtest...")
        
        # Get all valid states
        states = []
        for i in range(len(self.env.features_list)):
            state = self.env.get_state(i)
            
            # Filter by date range if provided
            if start_date and state.timestamp.date() < start_date:
                continue
            if end_date and state.timestamp.date() > end_date:
                continue
                
            states.append(state)
        
        # Run predictions on all states
        results = []
        capital = 100000  # Initial capital
        position = 0  # Current position (0 = none, 1 = long, -1 = short)
        entry_price = 0
        
        for i, state in enumerate(states):
            prediction = self.predict(state, use_mcts=use_mcts)
            action = prediction['action_index']
            
            # Get price data
            state_idx = -1
            for j, f in enumerate(self.env.features_list):
                if f['timestamp'] == state.timestamp:
                    state_idx = j
                    break
            
            if state_idx == -1:
                continue
                
            current_price = self.env.daily_data.iloc[state_idx + self.env.window_size]['nifty_close']
            
            # Update P&L if we have a position
            pnl = 0
            if position == 1:  # Long position
                pnl = (current_price - entry_price) / entry_price
            elif position == -1:  # Short position
                pnl = (entry_price - current_price) / entry_price
            
            # Execute trading decision
            if action == 0:  # Buy
                if position <= 0:  # Close short position if exists and open long
                    if position == -1:
                        # Close short position
                        trade_pnl = (entry_price - current_price) / entry_price
                        capital *= (1 + trade_pnl - self.env.commission - self.env.slippage)
                    
                    # Open long position
                    position = 1
                    entry_price = current_price
            
            elif action == 1:  # Sell
                if position >= 0:  # Close long position if exists and open short
                    if position == 1:
                        # Close long position
                        trade_pnl = (current_price - entry_price) / entry_price
                        capital *= (1 + trade_pnl - self.env.commission - self.env.slippage)
                    
                    # Open short position
                    position = -1
                    entry_price = current_price
            
            # Store results
            results.append({
                'date': state.timestamp,
                'action': ACTIONS[action],
                'confidence': prediction['confidence'],
                'price': current_price,
                'position': position,
                'capital': capital,
                'pnl': pnl
            })
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(states)} days - Current capital: {capital:.2f}")
        
        results_df = pd.DataFrame(results)
        
        # Calculate performance metrics
        if len(results_df) > 0:
            total_return = (results_df['capital'].iloc[-1] / 100000) - 1
            daily_returns = results_df['capital'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            max_drawdown = (results_df['capital'] / results_df['capital'].cummax() - 1).min()
            
            print(f"Backtest Results:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            
        return results_df

def extract_features(window):
    """Extract features from price data window"""
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
        nifty_returns.iloc[-1],  # 1-day return
        nifty_returns.iloc[-5:].mean(),  # 5-day mean return
        nifty_volatility.iloc[-1],  # Recent volatility
        price_to_ma5.iloc[-1] - 1,  # Deviation from 5-day MA
        price_to_ma10.iloc[-1] - 1,  # Deviation from 10-day MA
        vix_returns.iloc[-1],  # 1-day VIX return
        vix_ratio.iloc[-1] - 1,  # VIX deviation from 5-day MA
        vix_percentile.iloc[-1],  # VIX rank percentile
        current_day['vix_close'],  # Current VIX level
        current_day['nifty_volume'] / window['nifty_volume'].mean(),  # Volume ratio
    ])
    
    return features 