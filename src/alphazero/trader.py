import numpy as np
import os
import json
import pandas as pd
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Local imports
from .model import AlphaZeroModel
from .mcts import MCTS
from .environment import TradingEnvironment, ReplayBuffer

# Define constants
ACTIONS = ["buy", "sell", "hold"]
NUM_ACTIONS = len(ACTIONS)
MAX_EPISODES = 1000
BATCH_SIZE = 32
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

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
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
        
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
        
    def self_play(self, episodes=1, callback=None):
        """
        Execute self-play to generate training data
        
        Args:
            episodes: Number of episodes to play
            callback: Callback function for UI updates
        
        Returns:
            Dictionary of metrics from self-play
        """
        episode_rewards = []
        episode_actions = []
        
        for episode in range(episodes):
            episode_memory = []
            total_reward = 0
            actions_count = [0, 0, 0]  # Buy, Sell, Hold
            
            # Start from a random valid state
            start_idx = np.random.randint(0, len(self.env.features_list))
            state = self.env.get_state(start_idx)
            
            step = 0
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
                
                # Update action counts
                actions_count[action] += 1
                
                # Update training history for UI
                self.training_history['actions'][action].append(1)
                for a in range(NUM_ACTIONS):
                    if a != action:
                        self.training_history['actions'][a].append(0)
                
                # Move to next state
                state = next_state
                step += 1
                
                # Provide progress update through callback
                if callback and step % 10 == 0:
                    callback({
                        'episode': episode + 1,
                        'total_episodes': episodes,
                        'step': step,
                        'action': ACTIONS[action],
                        'reward': reward,
                        'total_reward': total_reward,
                        'actions_count': actions_count.copy()
                    })
            
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
            episode_rewards.append(total_reward)
            episode_actions.append(actions_count)
            
            # Print episode information
            print(f"Episode {episode+1}/{episodes} - Reward: {total_reward:.4f} - Actions: Buy: {actions_count[0]}, Sell: {actions_count[1]}, Hold: {actions_count[2]}")
            
            # Final callback for episode completion
            if callback:
                callback({
                    'episode': episode + 1,
                    'total_episodes': episodes,
                    'step': step,
                    'action': 'done',
                    'reward': 0,
                    'total_reward': total_reward,
                    'actions_count': actions_count,
                    'episode_complete': True
                })
        
        return {
            'rewards': episode_rewards,
            'actions': episode_actions
        }
    
    def train(self, batch_size=BATCH_SIZE, num_batches=10, callback=None):
        """
        Train the model on collected experiences
        
        Args:
            batch_size: Size of training batches
            num_batches: Number of batches to train on
            callback: Callback function for UI updates
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < batch_size:
            print(f"Not enough samples in replay buffer ({len(self.replay_buffer)}/{batch_size})")
            return None
            
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
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
            
            # Update metrics
            policy_loss = history.history['policy_loss'][0]
            value_loss = history.history['value_loss'][0]
            total_loss = history.history['loss'][0]
            
            metrics['policy_loss'].append(policy_loss)
            metrics['value_loss'].append(value_loss)
            metrics['total_loss'].append(total_loss)
            
            # Update training history
            self.training_history['policy_loss'].append(policy_loss)
            self.training_history['value_loss'].append(value_loss)
            self.training_history['total_loss'].append(total_loss)
            
            if i % 5 == 0 or i == num_batches - 1:
                print(f"Batch {i+1}/{num_batches} - "
                     f"Loss: {total_loss:.4f}, "
                     f"Policy Loss: {policy_loss:.4f}, "
                     f"Value Loss: {value_loss:.4f}")
                
            # Provide progress update through callback
            if callback:
                callback({
                    'batch': i + 1,
                    'total_batches': num_batches,
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'total_loss': total_loss,
                    'progress': (i + 1) / num_batches
                })
        
        return metrics
    
    def save_model(self, filename='alphazero_model'):
        """Save model to file"""
        filepath = os.path.join(self.model_dir, f"{filename}.h5")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save training history
        history_path = os.path.join(self.model_dir, f"{filename}_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            serializable_history = {
                'policy_loss': [float(x) for x in self.training_history['policy_loss']],
                'value_loss': [float(x) for x in self.training_history['value_loss']],
                'total_loss': [float(x) for x in self.training_history['total_loss']],
                'rewards': [float(x) for x in self.training_history['rewards']],
                'actions': [[int(a) for a in action_list] for action_list in self.training_history['actions']]
            }
            json.dump(serializable_history, f)
            
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
        """
        Make a trading decision for the given state
        
        Args:
            state: Market state to predict for
            use_mcts: Whether to use MCTS for prediction
            
        Returns:
            Dictionary with prediction details
        """
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
            
    def backtest(self, start_date=None, end_date=None, use_mcts=False, callback=None):
        """
        Backtest the trading strategy
        
        Args:
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            use_mcts: Whether to use MCTS for predictions
            callback: Callback function for UI updates
            
        Returns:
            DataFrame with backtest results
        """
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
        daily_returns = []
        
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
            
            # Calculate daily return
            if i > 0:
                daily_return = capital / results[-1]['capital'] - 1
                daily_returns.append(daily_return)
            
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
            
            # Provide progress update through callback
            if callback and (i % 5 == 0 or i == len(states) - 1):
                callback({
                    'current': i + 1,
                    'total': len(states),
                    'date': state.timestamp,
                    'action': ACTIONS[action],
                    'position': position,
                    'capital': capital,
                    'progress': (i + 1) / len(states)
                })
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(states)} days - Current capital: {capital:.2f}")
        
        results_df = pd.DataFrame(results)
        
        # Calculate performance metrics
        if len(results_df) > 0:
            total_return = (results_df['capital'].iloc[-1] / 100000) - 1
            sharpe_ratio = 0
            if len(daily_returns) > 0:
                daily_returns_array = np.array(daily_returns)
                sharpe_ratio = np.sqrt(252) * daily_returns_array.mean() / (daily_returns_array.std() + 1e-8)
            max_drawdown = (results_df['capital'] / results_df['capital'].cummax() - 1).min()
            
            print(f"Backtest Results:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            
            # Add metrics to results
            metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': results_df['action'].isin(['buy', 'sell']).sum(),
                'final_capital': results_df['capital'].iloc[-1]
            }
            
            # Provide final results through callback
            if callback:
                callback({
                    'complete': True,
                    'metrics': metrics,
                    'results_df': results_df
                })
            
        return results_df 