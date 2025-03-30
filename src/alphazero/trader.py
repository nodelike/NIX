import os
import sys
import random
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time
import matplotlib.pyplot as plt
import tensorflow as tf

# Local imports
from src.alphazero.model import AlphaZeroModel
from src.alphazero.mcts import MCTS
from src.alphazero.environment import TradingEnvironment, ReplayBuffer

# Define constants
LR = 0.001
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

class AlphaZeroTrader:
    """AlphaZero-based trading system"""
    
    def __init__(self, input_shape, n_actions=3, features_extractor=None, 
                 nifty_data=None, vix_data=None, trade_time='9:15', model_dir='models'):
        """
        Initialize the AlphaZero trader
        
        Args:
            input_shape: Shape of the input state
            n_actions: Number of possible actions (buy, sell, hold)
            features_extractor: Function to extract features from data
            nifty_data: Optional dataframe with NIFTY data
            vix_data: Optional dataframe with VIX data
            trade_time: Time to trade every day (string HH:MM)
            model_dir: Directory to save/load model
        """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.features_extractor = features_extractor
        self.nifty_data = nifty_data
        self.vix_data = vix_data
        self.trade_time = trade_time
        self.model_dir = model_dir
        
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize the environment if data is provided
        self.env = None
        if nifty_data is not None and vix_data is not None and features_extractor is not None:
            self.env = TradingEnvironment(
                nifty_data=nifty_data,
                vix_data=vix_data,
                features_extractor=features_extractor,
                trade_time=trade_time,
                test_mode=False
            )
            
        # Initialize the neural network model
        try:
            self.model = AlphaZeroModel(input_shape, n_actions=self.n_actions)
            print(f"Initialized AlphaZeroModel with input shape {input_shape}")
        except Exception as e:
            print(f"Error initializing AlphaZeroModel: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
        
        # Initialize MCTS 
        try:
            self.mcts = MCTS(self.model)
        except Exception as e:
            print(f"Error initializing MCTS: {e}")
            self.mcts = None
        
        # Replay buffer for training
        self.replay_buffer = ReplayBuffer()
        
        # Training history
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'rewards': [],
            'actions': {}
        }
        
        # Initialize total reward for tracking
        self.total_reward = 0
        
        # Check shape consistency
        self._check_shape_consistency()
        
    def _check_shape_consistency(self):
        """Verify and fix input shape consistency between model and environment"""
        try:
            # Get a sample state from the environment
            if hasattr(self.env, 'features_list') and len(self.env.features_list) > 0:
                sample_features = self.env.features_list[0]['features']
                
                # Check if model exists and has input_shape attribute
                if hasattr(self, 'model') and hasattr(self.model, 'input_shape'):
                    model_shape = self.model.input_shape
                    features_shape = sample_features.shape
                    
                    # If shapes mismatch (ignoring batch dimension)
                    if len(model_shape) >= 3 and len(features_shape) >= 3:
                        if model_shape[1:] != features_shape[1:]:
                            print(f"Shape mismatch detected: Model expects {model_shape}, features are {features_shape}")
                            # Reinitialize model with correct shape
                            self.model = AlphaZeroModel(features_shape, n_actions=self.n_actions)
                            print(f"Model reinitialized with shape {features_shape}")
        except Exception as e:
            print(f"Error checking shape consistency: {e}")
            print(f"Model shape: {getattr(self.model, 'input_shape', 'unknown')}")
            print(f"Sample features shape: {getattr(sample_features, 'shape', 'unknown') if 'sample_features' in locals() else 'unknown'}")
        
    def self_play(self, episodes=1, callback=None):
        """
        Generate training examples through self-play
        
        Args:
            episodes: Number of episodes to play
            callback: Optional callback function to receive progress updates
        
        Returns:
            List of training examples
        """
        if self.env is None or self.model is None or self.mcts is None:
            print("Error: Environment, model, or MCTS not initialized")
            return []
            
        total_examples = []
        total_reward = 0
        actions_count = {'buy': 0, 'sell': 0, 'hold': 0}
        capital_changes = []
        
        # Run self-play episodes
        for episode in range(episodes):
            try:
                print(f"Starting self-play episode {episode+1}/{episodes}")
                
                # Reset the environment
                try:
                    state = self.env.reset()
                except Exception as reset_error:
                    print(f"Error resetting environment: {reset_error}")
                    continue  # Skip this episode
                
                # Training examples for this episode
                examples = []
                
                # Episode statistics
                episode_reward = 0
                actions_taken = []
                step = 0
                max_steps = min(200, len(self.env.valid_indices) - 1)  # Prevent infinite loops
                
                # Separate dice for Dirichlet noise
                dice = np.random.dirichlet([DIRICHLET_ALPHA] * self.n_actions)
                
                # Main self-play loop
                done = False
                while not done and step < max_steps:
                    try:
                        # Get MCTS policy and action
                        noise = dice if step == 0 else None  # Only add noise at the first step
                        try:
                            policy, action, predicted_value = self.mcts.search(state, noise=noise, epsilon=DIRICHLET_EPSILON)
                        except Exception as mcts_error:
                            print(f"MCTS search error: {mcts_error}")
                            # Fall back to random policy and action
                            policy = np.ones(self.n_actions) / self.n_actions
                            action = np.random.choice(self.n_actions)
                            predicted_value = 0.0
                        
                        # Store examples of state, policy, and value (to be updated later)
                        examples.append([state, policy, None])  # Value placeholder
                        
                        # Track action distribution
                        action_name = self.env.actions[action]
                        actions_count[action_name] += 1
                        actions_taken.append(action_name)
                        
                        # Take the action
                        try:
                            next_state, reward, done, info = self.env.step(action)
                        except Exception as step_error:
                            print(f"Error taking step in environment: {step_error}")
                            break  # End this episode
                        
                        # Update episode statistics
                        episode_reward += reward
                        
                        # Update state
                        state = next_state
                        
                        # Track capital changes if available in info
                        if info and 'capital' in info:
                            capital_changes.append(info['capital'])
                
                        # Update progress
                        if callback:
                            callback({
                                'episode': episode + 1,
                                'step': step,
                                'action': action_name,
                                'reward': reward,
                                'total_reward': episode_reward
                            })
                        
                        step += 1
                        
                    except Exception as loop_error:
                        print(f"Error in self-play loop: {loop_error}")
                        import traceback
                        traceback.print_exc()
                        break  # End this episode
                
                print(f"Episode {episode+1} completed with {step} steps and reward {episode_reward}")
                print(f"Actions taken: {', '.join(actions_taken)}")
                
                # Calculate the final outcome value based on total reward
                final_value = np.tanh(episode_reward)  # Normalize between -1 and 1
                
                # Update value targets for all states
                for i in range(len(examples)):
                    # Simple temporal difference: all states get the final outcome
                    examples[i][2] = final_value
                
                # Add examples to the total set
                total_examples.extend(examples)
                total_reward += episode_reward
                
            except Exception as episode_error:
                print(f"Error in self-play episode {episode+1}: {episode_error}")
                import traceback
                traceback.print_exc()
                continue  # Skip to next episode
        
        # Summarize self-play
        avg_reward = total_reward / max(1, episodes)
        print(f"\nSelf-play completed: {episodes} episodes, {len(total_examples)} examples")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Actions distribution: Buy: {actions_count['buy']}, Sell: {actions_count['sell']}, Hold: {actions_count['hold']}")
        
        if capital_changes:
            final_capital = capital_changes[-1]
            initial_capital = capital_changes[0]
            capital_gain = ((final_capital - initial_capital) / initial_capital) * 100
            print(f"Capital gain: {capital_gain:.2f}% (from {initial_capital:.2f} to {final_capital:.2f})")
            
        # Convert examples to numpy arrays for the buffer
        states, policies, values = zip(*total_examples) if total_examples else ([], [], [])
        
        # Add examples to replay buffer - use the correct method
        for i in range(len(states)):
            self.replay_buffer.add(states[i], policies[i], values[i])
        
        print(f"Added {len(states)} examples to replay buffer")
            
            # Update training history
        self.training_history['rewards'].append(avg_reward)
        for action_name, count in actions_count.items():
            if action_name not in self.training_history['actions']:
                self.training_history['actions'][action_name] = []
            self.training_history['actions'][action_name].append(count)
        
        return total_examples
    
    def train(self, batch_size=BATCH_SIZE, num_batches=10, callback=None):
        """
        Train the model on examples from the replay buffer
        
        Args:
            batch_size: Number of examples per batch
            num_batches: Number of batches to train on
            callback: Optional callback function for progress updates
        
        Returns:
            Dictionary with training statistics
        """
        if self.model is None:
            print("Error: Model not initialized")
            return {"error": "Model not initialized"}
            
        if len(self.replay_buffer) < 10:
            print(f"Warning: Not enough examples in buffer for training (found {len(self.replay_buffer)}, need at least 10)")
            return {"error": "Not enough examples in buffer"}
            
        print(f"\nStarting model training: {num_batches} batches of {batch_size} examples")
        
        # Initialize tracking
        policy_losses = []
        value_losses = []
        total_losses = []
        
        # Track actions for reporting
        action_totals = {'buy': 0, 'sell': 0, 'hold': 0}
        
        try:
            # Train for specified number of batches
            for i in range(num_batches):
                # Sample batch from replay buffer
                states, policies, values = self.replay_buffer.sample(batch_size)
                
                if not states:
                    print("Error: Failed to sample from replay buffer")
                    break
                    
                # Convert to numpy arrays if needed
                if not isinstance(states, np.ndarray):
                    states = np.array(states)
                if not isinstance(policies, np.ndarray):
                    policies = np.array(policies)
                if not isinstance(values, np.ndarray):
                    values = np.array(values)
                
                # Train the model on this batch
                try:
                    losses = self.model.train(states, policies, values)
                    
                    # Extract individual losses
                    policy_loss = losses["policy_loss"]
                    value_loss = losses["value_loss"]
                    total_loss = losses["total_loss"]
                    
                    # Track losses
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    total_losses.append(total_loss)
                    
                    # Count actions in this batch 
                    for policy in policies:
                        action = np.argmax(policy)
                        action_name = self.env.actions[action]
                        action_totals[action_name] += 1
                    
                    # Report progress
                    if i % max(1, num_batches // 10) == 0 or i == num_batches - 1:
                        print(f"Batch {i+1}/{num_batches}: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}, total_loss={total_loss:.4f}")
                    
                    # Update callback if provided
                    if callback:
                        callback({
                            'batch': i + 1,
                            'total_batches': num_batches,
                            'policy_loss': policy_loss,
                            'value_loss': value_loss,
                            'total_loss': total_loss,
                            'progress': (i + 1) / num_batches
                        })
                
                except Exception as train_error:
                    print(f"Error in batch {i+1}: {train_error}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            
        # Calculate summary statistics
        summary = {
            'batches_completed': len(total_losses),
            'avg_policy_loss': np.mean(policy_losses) if policy_losses else 0,
            'avg_value_loss': np.mean(value_losses) if value_losses else 0,
            'avg_total_loss': np.mean(total_losses) if total_losses else 0,
            'action_totals': action_totals
        }
        
        # Print final summary
        print("\nTraining summary:")
        print(f"Completed {summary['batches_completed']} batches")
        print(f"Average losses - Policy: {summary['avg_policy_loss']:.4f}, Value: {summary['avg_value_loss']:.4f}, Total: {summary['avg_total_loss']:.4f}")
        print(f"Action distribution: Buy: {action_totals['buy']}, Sell: {action_totals['sell']}, Hold: {action_totals['hold']}")
        
        # Calculate trading performance metrics
        total_trades = action_totals['buy'] + action_totals['sell']
        profitable_trades = int(total_trades * (np.mean(self.training_history['rewards']) + 1) / 2)  # Rough estimate
        win_rate = profitable_trades / max(1, total_trades)
        
        print(f"Training performance estimates:")
        print(f"Total trades: {total_trades}")
        print(f"Profitable trades: {profitable_trades} (Win rate: {win_rate:.2%})")
        
        # Update training history
        self.training_history['policy_loss'].extend(policy_losses)
        self.training_history['value_loss'].extend(value_losses)
        self.training_history['total_loss'].extend(total_losses)
        
        return summary
    
    def save_model(self, filename='alphazero_model'):
        """Save model to file"""
        # Use path without extension
        filepath = os.path.join(self.model_dir, filename)
        
        # Save model with proper extension handling
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
                'actions': self.training_history['actions']
            }
            json.dump(serializable_history, f)
        
    def load_model(self, filename='alphazero_model'):
        """
        Load model from file
        
        Args:
            filename: Name of model file (without extension)
        
        Returns:
            True if successful, False otherwise
        """
        model_path = os.path.join(self.model_dir, f"{filename}")
        
        # First, clean up model files to ensure consistent naming
        self.cleanup_model_files(filename)
        
        if not os.path.exists(model_path + ".h5") and not os.path.exists(model_path + "_policy.keras"):
            print(f"Model file not found: {model_path}")
            print("Initializing a new model...")
            
            # Initialize a new model with the correct input shape
            try:
                # First, check if we have a sample state to determine shape
                if hasattr(self, 'env') and hasattr(self.env, 'features_list') and len(self.env.features_list) > 0:
                    sample_features = self.env.features_list[0]['features']
                    # Ensure the shape is in the correct format
                    if len(sample_features.shape) == 3:
                        self.input_shape = sample_features.shape
                    else:
                        # Reshape to (1, 1, features)
                        self.input_shape = (1, 1, sample_features.size)
                
                print(f"Using input shape: {self.input_shape}")
                self.model = AlphaZeroModel(self.input_shape, n_actions=self.n_actions)
                
                # Save the newly initialized model
                os.makedirs(self.model_dir, exist_ok=True)
                self.model.save(model_path)
                print(f"New model saved to {model_path}")
                
                return False  # Indicate no model was loaded, but we initialized a new one
            except Exception as e:
                print(f"Error initializing model: {e}")
                return False
        
        try:
            # Load the model
            print(f"Loading model from {model_path}...")
            self.model.load(model_path)
            
            # Check and ensure shape compatibility
            self._check_shape_consistency()
            
            print(f"Model successfully loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing a new model as fallback...")
            
            try:
                # Initialize a new model as fallback
                self.model = AlphaZeroModel(self.input_shape, n_actions=self.n_actions)
                return False
            except Exception as e2:
                print(f"Error initializing fallback model: {e2}")
            return False
            
    def cleanup_model_files(self, filename='alphazero_model'):
        """
        Clean up model files to ensure consistent naming conventions
        
        Args:
            filename: Base model filename (without extension)
        """
        try:
            base_path = os.path.join(self.model_dir, filename)
            
            # Check for inconsistent file naming patterns
            problematic_files = []
            
            # List all files in the model directory
            all_files = []
            if os.path.exists(self.model_dir):
                all_files = os.listdir(self.model_dir)
            
            # Create standard filenames
            standard_policy = f"{filename}_policy.keras"
            standard_value = f"{filename}_value.keras"
            standard_history = f"{filename}_history.json"
            
            # Check if standard files already exist
            has_standard_policy = os.path.exists(os.path.join(self.model_dir, standard_policy))
            has_standard_value = os.path.exists(os.path.join(self.model_dir, standard_value))
            
            # Pattern matching for policy files
            for file in all_files:
                full_path = os.path.join(self.model_dir, file)
                
                # Skip directories
                if not os.path.isfile(full_path):
                    continue
                
                # Handle policy files
                if (filename in file and 'policy' in file and file != standard_policy and
                    (file.endswith('.keras') or file.endswith('.h5'))):
                    
                    # Skip if we already have a standard file and this is older
                    if has_standard_policy:
                        std_mtime = os.path.getmtime(os.path.join(self.model_dir, standard_policy))
                        file_mtime = os.path.getmtime(full_path)
                        
                        if file_mtime < std_mtime:
                            print(f"Skipping older policy file: {file}")
                            continue
                    
                    target_file = os.path.join(self.model_dir, standard_policy)
                    print(f"Found inconsistent policy file: {file}")
                    print(f"Renaming to: {standard_policy}")
                    
                    try:
                        # If target already exists and we're here, we should replace it
                        if os.path.exists(target_file):
                            os.remove(target_file)
                            
                        # Rename the file to the standard format
                        os.rename(full_path, target_file)
                        print(f"Successfully renamed to standard format")
                        has_standard_policy = True
                    except Exception as e:
                        print(f"Warning: Could not rename file: {e}")
                        problematic_files.append(file)
                
                # Handle value files
                elif (filename in file and 'value' in file and file != standard_value and
                     (file.endswith('.keras') or file.endswith('.h5'))):
                    
                    # Skip if we already have a standard file and this is older
                    if has_standard_value:
                        std_mtime = os.path.getmtime(os.path.join(self.model_dir, standard_value))
                        file_mtime = os.path.getmtime(full_path)
                        
                        if file_mtime < std_mtime:
                            print(f"Skipping older value file: {file}")
                            continue
                    
                    target_file = os.path.join(self.model_dir, standard_value)
                    print(f"Found inconsistent value file: {file}")
                    print(f"Renaming to: {standard_value}")
                    
                    try:
                        # If target already exists and we're here, we should replace it
                        if os.path.exists(target_file):
                            os.remove(target_file)
                            
                        # Rename the file to the standard format
                        os.rename(full_path, target_file)
                        print(f"Successfully renamed to standard format")
                        has_standard_value = True
                    except Exception as e:
                        print(f"Warning: Could not rename file: {e}")
                        problematic_files.append(file)
                        
                # Handle history files
                elif (filename in file and 'history' in file and file != standard_history and 
                     file.endswith('.json')):
                    
                    target_file = os.path.join(self.model_dir, standard_history)
                    print(f"Found inconsistent history file: {file}")
                    print(f"Renaming to: {standard_history}")
                    
                    try:
                        # If target already exists, check which is newer
                        if os.path.exists(target_file):
                            std_mtime = os.path.getmtime(target_file)
                            file_mtime = os.path.getmtime(full_path)
                            
                            if file_mtime < std_mtime:
                                print(f"Skipping older history file")
                                continue
                            else:
                                os.remove(target_file)
                        
                        # Rename the file to the standard format
                        os.rename(full_path, target_file)
                        print(f"Successfully renamed to standard format")
                    except Exception as e:
                        print(f"Warning: Could not rename file: {e}")
                        problematic_files.append(file)
            
            return len(problematic_files) == 0
            
        except Exception as e:
            print(f"Error cleaning up model files: {e}")
            return False
    
    def predict(self, state, use_mcts=False):
        """
        Make a prediction for the given state
        
        Args:
            state: Current state to predict for
            use_mcts: Whether to use MCTS for prediction (more accurate but slower)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            print("Warning: Model not initialized, using random prediction")
            action = np.random.randint(0, self.n_actions)
            return {
                'action': action,
                'action_name': self.env.actions[action] if self.env else f"action_{action}",
                'confidence': 1.0 / self.n_actions,
                'policy': np.ones(self.n_actions) / self.n_actions,
                'value': 0.0
            }
            
        try:
            if use_mcts and self.mcts is not None:
                # Use MCTS for prediction (more accurate but slower)
                try:
                    policy, action, value = self.mcts.search(state)
                except Exception as mcts_error:
                    print(f"Error in MCTS prediction: {mcts_error}")
                    # Fall back to direct model prediction
                    policy, value = self.model.predict(state)
                    action = np.argmax(policy)
            else:
                # Use direct model prediction (faster but less accurate)
                policy, value = self.model.predict(state)
                action = np.argmax(policy)
                
            # Get action name if environment is available
            action_name = self.env.actions[action] if self.env else f"action_{action}"
            
            # Return prediction results
            return {
                'action': int(action),
                'action_name': action_name,
                'confidence': float(policy[action]),
                'policy': policy.tolist() if isinstance(policy, np.ndarray) else policy,
                'value': float(value)
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to random action
            action = np.random.randint(0, self.n_actions)
            return {
                'action': action,
                'action_name': self.env.actions[action] if self.env else f"action_{action}",
                'confidence': 1.0 / self.n_actions,
                'policy': [1.0 / self.n_actions] * self.n_actions,
                'value': 0.0,
                'error': str(e)
            }
            
    def backtest(self, start_date=None, end_date=None):
        """
        Backtest the model on historical data
        
        Args:
            start_date: Optional start date for backtesting (timezone-aware)
            end_date: Optional end date for backtesting (timezone-aware)
            
        Returns:
            Dictionary with backtest results
        """
        if self.env is None:
            print("Error: Environment not initialized")
            return {
                'error': 'Environment not initialized',
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0
            }
            
        # Set test mode for backtesting
        self.env.test_mode = True
        
        # Reset the environment
        state = self.env.reset()
        
        # Filter dates if provided
        valid_indices = list(range(len(self.env.features_list)))
        if start_date is not None or end_date is not None:
            filtered_indices = []
            for i, feature in enumerate(self.env.features_list):
                if 'timestamp' not in feature:
                    continue
                
                feature_date = feature['timestamp']
                # Ensure timezone consistency
                if hasattr(feature_date, 'tz') and feature_date.tz is None:
                    try:
                        feature_date = feature_date.tz_localize('Asia/Kolkata')
                    except:
                        pass
                        
                if start_date is not None and feature_date < start_date:
                    continue
                if end_date is not None and feature_date > end_date:
                    continue
                    
                filtered_indices.append(i)
                
            if not filtered_indices:
                print(f"Warning: No data points found in date range {start_date} to {end_date}")
                return {
                    'error': 'No data in specified date range',
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'win_rate': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0
                }
                
            valid_indices = filtered_indices
            print(f"Filtered to {len(valid_indices)} trading days in selected date range")
            
        # Use filtered indices for backtesting
        self.env.valid_indices = valid_indices
        self.env.current_idx = 0
        
        # Initialize tracking variables
        done = False
        total_reward = 0
        step = 0
        actions_taken = {'buy': 0, 'sell': 0, 'hold': 0}
        trade_history = []
        capital_history = []
        initial_capital = self.env.initial_capital
        current_capital = initial_capital
        max_capital = initial_capital
        max_drawdown = 0.0
        
        # Main backtest loop
        try:
            print(f"Starting backtest with {len(valid_indices)} trading days")
            
            while not done and step < len(valid_indices):
                # Get prediction from model
                prediction = self.predict(state, use_mcts=True)
                action = prediction['action']
                action_name = prediction['action_name']
                
                # Take action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Update tracking variables
                total_reward += reward
                actions_taken[action_name] += 1
                step += 1
                
                # Track capital if available
                if info and 'capital' in info:
                    current_capital = info['capital']
                    capital_history.append(current_capital)
                    
                    # Update max capital and drawdown
                    if current_capital > max_capital:
                        max_capital = current_capital
                    
                    # Calculate drawdown
                    drawdown = (max_capital - current_capital) / max_capital if max_capital > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Track trade if made
                if action_name in ['buy', 'sell'] and info:
                    trade_info = {
                        'date': info.get('date', '') if info.get('date') else 
                               (info.get('timestamp').date() if info.get('timestamp') else ''),
                        'timestamp': info.get('timestamp', None),
                        'action': action_name,
                        'price': info.get('price', 0),
                        'return': info.get('return', 0),
                        'capital': info.get('capital', 0)
                    }
                    trade_history.append(trade_info)
                
                # Update state
                state = next_state
                
                # Print progress every 100 steps
                if step % 100 == 0:
                    print(f"Backtest step {step}/{len(valid_indices)}, reward: {total_reward:.2f}")
            
            # Calculate backtest metrics
            total_trades = actions_taken['buy'] + actions_taken['sell']
            profitable_trades = sum(1 for trade in trade_history if trade.get('return', 0) > 0)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            total_return = (current_capital - initial_capital) / initial_capital if initial_capital > 0 else 0
            
            # Print backtest summary
            print("\nBacktest Summary:")
            print(f"Total steps: {step}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Actions: Buy: {actions_taken['buy']}, Sell: {actions_taken['sell']}, Hold: {actions_taken['hold']}")
            print(f"Total trades: {total_trades}")
            print(f"Profitable trades: {profitable_trades} (Win rate: {win_rate:.2%})")
            print(f"Initial capital: {initial_capital:.2f}, Final capital: {current_capital:.2f}")
            print(f"Total return: {total_return:.2%}")
            print(f"Max drawdown: {max_drawdown:.2%}")
            
            # Return backtest results
            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'actions': actions_taken,
                'trade_history': trade_history,
                'capital_history': capital_history,
                'initial_capital': initial_capital,
                'final_capital': current_capital
            }
            
        except Exception as e:
            print(f"Error during backtesting: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'error': str(e),
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0
            } 