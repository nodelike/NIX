import numpy as np
import pandas as pd
import random
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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
        # Ensure we return a consistent shape format
        # For model input, we need to return a numpy array in the right format
        features = self.features
        
        # If features is already 3D (batch, timesteps, features), keep as is
        if len(features.shape) == 3:
            return features
        
        # If features is 2D, add batch dimension if needed
        elif len(features.shape) == 2:
            # Reshape to (1, timesteps, features)
            return np.expand_dims(features, axis=0)
        
        # If features is 1D, reshape to add batch and time dimensions
        elif len(features.shape) == 1:
            # Reshape to (1, 1, features)
            return np.expand_dims(np.expand_dims(features, axis=0), axis=0)
        
        # Default case - ensure we return a numpy array
        return np.array(features)


class TradingEnvironment:
    """Trading environment for the AlphaZero agent"""
    
    def __init__(self, nifty_data=None, vix_data=None, window_size=10, trade_time='9:15', 
                 features_extractor=None, lot_size=50, initial_capital=100000, test_mode=False):
        """
        Initialize the trading environment.
        
        Args:
            nifty_data: Dataframe with NIFTY data
            vix_data: Dataframe with VIX data
            window_size: Number of days to use for state representation
            trade_time: Time to place trades (9:15 or 9:30)
            features_extractor: Function to extract features
            lot_size: Number of shares per trade
            initial_capital: Initial capital for backtest
            test_mode: Whether to run in test mode
        """
        self.nifty_data = nifty_data
        self.vix_data = vix_data
        self.window_size = window_size
        self.trade_time = trade_time
        self.features_extractor = features_extractor
        self.test_mode = test_mode
        
        # Trading parameters
        self.lot_size = lot_size
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # State tracking
        self.current_idx = 0
        self.position_held = 0
        self.last_action = "hold"
        self.trade_history = []
        
        # Initialize features list
        self.features_list = []
        
        # Process data if available
        if nifty_data is not None and vix_data is not None:
            self._prepare_data()
        
        # Define the action space
        self.actions = ["buy", "sell", "hold"]
        self.n_actions = len(self.actions)
        
        # Track current position for stop loss calculations
        self.current_position = 0  # 0 = none, 1 = long, -1 = short
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        
        # Prepare data and normalize features
        self._prepare_data()
        
        # Create valid indices list for training/testing
        self.valid_indices = list(range(len(self.features_list)))
        
        # Shuffle indices for training
        if not self.test_mode:
            np.random.shuffle(self.valid_indices)
        
    def _prepare_data(self):
        """Prepare data for training/testing"""
        try:
            # Process data if we haven't already
            if not self.features_list:
                self.features_list = self.process_data()
                
            if not self.features_list:
                print("Warning: No features extracted from data")
                return
                
            # Print summary of processed data
            print(f"Processed {len(self.features_list)} trading days")
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            import traceback
            traceback.print_exc()
        
    def get_state(self, idx):
        """Get the state at the specified index."""
        if idx < 0 or idx >= len(self.features_list):
            return None
            
        # Convert features to numpy array if needed
        features = self.features_list[idx]['features']
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Ensure the features are in the right shape for the model
        # Most RL models expect shape (batch_size, time_steps, features)
        # For a single state, batch_size=1, time_steps might be 1
        if len(features.shape) == 1:
            # If features is a 1D array, reshape to (1, 1, n_features)
            features = features.reshape(1, 1, -1)
        elif len(features.shape) == 2:
            # If features is a 2D array (time_steps, features), add batch dimension
            features = features.reshape(1, *features.shape)
            
        return features
    
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
        
        # Get high and low for the next day (for stop loss / take profit simulation)
        next_day_high = self.daily_data.iloc[state_idx + self.window_size + 1]['nifty_high']
        next_day_low = self.daily_data.iloc[state_idx + self.window_size + 1]['nifty_low']
        
        # Calculate price change percentage
        price_change = (next_price - current_price) / current_price
        
        # Initialize reward
        reward = 0
        
        # Apply the action
        if action == 0:  # Buy
            if self.current_position <= 0:  # Close short position if exists and open long
                if self.current_position == -1:
                    # Close short position
                    if current_price <= self.stop_loss_price:
                        # Stop loss hit (loss)
                        short_trade_pnl = (self.entry_price - self.stop_loss_price) / self.entry_price
                        reward -= self.commission + self.slippage
                    elif current_price >= self.take_profit_price:
                        # Take profit hit (profit)
                        short_trade_pnl = (self.entry_price - self.take_profit_price) / self.entry_price
                        reward += short_trade_pnl - self.commission - self.slippage
                    else:
                        # Regular close
                        short_trade_pnl = (self.entry_price - current_price) / self.entry_price
                        reward += short_trade_pnl - self.commission - self.slippage
                
                # Open long position
                self.current_position = 1
                self.entry_price = current_price
                self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                self.take_profit_price = current_price * (1 + self.take_profit_pct)
                
                # Check if stop loss or take profit hit on next day
                if next_day_low <= self.stop_loss_price:
                    # Stop loss hit
                    reward -= self.stop_loss_pct + self.commission + self.slippage
                    self.current_position = 0  # Position closed
                elif next_day_high >= self.take_profit_price:
                    # Take profit hit
                    reward += self.take_profit_pct - self.commission - self.slippage
                    self.current_position = 0  # Position closed
                else:
                    # Position still open, reward based on next close
                    reward += price_change - self.commission - self.slippage
        
        elif action == 1:  # Sell
            if self.current_position >= 0:  # Close long position if exists and open short
                if self.current_position == 1:
                    # Close long position
                    if current_price <= self.stop_loss_price:
                        # Stop loss hit (loss)
                        long_trade_pnl = (self.stop_loss_price - self.entry_price) / self.entry_price
                        reward -= self.commission + self.slippage
                    elif current_price >= self.take_profit_price:
                        # Take profit hit (profit)
                        long_trade_pnl = (self.take_profit_price - self.entry_price) / self.entry_price
                        reward += long_trade_pnl - self.commission - self.slippage
                    else:
                        # Regular close
                        long_trade_pnl = (current_price - self.entry_price) / self.entry_price
                        reward += long_trade_pnl - self.commission - self.slippage
                
                # Open short position
                self.current_position = -1
                self.entry_price = current_price
                self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
                self.take_profit_price = current_price * (1 - self.take_profit_pct)
                
                # Check if stop loss or take profit hit on next day
                if next_day_high >= self.stop_loss_price:
                    # Stop loss hit
                    reward -= self.stop_loss_pct + self.commission + self.slippage
                    self.current_position = 0  # Position closed
                elif next_day_low <= self.take_profit_price:
                    # Take profit hit
                    reward += self.take_profit_pct - self.commission - self.slippage
                    self.current_position = 0  # Position closed
                else:
                    # Position still open, reward based on next close
                    reward += -price_change - self.commission - self.slippage
        
        else:  # Hold
            # If we have an existing position, calculate the reward
            if self.current_position == 1:  # Long position
                if next_day_low <= self.stop_loss_price:
                    # Stop loss hit
                    reward -= self.stop_loss_pct + self.commission + self.slippage
                    self.current_position = 0  # Position closed
                elif next_day_high >= self.take_profit_price:
                    # Take profit hit
                    reward += self.take_profit_pct - self.commission - self.slippage
                    self.current_position = 0  # Position closed
                else:
                    # Position still open, reward based on next close
                    reward += price_change
            
            elif self.current_position == -1:  # Short position
                if next_day_high >= self.stop_loss_price:
                    # Stop loss hit
                    reward -= self.stop_loss_pct + self.commission + self.slippage
                    self.current_position = 0  # Position closed
                elif next_day_low <= self.take_profit_price:
                    # Take profit hit
                    reward += self.take_profit_pct - self.commission - self.slippage
                    self.current_position = 0  # Position closed
                else:
                    # Position still open, reward based on next close
                    reward += -price_change
            
            else:  # No position
                reward = 0  # No reward for holding cash
        
        return reward

    def reset(self):
        """Reset the environment to an initial state."""
        # Reset the environment
        self.current_idx = 0
        self.position_held = 0
        self.last_action = "hold"
        
        # Reset capital and trade history
        self.current_capital = self.initial_capital
        self.trade_history = []
        
        # Get the initial state
        initial_state = self.get_state(self.current_idx)
        
        return initial_state
        
    def step(self, action):
        """Take a step in the environment by placing a trade."""
        # Make sure we have features
        if self.current_idx >= len(self.features_list) - 1:
            # End of data, return terminal state
            return self.get_state(self.current_idx), 0, True, {}
            
        # Get current and next features
        current_feature = self.features_list[self.current_idx]
        next_feature = self.features_list[self.current_idx + 1]
        
        # Get the current close price
        current_price = current_feature['nifty_close']
        next_price = next_feature['nifty_close']
        
        # Move to the next timestep
        self.current_idx += 1
        self.position_held += 1
        
        # Placeholder for trade information
        info = {
            'entry_price': current_price,
            'exit_price': next_price,
            'timestamp': current_feature.get('timestamp', None),
            'date': current_feature.get('date', None),
            'price': current_price,
            'capital': self.current_capital,
            'return': 0.0  # Will be updated based on action
        }
        
        # Calculate price movement
        price_change = (next_price / current_price - 1) * 100
        
        # Default reward and return values
        reward = 0
        trade_return = 0
        
        # Calculate amount of trade based on lot size
        amount = self.lot_size * current_price
        
        # Calculate stop loss and take profit levels
        if action == 0:  # Buy action
            # Set stop loss at 0.5% below entry
            stop_loss = current_price * 0.995
            # Set take profit at 1% above entry
            take_profit = current_price * 1.01
            
            # Record the action as buy
            self.last_action = "buy"
            
            # Calculate reward based on price change
            # If price goes up, positive reward
            reward = price_change
            trade_return = price_change
            
            # Update info with trade details
            info.update({
                'action': 'buy',
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'amount': amount,
                'return': trade_return
            })
            
            # Update capital based on the trade return
            self.current_capital += (self.current_capital * (trade_return / 100))
            
            # Add to trade history
            self.trade_history.append({
                'date': next_feature.get('date', pd.Timestamp.now()),
                'action': 'buy',
                'price': current_price,
                'exit_price': next_price,
                'return': trade_return,
                'amount': amount,
                'capital': self.current_capital
            })
            
        elif action == 1:  # Sell action
            # Set stop loss at 0.5% above entry (for short position)
            stop_loss = current_price * 1.005
            # Set take profit at 1% below entry (for short position)
            take_profit = current_price * 0.99
            
            # Record the action as sell
            self.last_action = "sell"
            
            # Calculate reward based on price change
            # If price goes down, positive reward
            reward = -price_change
            trade_return = -price_change
            
            # Update info with trade details
            info.update({
                'action': 'sell',
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'amount': amount,
                'return': trade_return
            })
            
            # Update capital based on the trade return
            self.current_capital += (self.current_capital * (trade_return / 100))
            
            # Add to trade history
            self.trade_history.append({
                'date': next_feature.get('date', pd.Timestamp.now()),
                'action': 'sell',
                'price': current_price,
                'exit_price': next_price,
                'return': trade_return,
                'amount': amount,
                'capital': self.current_capital
            })
            
        else:  # Hold action
            # Record the action as hold
            self.last_action = "hold"
            
            # Small negative reward for holding to encourage action
            reward = -0.01
            
            # Update info
            info.update({
                'action': 'hold',
                'return': 0
            })
        
        # Get next state
        next_state = self.get_state(self.current_idx)
        
        # Determine if this is a terminal state
        done = self.current_idx >= len(self.features_list) - 1
        
        return next_state, reward, done, info

    def process_data(self):
        """Process NIFTY and VIX data to extract features.
        
        Returns:
            List of dictionaries containing date, features, and other info
        """
        if self.nifty_data is None or self.vix_data is None:
            print("Error: No data available for processing")
            return []
            
        # Ensure we have aligned data
        common_index = self.nifty_data.index.intersection(self.vix_data.index)
        if len(common_index) == 0:
            print("Error: No common dates between NIFTY and VIX data")
            return []
            
        nifty_df = self.nifty_data.loc[common_index]
        vix_df = self.vix_data.loc[common_index]
        
        # Make sure data is sorted by date
        nifty_df = nifty_df.sort_index()
        vix_df = vix_df.sort_index()
        
        # Ensure timezone is consistent
        if nifty_df.index.tzinfo is None:
            nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
        if vix_df.index.tzinfo is None:
            vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
        
        # Create a combined dataframe with features from both
        combined_df = pd.DataFrame(index=nifty_df.index)
        combined_df['nifty_open'] = nifty_df['Open']
        combined_df['nifty_high'] = nifty_df['High']
        combined_df['nifty_low'] = nifty_df['Low']
        combined_df['nifty_close'] = nifty_df['Close']
        combined_df['nifty_volume'] = nifty_df['Volume'] if 'Volume' in nifty_df.columns else 0
        
        combined_df['vix_open'] = vix_df['Open']
        combined_df['vix_high'] = vix_df['High']
        combined_df['vix_low'] = vix_df['Low']
        combined_df['vix_close'] = vix_df['Close']
        combined_df['vix_volume'] = vix_df['Volume'] if 'Volume' in vix_df.columns else 0
        
        # Generate features for each valid day
        features_list = []
        for i in range(self.window_size, len(combined_df)):
            window = combined_df.iloc[i - self.window_size:i + 1]
            date = window.index[-1]
            
            # Extract features
            try:
                if self.features_extractor:
                    features = self.features_extractor(window)
                else:
                    features = extract_features(window)
                    
                # Create feature dictionary with timestamp
                feature_dict = {
                    'timestamp': date,  # This is a timezone-aware timestamp
                    'date': date.date(),  # Store date separately for easy access
                    'features': features,
                    'nifty_open': window['nifty_open'].iloc[-1],
                    'nifty_high': window['nifty_high'].iloc[-1],
                    'nifty_low': window['nifty_low'].iloc[-1],
                    'nifty_close': window['nifty_close'].iloc[-1],
                    'vix_close': window['vix_close'].iloc[-1],
                }
                
                features_list.append(feature_dict)
            except Exception as e:
                print(f"Error extracting features for {date}: {e}")
                continue
        
        return features_list


class ReplayBuffer:
    """Experience replay buffer to store training data"""
    
    def __init__(self, max_size=10000):
        """
        Initialize replay buffer
        
        Args:
            max_size: Maximum size of buffer
        """
        self.buffer = []
        self.max_size = max_size
        self.index = 0
        
    def add(self, state, policy, value):
        """Add experience to buffer"""
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, policy, value))
        else:
            # Replace old experiences using circular buffer
            self.buffer[self.index] = (state, policy, value)
            self.index = (self.index + 1) % self.max_size
        
    def sample(self, batch_size):
        """Sample batch of experiences from buffer"""
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            
        states, policies, values = zip(*batch)
        return states, policies, values
        
    def __len__(self):
        return len(self.buffer) 