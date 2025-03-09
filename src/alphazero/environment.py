import numpy as np
import pandas as pd
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
        return np.expand_dims(self.features, axis=0)


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