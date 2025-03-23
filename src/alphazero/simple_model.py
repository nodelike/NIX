"""
A simpler version of the AlphaZero model to avoid TensorFlow/Keras compatibility issues.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

class SimpleAlphaZeroModel:
    """A simplified Neural Network model for AlphaZero"""
    
    def __init__(self, input_shape, n_actions=3, learning_rate=0.001):
        """
        Initialize the AlphaZero model
        
        Args:
            input_shape: Shape of the input state
            n_actions: Number of possible actions (buy, sell, hold)
            learning_rate: Learning rate for the optimizer
        """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        
        # Build the neural network model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build a simpler neural network model architecture"""
        # Input layer
        input_layer = Input(shape=self.input_shape)
        
        # Flatten the input (handles any input shape)
        x = Flatten()(input_layer)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        
        # Policy head (action probabilities)
        policy_output = Dense(self.n_actions, activation='softmax', name='policy')(x)
        
        # Value head (state value)
        value_output = Dense(1, activation='tanh', name='value')(x)
        
        # Create model with both outputs
        model = Model(inputs=input_layer, outputs=[policy_output, value_output])
        
        # Compile model with separate loss functions
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            }
        )
        
        return model
    
    def predict(self, state):
        """
        Get policy and value predictions from the model
        
        Args:
            state: State to predict for
            
        Returns:
            (policy, value): Policy probabilities and value estimate
        """
        try:
            # Convert state to numpy array if needed
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            
            # Add batch dimension if needed
            if len(state.shape) == len(self.input_shape):
                state = np.expand_dims(state, axis=0)
                
            # Ensure the state has the right shape
            if len(state.shape) > len(self.input_shape) + 1:
                # Too many dimensions, try to reshape
                state = state.reshape(-1, *self.input_shape)
                
            # Make prediction
            policy, value = self.model.predict(state, verbose=0)
            
            # Extract from batch dimension
            policy = policy[0]
            value = value[0][0]
            
            return policy, value
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            import traceback
            traceback.print_exc()
            
            # Return uniform policy and zero value if error
            return np.ones(self.n_actions) / self.n_actions, 0.0
    
    def train(self, states, policies, values):
        """
        Train the model on a batch of examples
        
        Args:
            states: Array of states
            policies: Array of policies (action probabilities)
            values: Array of values (expected return)
            
        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        try:
            # Convert inputs to numpy arrays
            states = np.array(states)
            policies = np.array(policies)
            values = np.array(values).reshape(-1, 1)  # Reshape to column vector
            
            # Train for one batch
            history = self.model.fit(
                x=states,
                y=[policies, values],
                batch_size=len(states),
                verbose=0,
                epochs=1
            )
            
            # Extract loss values
            total_loss = history.history['loss'][0]
            policy_loss = history.history.get('policy_loss', [0])[0]
            value_loss = history.history.get('value_loss', [0])[0]
            
            return total_loss, policy_loss, value_loss
            
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0
    
    def save(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load model from file"""
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            return True
        return False 