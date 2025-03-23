# Import required libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Constants
LR = 0.001  # Learning rate

class AlphaZeroModel:
    """Neural network model for AlphaZero"""
    
    def __init__(self, input_shape, n_actions=3, learning_rate=LR):
        """
        Initialize AlphaZero model
        
        Args:
            input_shape: Shape of the state input (can be 1D, 2D, or 3D)
            n_actions: Number of possible actions
            learning_rate: Learning rate for the optimizer
        """
        # Ensure input_shape is a tuple
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)
        elif not isinstance(input_shape, tuple):
            # If it's not a tuple, make it a tuple with a single value
            input_shape = (input_shape,)
        
        # If input shape is too simplistic, add dimensions to make it compatible
        if len(input_shape) == 1:
            # For 1D inputs, reshape to (1, n_features)
            input_shape = (1, input_shape[0])
        
        # Store model parameters
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        
        # Print input shape for debugging
        print(f"AlphaZeroModel input shape: {self.input_shape}")
        
        try:
            # Build the neural network
            self.policy_model, self.value_model = self._build_model()
        except Exception as e:
            print(f"Error building model: {e}")
            import traceback
            traceback.print_exc()
            
            # Create emergency fallback models that at least won't crash
            self._build_fallback_models()
    
    def _build_model(self):
        """Build the neural network model architecture"""
        # Use a simpler model architecture that's more compatible with TensorFlow
        
        # Determine flattened input size
        flattened_size = np.prod(self.input_shape)
        
        # Policy model
        self.policy_model = Sequential([
            # Use Input layer first to properly set input shape
            Input(shape=self.input_shape),
            # Flatten any input shape
            Flatten(),
            # Dense layers with proper regularization
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            # Output layer for action probabilities
            Dense(self.n_actions, activation='softmax', name='policy_output')
        ])
        
        # Value model
        self.value_model = Sequential([
            # Use Input layer first to properly set input shape
            Input(shape=self.input_shape),
            # Flatten any input shape
            Flatten(),
            # Dense layers with proper regularization
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            # Output layer for state value
            Dense(1, activation='tanh', name='value_output')
        ])
        
        # Compile both models
        self.policy_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy'
        )
        
        self.value_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )
        
        # Print model summaries
        print("Policy Model:")
        self.policy_model.summary()
        print("\nValue Model:")
        self.value_model.summary()
        
        return self.policy_model, self.value_model
    
    def predict(self, state):
        """
        Get policy and value predictions from the model
        
        Args:
            state: State to predict for
            
        Returns:
            (policy, value): Policy probabilities and value estimate
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
                
            # Ensure we have a proper shape for prediction
            # Keras models expect a specific batch dimension
            original_shape = state.shape
            
            # Reshape strategy:
            # 1. Flatten the array completely
            flat_state = state.flatten()
            
            # 2. Reshape to match expected input with explicit batch dimension
            # The input_shape doesn't include batch dimension, so we add it as the first dimension
            if len(self.input_shape) == 3:  # (time_steps, features, channels) format
                # Attempt to reshape to the expected format with batch_size=1
                expected_elements = np.prod(self.input_shape)
                if len(flat_state) == expected_elements:
                    reshaped_state = flat_state.reshape(1, *self.input_shape)
                else:
                    # Use a safe fallback shape if dimensions don't match
                    print(f"Warning: Unable to reshape state {original_shape} to expected input shape {self.input_shape}")
                    print(f"Using uniform policy and zero value.")
                    return np.ones(self.n_actions) / self.n_actions, 0.0
            elif len(self.input_shape) == 2:  # (time_steps, features) format
                expected_elements = np.prod(self.input_shape)
                if len(flat_state) == expected_elements:
                    reshaped_state = flat_state.reshape(1, *self.input_shape)
                else:
                    print(f"Warning: Unable to reshape state {original_shape} to expected input shape {self.input_shape}")
                    print(f"Using uniform policy and zero value.")
                    return np.ones(self.n_actions) / self.n_actions, 0.0
            else:  # Generic fallback
                # Just add a batch dimension at the front
                reshaped_state = np.expand_dims(state, axis=0)
            
            # Ensure we have float32 data type for TensorFlow
            reshaped_state = reshaped_state.astype(np.float32)
            
            # Use model.predict with a properly formatted input
            try:
                # Make predictions with each model - use a simpler approach to avoid shape issues
                policy = self.policy_model(reshaped_state).numpy()
                value = self.value_model(reshaped_state).numpy()
                
                # Extract results from batch dimension
                policy = policy[0]
                value = value[0][0]
                
                return policy, value
            except Exception as inner_e:
                print(f"Error during model inference: {inner_e}")
                print(f"Input shape was: {reshaped_state.shape}")
                # Fall back to returning a uniform policy and zero value
                return np.ones(self.n_actions) / self.n_actions, 0.0
                
        except Exception as e:
            print(f"Error in model prediction: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to returning a uniform policy and zero value
            return np.ones(self.n_actions) / self.n_actions, 0.0
    
    def train(self, states, policies, values):
        """
        Train the model on a batch of examples
        
        Args:
            states: List of states
            policies: List of policy vectors
            values: List of value scalars
            
        Returns:
            Dictionary with losses for policy and value networks
        """
        if len(states) == 0:
            print("Error: No training data provided")
            return {"policy_loss": 0, "value_loss": 0, "total_loss": 0}
            
        try:
            # Convert to numpy arrays if needed
            if not isinstance(states, np.ndarray):
                states = np.array(states, dtype=np.float32)
            if not isinstance(policies, np.ndarray):
                policies = np.array(policies, dtype=np.float32)
            if not isinstance(values, np.ndarray):
                values = np.array(values, dtype=np.float32)
                
            # Print shapes for debugging
            print(f"Training shapes - States: {states.shape}, Policies: {policies.shape}, Values: {values.shape}")
            
            # Ensure values are properly shaped for training
            if len(values.shape) == 1:
                values = values.reshape(-1, 1)
                
            # Train the policy network
            policy_history = self.policy_model.fit(
                states, policies,
                epochs=1,
                batch_size=32,
                verbose=0
            )
            
            # Train the value network
            value_history = self.value_model.fit(
                states, values,
                epochs=1,
                batch_size=32,
                verbose=0
            )
            
            # Extract losses
            policy_loss = policy_history.history['loss'][0]
            value_loss = value_history.history['loss'][0]
            total_loss = policy_loss + value_loss
            
            # Return the loss values
            return {
                "policy_loss": float(policy_loss),
                "value_loss": float(value_loss),
                "total_loss": float(total_loss)
            }
        
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            # Return zeros for losses if training failed
            return {"policy_loss": 0, "value_loss": 0, "total_loss": 0}
    
    def save(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Base file path to save to (without extension)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save policy and value models separately
            policy_path = f"{filepath}_policy.keras"
            value_path = f"{filepath}_value.keras"
            
            print(f"Saving policy model to {policy_path}...")
            self.policy_model.save(policy_path)
            
            print(f"Saving value model to {value_path}...")
            self.value_model.save(value_path)
            
            print(f"Models saved successfully to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load(self, filepath):
        """
        Load the model from a file
        
        Args:
            filepath: Base file path to load from (without extension)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Check if model files exist
            policy_path = f"{filepath}_policy.keras"
            value_path = f"{filepath}_value.keras"
            
            if not os.path.exists(policy_path) or not os.path.exists(value_path):
                # Try older format (h5)
                policy_path = f"{filepath}_policy.h5"
                value_path = f"{filepath}_value.h5"
                
                if not os.path.exists(policy_path) or not os.path.exists(value_path):
                    print(f"Model files not found at {filepath}")
                    return False
            
            print(f"Loading policy model from {policy_path}...")
            self.policy_model = tf.keras.models.load_model(policy_path)
            
            print(f"Loading value model from {value_path}...")
            self.value_model = tf.keras.models.load_model(value_path)
            
            print(f"Models loaded successfully from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback models
            self._build_fallback_models()
            return False
    
    def _build_fallback_models(self):
        """Build extremely simple fallback models that won't crash"""
        print("Using fallback models due to error in main model creation")
        
        # Create the simplest possible models - just flattening and a single layer
        # These won't be good at prediction but at least they won't crash
        
        # Policy model
        self.policy_model = Sequential([
            # Use a more forgiving input layer
            Input(shape=(None,), ragged=True),
            # Simplest possible layer setup
            Dense(self.n_actions, activation='softmax')
        ])
        
        # Value model
        self.value_model = Sequential([
            # Use a more forgiving input layer
            Input(shape=(None,), ragged=True),
            # Simplest possible layer setup
            Dense(1, activation='tanh')
        ])
        
        # Compile with simple optimizers
        self.policy_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy'
        )
        
        self.value_model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        ) 