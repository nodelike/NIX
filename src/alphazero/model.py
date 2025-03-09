import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Default hyperparameters
LR = 0.001
TRAIN_EPOCHS = 10
BATCH_SIZE = 32

class AlphaZeroModel:
    """Neural network model for AlphaZero"""
    
    def __init__(self, input_shape, learning_rate=LR):
        """
        Initialize the AlphaZero model
        
        Args:
            input_shape: Shape of input features
            learning_rate: Learning rate for optimizer
        """
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
        policy_output = Dense(3, activation='softmax', name='policy')(policy_head)
        
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