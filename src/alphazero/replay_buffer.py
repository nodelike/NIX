import numpy as np

# Constants
MAX_BUFFER_SIZE = 10000

class ReplayBuffer:
    """Buffer to store self-play examples for training"""
    
    def __init__(self, capacity=MAX_BUFFER_SIZE):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of examples to store
        """
        self.states = []
        self.policies = []
        self.values = []
        self.capacity = capacity
        
    def add_examples(self, states, policies, values):
        """
        Add a batch of examples to the buffer
        
        Args:
            states: List of states
            policies: List of policy vectors
            values: List of value scalars
        """
        if len(states) != len(policies) or len(states) != len(values):
            print(f"Error: Mismatched lengths - states: {len(states)}, policies: {len(policies)}, values: {len(values)}")
            return
            
        # Add examples
        self.states.extend(states)
        self.policies.extend(policies)
        self.values.extend(values)
        
        # Trim if over capacity
        if len(self.states) > self.capacity:
            excess = len(self.states) - self.capacity
            self.states = self.states[excess:]
            self.policies = self.policies[excess:]
            self.values = self.values[excess:]
            
        print(f"Added {len(states)} examples to buffer. Buffer size: {len(self.states)}")
        
    def sample(self, batch_size):
        """
        Sample a batch of examples from the buffer
        
        Args:
            batch_size: Number of examples to sample
            
        Returns:
            Tuple of (states, policies, values)
        """
        buffer_size = len(self.states)
        if buffer_size == 0:
            return [], [], []
            
        # Ensure we don't sample more than what's available
        batch_size = min(batch_size, buffer_size)
        
        # Sample random indices
        indices = np.random.choice(buffer_size, batch_size, replace=False)
        
        # Get samples
        sampled_states = [self.states[i] for i in indices]
        sampled_policies = [self.policies[i] for i in indices]
        sampled_values = [self.values[i] for i in indices]
        
        return sampled_states, sampled_policies, sampled_values
        
    def __len__(self):
        """Get the number of examples in the buffer"""
        return len(self.states) 