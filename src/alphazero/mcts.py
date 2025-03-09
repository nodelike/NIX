import numpy as np

# Default MCTS parameters
C_PUCT = 1.0  # Controls exploration vs exploitation
MCTS_SIMULATIONS = 100

class Node:
    """Node in the Monte Carlo Tree Search"""
    
    def __init__(self, state, prior=0, parent=None):
        """
        Initialize a node in the MCTS tree
        
        Args:
            state: The market state at this node
            prior: Prior probability from policy network
            parent: Parent node
        """
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
        """
        Initialize MCTS
        
        Args:
            model: The neural network model (policy and value)
            env: The trading environment
            simulations: Number of simulations per search
        """
        self.model = model
        self.env = env
        self.simulations = simulations
        
    def search(self, state):
        """
        Run MCTS simulations starting from current state
        
        Args:
            state: Starting state
            
        Returns:
            Policy distribution over actions
        """
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
        visits = np.zeros(3)  # 3 actions: buy, sell, hold
        for action, child in root.children.items():
            visits[action] = child.visits
            
        # Convert to policy
        policy = visits / np.sum(visits)
        
        return policy 