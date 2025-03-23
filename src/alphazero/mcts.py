import numpy as np
import math

# Default MCTS parameters
C_PUCT = 1.0  # Controls exploration vs exploitation
MCTS_SIMULATIONS = 100

class Node:
    """Node in the MCTS tree"""
    
    def __init__(self, state, prior=0.0, parent=None):
        """
        Initialize a new node
        
        Args:
            state: Game state at this node
            prior: Prior probability of selecting this node
            parent: Parent node
        """
        self.state = state
        self.prior = prior
        self.parent = parent
        self.children = {}  # Maps actions to child nodes
        self.value_sum = 0.0  # Sum of backpropagated values
        self.visit_count = 0  # Number of visits to this node
        
    def value(self):
        """Get the mean value (Q) for this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    def is_expanded(self):
        """Check if the node has been expanded"""
        return len(self.children) > 0
        
    def expand(self, action_priors):
        """
        Expand the node with the given action priors
        
        Args:
            action_priors: Dictionary mapping actions to prior probabilities
        """
        for action, prob in action_priors.items():
            if action not in self.children:
                self.children[action] = Node(self.state, prob, self)
                
    def select_child(self, c_puct):
        """
        Select a child according to the PUCT formula
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            Selected child node
        """
        # Get total visits to the parent for UCB calculation
        parent_visits = self.visit_count or 1
        
        # Find the child with the highest UCB score
        best_score = -float('inf')
        best_child = None
        best_action = None
        
        for action, child in self.children.items():
            # UCB score = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b(N(s,b))) / (1 + N(s,a))
            q_value = child.value()
            u_value = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            ucb_score = q_value + u_value
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                best_action = action
                
        if best_child is None:
            # If no children (shouldn't happen if is_expanded is checked)
            raise ValueError("No children available for selection")
            
        return best_child
        
    def update(self, value):
        """
        Update the node statistics with a new value
        
        Args:
            value: Value to backpropagate
        """
        self.value_sum += value
        self.visit_count += 1


class MCTS:
    """Monte Carlo Tree Search for AlphaZero"""
    
    def __init__(self, model, simulations=50, c_puct=1.0):
        """
        Initialize MCTS
        
        Args:
            model: AlphaZero model for policy and value prediction
            simulations: Number of simulations to run
            c_puct: Exploration constant
        """
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        
        # Action space
        self.n_actions = 3  # buy, sell, hold
        
    def search(self, state, noise=None, epsilon=0.0):
        """
        Run MCTS search and return action probabilities and value
        
        Args:
            state: Current state
            noise: Optional Dirichlet noise to add to the root node
            epsilon: How much to weight the noise (0 = no noise)
            
        Returns:
            (policy, action, value): Tuple of policy probabilities, best action, and estimated value
        """
        try:
            # Create root node
            root = Node(state, 1.0, None)  # Prior of 1.0 for root
            
            # Get initial policy and value from the model
            try:
                policy, value = self.model.predict(state)
            except Exception as e:
                print(f"Error getting initial policy from model: {e}")
                # Fall back to uniform policy
                policy = np.ones(self.n_actions) / self.n_actions
                value = 0.0
            
            # Add exploration noise to the prior at the root if provided
            if noise is not None and epsilon > 0:
                noisy_policy = (1 - epsilon) * policy + epsilon * noise
                policy = noisy_policy
            
            # Convert policy to action-prior dictionary for expanding node
            action_priors = {action: policy[action] for action in range(len(policy))}
            
            # Expand root with policy
            root.expand(action_priors)
            
            # Run simulations
            for _ in range(self.simulations):
                # Selection
                node = root
                search_path = [node]
                
                # Select until reaching a leaf node
                while node.is_expanded():
                    try:
                        node = node.select_child(self.c_puct)
                        search_path.append(node)
                    except Exception as select_err:
                        print(f"Error in child selection: {select_err}")
                        break
                        
                # If node is terminal or already fully expanded, use its value
                if node.is_expanded():
                    value = node.value()
                else:
                    # Expansion and evaluation
                    try:
                        # Get policy and value for this state
                        policy, value = self.model.predict(node.state)
                        
                        # Convert policy to action-prior dictionary
                        action_priors = {action: policy[action] for action in range(len(policy))}
                        
                        # Expand node with action priors
                        node.expand(action_priors)
                    except Exception as predict_err:
                        print(f"Error in node expansion: {predict_err}")
                        # If prediction fails, use a default value
                        value = 0.0
                
                # Backpropagation - update values and visit counts up the search path
                for node in reversed(search_path):
                    node.update(value)
            
            # Extract visit counts from root's children
            visits = np.array([
                root.children[a].visit_count if a in root.children else 0
                for a in range(self.n_actions)
            ])
            
            # Prevent division by zero if no visits
            if np.sum(visits) == 0:
                print("Warning: No valid visits found, using uniform policy")
                visits = np.ones(self.n_actions)
                
            # Normalize to get a probability distribution
            policy = visits / np.sum(visits)
            
            # Find best action (highest visit count)
            action = np.argmax(visits)
            
            # Return the normalized visit counts as policy, selected action, and value
            return policy, action, root.value()
            
        except Exception as e:
            print(f"Error in MCTS search: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a fallback uniform policy and neutral value
            uniform_policy = np.ones(self.n_actions) / self.n_actions
            random_action = np.random.randint(0, self.n_actions)
            return uniform_policy, random_action, 0.0