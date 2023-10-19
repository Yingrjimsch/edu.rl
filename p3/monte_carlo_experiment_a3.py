from collections import defaultdict, namedtuple
import random

class MonteCarloExperiment(object):
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.values = defaultdict(float)  # Expected returns for state-action pairs
        self.env = env  # The environment in which the agent operates
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

    def _to_key(self, state, action):
        return (state, action)
    
    def argmax(self, a):
        return max(range(len(a)), key=lambda x: a[x])

    def action_value(self, state, action) -> float:
        """Compute the value of a state-action pair."""
        key = self._to_key(state, action)
        return self.values[key]

    def get_best_action(self, state):
        # ε-greedy
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.env.action_space)
        
        # Exploitation
        q_values = [self.action_value(state, a) for a in self.env.action_space]
        return self.env.action_space[self.argmax(q_values)]

    def update_q_value(self, state, action, reward, next_state):
        max_q_next_state = max([self.action_value(next_state, a) for a in self.env.action_space])
        key = self._to_key(state, action)
        self.values[key] = self.values[key] + self.alpha * (reward + self.gamma * max_q_next_state - self.values[key])