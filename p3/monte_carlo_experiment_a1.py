from collections import defaultdict
from monte_carlo_generation_a1 import MonteCarloGeneration
import random
from point import Point
import numpy as np

class MonteCarloExperiment(object):
  def __init__(self, env: MonteCarloGeneration, epsilon = .1):
    self.env = env
    self.epsilon = epsilon
    self.values = defaultdict(float)
    self.counts = defaultdict(float)
    self.random_counter = 0

  def _to_key(self, state, action):
    return (state, action)
  
  def action_value(self, state, action) -> float:
    key = self._to_key(state, action)
    if self.counts[key] > 0:
      return self.values[key] / self.counts[key]
    else:
      return 0.0
    
  def state_value_function(self, pos) -> float:
    """Compute the value of a state."""
    # Example: The value of a state might be the average of the values of all possible actions in this state.
    state_value = sum([self.action_value(pos, d) for d in self.env.action_space]) / len(self.env.action_space)
    return state_value
  
  def get_best_action(self):
    if random.random() < self.epsilon:
      self.random_counter += 1
      return random.choice(self.env.action_space)
    
    state_values = np.array([[x[0], self.state_value_function(x[1])] for x in self.env.get_next_possible_pos(self.env.cur_pos)])
    return np.random.choice(state_values[:, 0], p=self.normalize_state_values(state_values[:, 1]))
  
  def normalize_state_values(self, state_values):
    if len(set(state_values)) == 1:
      return [1/len(state_values)] * len(state_values)
    state_values = (state_values - np.min(state_values))/(np.max(state_values) - np.min(state_values))
    return [float(i)/sum(state_values) for i in state_values]
    