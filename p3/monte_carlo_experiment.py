from collections import defaultdict
from monte_carlo_generation import MonteCarloGeneration

class MonteCarloExperiment(object):
  def __init__(self, generator: MonteCarloGeneration):
    self.generator = generator
    self.values = defaultdict(float)
    self.counts = defaultdict(float)

  def _to_key(self, state, action):
    return (state, action)
  
  def action_value(self, state, action) -> float:
    key = self._to_key(state, action)
    if self.counts[key] > 0:
      return self.values[key] / self.counts[key]
    else:
      return 0.0
    
  def run_episode(self) -> None:
    
    trajectory = self.generator.run() # Generate a trajectory
    episode_reward = 0
    for i, t in enumerate(reversed(trajectory)): # Starting from the terminal state
      state, action, reward = t
      key = self._to_key(state, action)
      episode_reward += reward  # Add the reward to the buffer
      self.values[key] += episode_reward # And add this to the value of this action
      self.counts[key] += 1 # Increment counter