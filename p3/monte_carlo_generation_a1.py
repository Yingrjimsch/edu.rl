from typing import List

class MonteCarloGeneration(object):
  def __init__(self, env: object, agent: object, max_steps: int = 1000, debug: bool = False):
    self.env = env
    self.agent = agent
    self.max_steps = max_steps
    self.debug = debug
    
# update statement 
  def update_values(self, key, reward):
      self.agent.counts[key] += 1 # Increment counter
      self.agent.values[key] += reward # And add this to the value of this action



  def run(self) -> List:
    buffer = []
    n_steps = 0 # Keep track of the number of steps so I can bail out if it takes too long
    state, _, _ = self.env.reset() # Reset environment back to start
    terminal = False
    while not terminal: # Run until terminal state
      action = self.agent.get_best_action() # HERE: new state-value function einfügen bzw. aufrufen
      next_state, reward, terminal = self.env.step(action) # Take action in environment
      buffer.append((state, action, reward)) # Store the result
      state = next_state # Ready for the next step
      n_steps += 1
      if n_steps >= self.max_steps:
        if self.debug:
          print("Terminated early due to large number of steps")
        terminal = True # Bail out if we've been working for too long
    return buffer


# bestehende Version

  def run_episode(self) -> None: # analog update nach vollendeter Episode
    trajectory = self.run() # Generate a trajectory
    episode_reward = 0
    for i, t in enumerate(reversed(trajectory)): # Starting from the terminal state
      state, action, reward = t
      key = self.agent._to_key(state, action)
      episode_reward += reward  # Add the reward to the buffer
      self.update_values(key, episode_reward)
    return trajectory, episode_reward # neu





