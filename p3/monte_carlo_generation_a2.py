from typing import List
import random

class MonteCarloGeneration(object):
  def __init__(self, env: object, agent: object, max_steps: int = 1000, debug: bool = False):
    self.env = env
    self.agent = agent
    self.max_steps = max_steps
    self.debug = debug
    

# update statement mc
  def update_values_mc(self, key, reward):
      if key not in self.agent.counts:
          self.agent.counts[key] = 0
          self.agent.values[key] = 0
      self.agent.counts[key] += 1
      self.agent.values[key] =  self.agent.values[key]  + (1 / self.agent.counts[key]) * (reward - self.agent.values[key] )
  

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
      #self.update_values(key, episode_reward)
      #self.update_values_mc(key, episode_reward)
      self.update_values_mc(key, episode_reward)
    return trajectory, episode_reward # neu


#Every Visit Monte Carlo
'''
  def run_episode(self) -> None:
        trajectory = self.run()  # Generate a trajectory
        episode_reward = 0
        for i, t in enumerate(reversed(trajectory)):  # Starting from the terminal state
            state, action, reward = t
            
            # This is the key for the state, not state-action pair as before.
            key = self.agent._to_key(state, action)
            
            episode_reward += reward  # Cumulative reward till this state
    
            # Incremental update for the state value
            if key not in self.agent.counts:
                self.agent.counts[key] = 0
                self.agent.values[key] = 0
            
            # Compute the average return for this state using the formula
            n = self.agent.counts[key]
            total_value = self.agent.values[key] * n + episode_reward  # Multiply old average by count to get total value, and add new return
            n += 1  # Increment the visit count
            self.agent.values[key] = total_value / n  # Update the average return for this state
            
            self.agent.counts[key] = n  # Update the counts
    
        return trajectory, episode_reward
'''


  # Incremental Montecarlo
'''
  def run_episode(self) -> None: # analog update nach vollendeter Episode
        trajectory = self.run()  # Generate a trajectory
        episode_reward = 0
        for i, t in enumerate(reversed(trajectory)):  # Starting from the terminal state
            state, action, reward = t
            key = self.agent._to_key(state, action)
            episode_reward += reward  # Cumulative reward till this state
    
            # Inkrementelle Aktualisierung
            if key not in self.agent.counts:
                self.agent.counts[key] = 0
                self.agent.values[key] = 0
    
            old_value = self.agent.values[key]
            n = self.agent.counts[key]
            
            # Inkrementelle Monte Carlo Update-Formel
            self.agent.values[key] = old_value + (1 / (n + 1)) * (episode_reward - old_value)
            #self.agent.values[key] = old_value + 0.05 * (episode_reward - old_value)
            self.agent.counts[key] += 1
    
        return trajectory, episode_reward

'''



