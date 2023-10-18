from typing import List
import random

class MonteCarloGeneration(object):
  def __init__(self, env: object, max_steps: int = 1000, debug: bool = False):
    self.env = env
    self.max_steps = max_steps
    self.debug = debug

  def run(self) -> List:
    buffer = []
    n_steps = 0 # Keep track of the number of steps so I can bail out if it takes too long
    state, _, _ = self.env.reset() # Reset environment back to start
    terminal = False
    while not terminal: # Run until terminal state
      action = random.choice(self.env.action_space) # Random action. Try replacing this with Direction.EAST
      next_state, reward, terminal = self.env.step(action) # Take action in environment
      buffer.append((state, action, reward)) # Store the result
      state = next_state # Ready for the next step
      n_steps += 1
      if n_steps >= self.max_steps:
        if self.debug:
          print("Terminated early due to large number of steps")
        terminal = True # Bail out if we've been working for too long
    return buffer