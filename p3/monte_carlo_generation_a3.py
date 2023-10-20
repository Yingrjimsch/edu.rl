class MonteCarloGeneration(object):
  def __init__(self, env: object, agent: object, max_steps: int = 1000, debug: bool = False):
    self.agent = agent
    self.env = env
    self.max_steps = max_steps
    self.debug = debug
    
  def run(self) -> list:
    buffer = []
    n_steps = 0 
    state, _, _ = self.env.reset() 
    terminal = False
    
    while not terminal and n_steps < self.max_steps: 
        action = self.agent.get_best_action(state)  # Wählen der besten Aktion basierend auf dem aktuellen Zustand
        next_state, reward, terminal = self.env.step(action)  # Ausführen der Aktion in der Umgebung
        self.agent.update_q_value(state, action, reward, next_state)  # Aktualisieren des Q-Wertes
        
        buffer.append((state, action, reward))  # Ergebnis speichern
        state = next_state  # Bereitmachen für den nächsten Schritt
        n_steps += 1
        
    if n_steps >= self.max_steps and self.debug:
        print("Terminated early due to large number of steps")
        
    return buffer


  def run_episode(self, agent) -> tuple:
    trajectory = self.run()  # Generate a trajectory
    episode_reward = sum(t[2] for t in trajectory)  # Summing up all the rewards in the trajectory
    return trajectory, episode_reward
