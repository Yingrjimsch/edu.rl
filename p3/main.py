# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:48:20 2023

@author: vonwareb
"""

from simple_grid_world import SimpleGridWorld
from monte_carlo_experiment_a3 import MonteCarloExperiment
from monte_carlo_generation_a3 import MonteCarloGeneration
from visualize import state_value_2d, next_best_value_2d


def main():
    env = SimpleGridWorld(width=4, height=4, debug=False)  # Beispielumgebung
    agent = MonteCarloExperiment(env)
    generator = MonteCarloGeneration(env=env, agent=agent, max_steps=1000, debug=True)

    # Um den Agenten basierend auf einer Episode zu aktualisieren:
    for i in range(1000):
        trajectory, episode_reward = generator.run_episode(agent)
        print(f"total reward: {sum([t[2] for t in trajectory])}") # Print final reward
    print(state_value_2d(env, agent))
    print(next_best_value_2d(env, agent))
        
        
if __name__ == "__main__":
    main()