from typing import List
from collections import defaultdict



class MonteCarloGeneration(object):
    def __init__(self, env: object, agent: object, max_steps: int = 1000, debug: bool = False):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.debug = debug

    def run(self) -> List:
        buffer = []
        n_steps = 0  # Keep track of the number of steps so I can bail out if it takes too long
        state, _, _ = self.env.reset()  # Reset environment back to start
        terminal = False
        while not terminal:  # Run until terminal state
            # HERE: new state-value function einfügen bzw. aufrufen
            action = self.agent.get_best_action()
            next_state, reward, terminal = self.env.step(
                action)  # Take action in environment
            buffer.append((state, action, reward))  # Store the result
            state = next_state  # Ready for the next step
            n_steps += 1
            if n_steps >= self.max_steps:
                if self.debug:
                    print("Terminated early due to large number of steps")
                terminal = True  # Bail out if we've been working for too long
        return buffer

    def run_episode_sutton_2_2(self) -> None:
        trajectory = self.run()  # Generate a trajectory
        episode_reward = 0
        rewards = defaultdict(list)
        # Starting from the terminal state
        for t in reversed(trajectory):
            state, action, reward = t
            key = self.agent._to_key(state, action)
            episode_reward += reward
            rewards[key].append(episode_reward)
            self.agent.counts[key] += 1
        for state_action in rewards.items():
            for i, value in enumerate(state_action[1]):
                #print(value, len(state_action), i)
                self.agent.values[state_action[0]] += value / (len(state_action[1]) - i) 
        #print(self.agent.values)


# bestehende Version
    def run_episode(self, update_function) -> None:  # analog update nach vollendeter Episode
        trajectory = self.run()  # Generate a trajectory
        episode_reward = 0
        # Starting from the terminal state
        for t in trajectory:
            state, action, reward = t
            key = self.agent._to_key(state, action)
            self.agent.all_counts[key] += 1
        for i, t in enumerate(reversed(trajectory)):
            state, action, reward = t
            key = self.agent._to_key(state, action)
            episode_reward += reward  # Add the reward to the buffer
            update_function(self.agent, key, episode_reward)
        return trajectory, episode_reward  # neu
