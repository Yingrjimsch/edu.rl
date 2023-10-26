def update_basic(agent, key, reward):
    agent.counts[key] += 1  # Increment counter
    # And add this to the value of this action
    agent.values[key] += reward


def update_sutton_2_3(agent, key, reward):
    agent.counts[key] += 1  # Increment counter
    # And add this to the value of this action
    agent.values[key] += reward / (agent.all_counts[key] - (agent.counts[key] -1))

