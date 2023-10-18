from point import Point

def state_value_2d(env, agent):
    res = ""
    for y in reversed(range(env.height)):
      for x in range(env.width):
        if env.goal.x == x and env.goal.y == y:
          res += "   @  "
        else:
          state_value = sum([agent.action_value(Point(x,y), d) for d in env.action_space]) / len(env.action_space)
          res += f'{state_value:6.2f}'
        res += " | "
      res += "\n"
    return res

def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])

def next_best_value_2d(env, agent):
    res = ""
    for y in reversed(range(env.height)):
      for x in range(env.width):
        if env.goal.x == x and env.goal.y == y:
          res += "@"
        else:
          # Find the action that has the highest value
          loc = argmax([agent.action_value(Point(x,y), d) for d in env.action_space])
          res += f'{env.action_space[loc].value}'
        res += " | "
      res += "\n"
    return res
