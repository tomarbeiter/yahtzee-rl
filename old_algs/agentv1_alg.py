"""v0 algorithm for QAgentv1

- Potentially better eps decay value (slower decay)
"""

import sys
sys.path.insert(1, "C:/Users/Tom Arbeiter/Desktop/Yahtzee/yahtzee-agents")
from yahtzee_envs.yahtzee_singleV0 import YahtzeeSinglePlayerV0
from yahtzee_agents.q_agentv1 import QAgentV1
import math
import statistics


# Initialize a single player environment.
env = YahtzeeSinglePlayerV0(1, 10)
# Initialize a Q-LearningV0 agent.
agent = QAgentV1(.9, .5, .5)


# Main algorithm
scores = []
iterations = []
for i in range(1, 500001):
    agent.eps = round(100 * math.exp(-.000005 * i), 2)
    while done is False:
        game, reward, done, debug = env.step(act)
        key = agent.make_key(game)
        next_act = agent.action_from_state(key, game)
        game, reward, done, debug = env.step(next_act)
        new_key = agent.make_key(game)
        act = agent.action_from_state(new_key, game)
        agent.update_q(key, new_key, act, reward)
    agent.scores.append(env.ytz.c_player.score)
    agent.iterations.append(i)
    env.reset()
    print(i)

# Data output
agent.max_score = max(agent.scores)
agent.games_played = agent.iterations[-1]
agent.avg_score = sum(agent.scores) / agent.games_played
agent.median = statistics.median(agent.scores)
agent.write_data("./data/q_agentv1_data/slow_eps/high_gma5.txt")
agent.generate_plot("./data/q_agentv1_data/slow_eps/high_gma5.png")
