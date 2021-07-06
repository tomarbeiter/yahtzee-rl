import sys
sys.path.insert(1, "C:/Users/Tom Arbeiter/Desktop/Yahtzee/yahtzee-agents")
from yahtzee_envs.yahtzee_singleV0 import YahtzeeSinglePlayerV0
from yahtzee_agents.q_agentv0 import QAgentV0
import math


# Initialize a single player environment.
env = YahtzeeSinglePlayerV0(1, 10)
# Initialize a Q-LearningV0 agent.
agent = QAgentV0(.1, .5, .5)


# Main algorithm
scores = []
iterations = []
for i in range(1, 5001):
    agent.eps = round(50 * math.exp(-.005 * i), 2)
    done = False
    act = 25
    while done is False:
        game, reward, done, debug = env.step(act)
        key = str(game.c_player.t_scorecard)
        next_act = agent.action_from_state(key, game)
        game, reward, done, debug = env.step(next_act)
        new_key = str(game.c_player.t_scorecard)
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
agent.write_data("./data/q_agentv0_data/slow_eps/low_gma3.txt")
agent.generate_plot("./data/q_agentv0_data/slow_eps/low_gma3.png")
