from yahtzee_envs.yahtzee_singleV1 import YahtzeeSinglePlayerV1
import math
from q_agentv2 import QAgentV2


env = YahtzeeSinglePlayerV1(0, 0)
agent = QAgentV2(1, 0.8, 0.8)

# Main algorithm
for i in range(1, 10001):
    #agent.eps = round(100 * math.exp(-.000005 * i), 2)
    # Make first move
    if i == 9000:
        agent.eps = 0.05
    done = False
    first = True
    while done is False:
        action = 0
        if first:
            game, reward, done, debug = env.step(26)
            first = False
        else:
            action = agent.get_action(env.ytz)
            game, reward, done, debug = env.step(action)
        next_action = agent.get_action(env.ytz)
        game, reward, done, debug = env.step(next_action)
        if action < 13:
            indices = env.ytz.c_player.t_scorecard[action][1]
        elif action < 26:
            indices = env.ytz.c_player.t_scorecard[action - 13][1]
        else:
            indices = [0, 0, 0, 0, 0]
        agent.update_q(action, next_action, indices, reward)
    agent.scores.append(env.ytz.c_player.score)
    agent.iterations.append(i)
    env.reset()


# Data output
agent.max_score = max(agent.scores)
agent.games_played = agent.iterations[-1]
agent.avg_score = sum(agent.scores) / agent.games_played
agent.last_avg_score = sum(agent.scores[-100:]) / 100
agent.write_data("./data/q_agentv3_data/10k.txt")
agent.generate_plot("./data/q_agentv3_data/10k.png")