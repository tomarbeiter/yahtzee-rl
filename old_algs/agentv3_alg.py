from yahtzee_singleV2 import YahtzeeSinglePlayerV2
import math
from agentv3 import QAgentV3


env = YahtzeeSinglePlayerV2()
agent = QAgentV3(1, 0.99, 0.99)

# Main algorithm
for i in range(1, 20001):
    # eps decay
    agent.eps = math.exp(-.00013 * i) if math.exp(-.00013 * i) > .085 else .085
    done = False
    first_roll = True
    while done is False:
        turns_left = env.ytz.remaining_turns
        # Player must roll all 5 dice on first roll of each turn
        if first_roll:
            game, reward, done, debug = env.step(26)
            new_action, new_state = agent.get_action_and_state(game)
            agent.update_q(26, 0, new_action, new_state, reward)
            action = new_action
            state = new_state
            first_roll = False
        else:
            game, reward, done, debug = env.step(action)
            new_action, new_state = agent.get_action_and_state(game)
            agent.update_q(action, state, new_action, new_state, reward)
            # These two lines are irrelevant if the action is scoring b/c
            # they are reset on the new turn
            action = new_action
            state = new_state
            if turns_left > env.ytz.remaining_turns:
                first_roll = True
    agent.scores.append(env.ytz.c_player.score)
    agent.iterations.append(i)
    env.reset()


# Data output
agent.max_score = max(agent.scores)
agent.games_played = agent.iterations[-1]
agent.avg_score = sum(agent.scores) / agent.games_played
agent.last_avg_score = sum(agent.scores[-100:]) / 100
agent.write_data("./data/q_agentv3_data/envV2/20k.txt")
agent.generate_plot("./data/q_agentv3_data/envV2/20k.png")

agent.eps = 0
done = False
first_roll = True
while done is False:
    turns_left = env.ytz.remaining_turns
    # Player must roll all 5 dice on first roll of each turn
    if first_roll:
        game, reward, done, debug = env.step(26)
        new_action, new_state = agent.get_action_and_state(game)
        agent.update_q(26, 0, new_action, new_state, reward)
        action = new_action
        state = new_state
        first_roll = False
    else:
        game, reward, done, debug = env.step(action)
        new_action, new_state = agent.get_action_and_state(game)
        agent.update_q(action, state, new_action, new_state, reward)
        # These two lines are irrelevant if the action is scoring b/c
        # they are reset on the new turn
        action = new_action
        state = new_state
        if turns_left > env.ytz.remaining_turns:
            first_roll = True
agent.scores.append(env.ytz.c_player.score)
agent.iterations.append(i)
env.reset()
