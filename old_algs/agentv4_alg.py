from yahtzee_singleV3 import YahtzeeSinglePlayerV3
import math
from agentv4 import QAgentV4


env = YahtzeeSinglePlayerV3()
agent = QAgentV4(1, 0.99, 0.99)

# Main algorithm
for i in range(1, 2001):
    # eps decay
    agent.eps = math.exp(-.0025 * i) if math.exp(-.0025 * i) > .093 else .085
    done = False
    first_roll = True
    while done is False:
        turns_left = env.ytz.remaining_turns
        # Player must roll all 5 dice on first roll of each turn
        if first_roll:
            agent, new_state, reward, done, debug = env.step(26, agent)
            new_action = agent.get_action(env.ytz, new_state)
            agent.update_q(26, str([0 for _ in range(13)]), new_action, new_state, reward)
            action = new_action
            state = new_state
            first_roll = False
        else:
            agent, new_state, reward, done, debug = env.step(action, agent)
            new_action = agent.get_action(env.ytz, new_state)
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
agent.last_avg_score = sum(agent.scores[-50:]) / 50
agent.write_data("./data/agentv4/envV3/2k.txt")
agent.generate_plot("./data/agentv4/envV3/2k.png")

agent.eps = 0
done = False
first_roll = True
while done is False:
    turns_left = env.ytz.remaining_turns
    # Player must roll all 5 dice on first roll of each turn
    if first_roll:
        agent, new_state, reward, done, debug = env.step(26, agent)
        new_action = agent.get_action(env.ytz, new_state)
        agent.update_q(26, str([0 for _ in range(13)]), new_action, new_state, reward)
        action = new_action
        state = new_state
        first_roll = False
    else:
        agent, new_state, reward, done, debug = env.step(action, agent)
        new_action = agent.get_action(env.ytz, new_state)
        agent.update_q(action, state, new_action, new_state, reward)
        # These two lines are irrelevant if the action is scoring b/c
        # they are reset on the new turn
        action = new_action
        state = new_state
        if turns_left > env.ytz.remaining_turns:
            first_roll = True