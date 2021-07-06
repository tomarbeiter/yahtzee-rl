from yahtzee_singleV7 import YahtzeeSinglePlayerV7
import math
from agentv7 import AgentV7
import time

env = YahtzeeSinglePlayerV7()
agent = AgentV7(1, 0.5, 0.1)

# Main algorithm
for i in range(1, 10001):
    agent.eps = math.exp(-0.00025 * i)
    if i >= 9900:
        agent.eps = 0.05
    done = False
    first_roll = True
    time1 = time.perf_counter()
    while done is False:
        turns_left = env.ytz.remaining_turns
        # Player must roll all 5 dice on first roll of each turn
        if first_roll:
            new_state, possible_actions, reward, done, debug = env.step(76, agent.action_space, agent.state_space)
            new_action = agent.get_action(env.ytz, new_state, possible_actions)
            agent.update_q(76, 504, new_action, new_state, reward)
            action = new_action
            state = new_state
            first_roll = False
        else:
            new_state, possible_actions, reward, done, debug = env.step(action, agent.action_space, agent.state_space)
            new_action = agent.get_action(env.ytz, new_state, possible_actions)
            # if it isn't scoring, update as usual
            if action >= 13:
                agent.update_q(action, state, new_action, new_state, reward)
                action = new_action
                state = new_state
            # If it is scoring, new_state and new_action are junk
            else:
                agent.update_q(action, state, 76, 504, reward)
            if turns_left > env.ytz.remaining_turns:
                first_roll = True
    time2 = time.perf_counter()
    print(time2 - time1)
    agent.scores.append(env.ytz.c_player.score)
    agent.iterations.append(i)
    env.reset()


# Data output
agent.max_score = max(agent.scores)
agent.games_played = agent.iterations[-1]
agent.avg_score = sum(agent.scores) / agent.games_played
agent.last_avg_score = sum(agent.scores[-100:]) / 100
agent.write_data("./data/agentv7/envV7/10kv6.txt")
agent.generate_plot("./data/agentv7/envV7/10kv6.png")

agent.eps = 0

done = False
first_roll = True
time1 = time.perf_counter()
while done is False:
    turns_left = env.ytz.remaining_turns
    # Player must roll all 5 dice on first roll of each turn
    if first_roll:
        new_state, possible_actions, reward, done, debug = env.step(76, agent.action_space, agent.state_space)
        new_action = agent.get_action(env.ytz, new_state, possible_actions)
        agent.update_q(76, 504, new_action, new_state, reward)
        action = new_action
        state = new_state
        first_roll = False
    else:
        new_state, possible_actions, reward, done, debug = env.step(action, agent.action_space, agent.state_space)
        new_action = agent.get_action(env.ytz, new_state, possible_actions)
        # if it isn't scoring, update as usual
        if action >= 13:
            agent.update_q(action, state, new_action, new_state, reward)
            action = new_action
            state = new_state
        # If it is scoring, new_state and new_action are junk
        else:
            agent.update_q(action, state, 76, 504, reward)
        if turns_left > env.ytz.remaining_turns:
            first_roll = True
time2 = time.perf_counter()
print(time2 - time1)

agent.scores.append(env.ytz.c_player.score)
agent.iterations.append(i)
env.reset()