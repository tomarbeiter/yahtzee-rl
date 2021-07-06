from yahtzee_singleV5 import YahtzeeSinglePlayerV5
import math
from agentv6 import AgentV6
import time

env = YahtzeeSinglePlayerV5()
agent = AgentV6(1, 0.99, 0.99)

# Main algorithm
for i in range(1, 10001):
    if i < 1000:
        agent.eps = 1
    else:
        # eps decay
        agent.eps = math.exp(-0.00025 * i + 0.25)
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
            agent.update_q(76, 516096, new_action, new_state, reward)
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
                agent.update_q(action, state, 76, 516096, reward)
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
agent.write_data("./data/agentv6/envV5/100kv1.txt")
agent.generate_plot("./data/agentv6/envV5/100kv1.png")

agent.eps = 0

done = False
first_roll = True
while done is False:
    turns_left = env.ytz.remaining_turns
    # Player must roll all 5 dice on first roll of each turn
    if first_roll:
        new_state, possible_actions, reward, done, debug = env.step(76, agent.action_space, agent.state_space)
        new_action = agent.get_action(env.ytz, new_state, possible_actions)
        agent.update_q(76, 946176, new_action, new_state, reward)
        action = new_action
        state = new_state
        first_roll = False
    else:
        new_state, possible_actions, reward, done, debug = env.step(agent.action_space[action], agent.action_space, agent.state_space)
        new_action = agent.get_action(env.ytz, new_state, possible_actions)
        # if it isn't scoring, update as usual
        if action >= 13:
            agent.update_q(action, state, new_action, new_state, reward)
            action = new_action
            state = new_state
        # If it is scoring, new_state and new_action are junk
        else:
            agent.update_q(action, state, 76,  516096, reward)
        if turns_left > env.ytz.remaining_turns:
            first_roll = True
agent.scores.append(env.ytz.c_player.score)
agent.iterations.append(i)
env.reset()