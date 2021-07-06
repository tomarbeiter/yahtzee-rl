from yahtzee_singleV4 import YahtzeeSinglePlayerV4
import math
from agentv5 import AgentV5


env = YahtzeeSinglePlayerV4()
agent = AgentV5(1, 0.99, 0.99)

# Main algorithm
for i in range(1, 201):
    # eps decay
    agent.eps = math.exp(-.025 * i)
    done = False
    first_roll = True
    while done is False:
        turns_left = env.ytz.remaining_turns
        # Player must roll all 5 dice on first roll of each turn
        if first_roll:
            new_state, reward, done, debug = env.step(26)
            new_action = agent.get_action(env.ytz, new_state)
            agent.update_q(26, 12, new_action, new_state, reward)
            action = new_action
            state = new_state
            first_roll = False
        else:
            new_state, reward, done, debug = env.step(action)
            new_action = agent.get_action(env.ytz, new_state)
            # if it isn't scoring, update as usual
            if action >= 13:
                agent.update_q(action, state, new_action, new_state, reward)
                action = new_action
                state = new_state
            # If it is scoring, new_state and new_action are junk
            else:
                agent.update_q(action, state, 26, 12, reward)
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
agent.write_data("./data/agentv5/envV4/200v3.txt")
agent.generate_plot("./data/agentv5/envV4/200v3.png")

agent.eps = 0

done = False
first_roll = True
while done is False:
    turns_left = env.ytz.remaining_turns
    # Player must roll all 5 dice on first roll of each turn
    if first_roll:
        new_state, reward, done, debug = env.step(26)
        new_action = agent.get_action(env.ytz, new_state)
        agent.update_q(26, 12, new_action, new_state, reward)
        action = new_action
        state = new_state
        first_roll = False
    else:
        new_state, reward, done, debug = env.step(action)
        new_action = agent.get_action(env.ytz, new_state)
        # if it isn't scoring, update as usual
        if action >= 13:
            agent.update_q(action, state, new_action, new_state, reward)
        # If it is scoring, new_state and new_action are junk
        else:
            agent.update_q(action, state, 26, 12, reward)
        # These two lines are irrelevant if the action is scoring b/c
        # they are reset on the new turn
        action = new_action
        state = new_state
        if turns_left > env.ytz.remaining_turns:
            first_roll = True