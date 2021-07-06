import itertools as it
import copy
import numpy as np

def make_state_space():
    """Quick and dirty way to generate the state space reference list for v6 agent.
    
    Total states: 505
    """
    a = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                for l in range(6):
                    for m in range(6):
                        for n in range(6):
                            a.append([i, j, k, l, m, n])
                            
    new_a = []
    for x in a:
        if sum(x) == 5:
            new_a.append(x)

    no_rolls = []
    for i in range(len(new_a)):
            no_rolls.append(str((new_a[i], False)))

    rolls = []
    for i in range(len(new_a)):
            rolls.append(str((new_a[i], True)))

    final_states = copy.deepcopy(no_rolls + rolls)
    # Catch all state for rolling on first turn
    final_states.append(0)
    return final_states


def make_action_space():
    """Quick and drity action space template generator for v6.
    
    Total actions: 77
    """

    # First 13 states are scoring states, everything else is rolling.
    action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    for i in range(1, 7):
        action_space += list(it.combinations([1, 2, 3, 4, 5, 6], i))
    for i in range(13, len(action_space)):
        action_space[i] = list(action_space[i])
    
    # can't have all 6 values at once so throw out the last element
    action_space.pop()
    # action to keep 1 of everything that we have
    action_space.append([1, 2, 3, 4, 5, 6])
    # action to roll everything
    action_space.append([0])
    return action_space