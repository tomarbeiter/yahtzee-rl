"""V4 of the agent.

Major upgrades: Completely redesigned state space
to allow for greater detail in each state.

State will be a list of the number of 1's in the dice indices
for each action that is possible from a given turn (i.e., if
a score is already scored it will be 0'd out in the state.) 
State space is built dynamically as states are discovered because 
it is massive. Will likely require a high number of games to encounter enough
states for learning. 
"""
import random
from matplotlib import pyplot as plt

class QAgentV4:
    def __init__(self, eps, gma, alp):
        
        # Will be used as a lookup table to find the index of the actual q_table to update reward
        # Initialized as an empty row for each action, states will be added as they are discovered
        self.q_ref_table = [[] for _ in range(27)]
        self.q_ref_table[26].append(str([0 for _ in range(13)]))
        # Initialize 27x32 table of 0's
        self.q_table = [[] for _ in range(27)]
        # Just need a blank state to get the game started
        self.q_table[26].append(0)
        self.eps = eps
        self.alp = alp
        self.gma = gma
        self.iterations = []
        self.scores = []
        self.games_played = 0
        self.max_score = 0
        self.avg_score = 0
        self.last_avg_score = 0

    def get_action(self, game, state):
        eps_cond = random.uniform(0, 1)
        # Explore
        if eps_cond <= self.eps and game.c_player.rolls_left > 0:
            action = random.randint(0, 26)
            return action
        # Force scoring if no rolls left
        elif eps_cond <= self.eps and game.c_player.rolls_left == 0:
            action = random.randint(0, 12)
            return action
        # Exploit
        # Find the index of each state at each action and 
        # use that to get the Q values, then choose max Q value as action.
        elif eps_cond > self.eps:
            values = []
            for i in range(27):
                try:
                    # Find the index of the state at each action and get the corresponding Q value
                    values.append(self.q_table[i][self.q_ref_table[i].index(state)])
                except ValueError:
                    # If state is not found at an action, put a -1 in its place
                    values.append(-1)
            # Action is the index of the maximum Q value
            if game.c_player.rolls_left == 0:
                action = values[:13].index(max(values[:13])) if max(values[:13]) > 0 else random.randint(0, 12)
            else:
                action = values.index(max(values)) if max(values) > 0 else random.randint(0, 26)
        return action

    def update_q(self, action, state, new_action, new_state, rwd):
        """Update Q value for previous action."""
        if action > 12:
            state = self.q_ref_table[action].index(state)
            new_state = self.q_ref_table[new_action].index(new_state)
            self.q_table[action][state] = self.q_table[action][state] + \
                (self.alp * (rwd + (self.gma * (self.q_table[new_action][new_state] - self.q_table[action][state]))))
        # If the agent is scoring the new state is just the arbitrary starting state, so ignore it.
        else:
            state = self.q_ref_table[action].index(state)
            self.q_table[action][state] = self.q_table[action][state] + \
                (self.alp * (rwd + (self.gma * self.q_table[action][state])))

    def generate_plot(self, path):
        plt.plot(self.iterations, self.scores, marker=".", markersize=5)
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.savefig(path)

    def write_data(self, path, overwrite=True):
        with open(path, 'w' if overwrite else 'a') as f:
            f.write("Max Score: " + str(self.max_score) + "\n" )
            f.write("Average Score: " + str(self.avg_score) + "\n")
            f.write("Average Score last 50: " + str(self.last_avg_score) + "\n")
            f.write("Games Played: " + str(self.games_played) + "\n")
            f.write("Gamma, Epsilon, Alpha: " + str(self.gma) + " " + str(self.eps) + " " + str(self.alp) + "\n")
            for i in range(27):
                f.write(str(self.q_table[i]) + "\n")
            f.write(str(self.scores))
        f.close()
