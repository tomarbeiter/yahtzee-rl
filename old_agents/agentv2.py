"""V2 of the q_agent.

Major upgrades: Designed to make use of the API v1.3.0 updated 
                recommendation system. Previous agents do not 
                work at all and should not be used. The action-
                state space for this agent is dramatically 
                compressed (with some tradeoffs, detailed elsewhere).
                Uses environment V1, with upgraded reward system.
"""
import random
from matplotlib import pyplot as plt

class QAgentV2:
    def __init__(self, eps, gma, alp):
        # Row in the q-table with all possible configs of dice indices to reroll
        self.q_row = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0], [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0], [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]

        # Will be used as a lookup table to find the index of the actual q_table to update reward
        self.q_ref_table = [self.q_row for _ in range(27)]
        # Initialize 27x32 table of 0's
        self.q_table = [[0 for _ in range(32)] for _ in range(27)]
        self.eps = eps
        self.alp = alp
        self.gma = gma
        self.iterations = []
        self.scores = []
        self.games_played = 0
        self.max_score = 0
        self.avg_score = 0
        self.median = 0
        self.last_avg_score = 0
        
    def get_action(self, game):
        eps_cond = random.uniform(0, 1)
        # Explore
        if eps_cond <= self.eps and game.c_player.rolls_left > 0:
            return random.randint(0, 26)
        # Force scoring if no rolls left
        elif eps_cond <= self.eps and game.c_player.rolls_left == 0:
            return random.randint(0, 12)
        # Exploit
        # Interpret the larger "state" from analyzing the q_table at each action
        elif eps_cond > self.eps:
            max_values = []
            for i in range(26):
                # Don't consider actions that have already been scored (not allowed)
                check = i - 13 if i >= 13 else i
                if game.c_player.scorecard[check][2] != 0:
                    continue
                max_values.append(self.q_table[i][self.q_ref_table[i].index(game.c_player.t_scorecard[i - 13 if i >= 13 else i][1])])
            # If no rolls left, score the highest
            if game.c_player.rolls_left == 0:
                return max_values.index(max(max_values)) if max_values.index(max(max_values)) < 13 else max_values.index(max(max_values)) - 13
            else:
                return max_values.index(max(max_values)) if max(max_values) > 0 else random.randint(0, 26)

    def update_q(self, action, new_action, indices, rwd):
        """Update Q value for previous action."""
        self.q_table[action][self.q_ref_table[action].index(indices)] = self.q_table[action][self.q_ref_table[action].index(indices)] + self.alp * \
            (rwd + (self.gma * (max(self.q_table[new_action])) - self.q_table[action][self.q_ref_table[action].index(indices)]))

    def generate_plot(self, path):
        plt.plot(self.iterations, self.scores, marker=".", markersize=5)
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.savefig(path)

    def write_data(self, path, overwrite=True):
        with open(path, 'w' if overwrite else 'a') as f:
            f.write("Max Score: " + str(self.max_score) + "\n" )
            f.write("Average Score: " + str(self.avg_score) + "\n")
            f.write("Average Score last 100: " + str(self.last_avg_score) + "\n")
            f.write("Median Score: " + str(self.median) + "\n")
            f.write("Games Played: " + str(self.games_played) + "\n")
            f.write("Gamma, Epsilon, Alpha: " + str(self.gma) + " " + str(self.eps) + " " + str(self.alp) + "\n")
            for i in range(27):
                f.write(str(self.q_table[i]) + "\n")
            f.write(str(self.scores))
        f.close()
