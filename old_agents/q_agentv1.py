"""v1 of the Q learning agent. 

Improvements from v0 include:
    - Compressed state space to account for duplicate states
      (Removed rolls_left from key and compressed duplicate dice)
    - Added median score to diagnostic data
"""

from matplotlib import pyplot as plt
import random
import copy

class QAgentV1:
    def __init__(self, gma, eps, alp):
        self.q_table = {}
        self.scores = []
        self.iterations = []
        self.games_played = 0
        self.max_score = 0
        self.avg_score = 0
        self.median = 0
        self.gma = gma
        self.eps = eps
        self.alp = alp

    def generate_plot(self, path):
        plt.plot(self.iterations, self.scores, marker=".", markersize=5)
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.savefig(path)

    def write_data(self, path, overwrite=True):
        with open(path, 'w' if overwrite else 'a') as f:
            f.write("Max Score: " + str(self.max_score) + "\n" )
            f.write("Average Score: " + str(self.avg_score) + "\n")
            f.write("Median Score: " + str(self.median) + "\n")
            f.write("Games Played: " + str(self.games_played) + "\n")
            f.write("Gamma, Epsilon, Alpha: " + str(self.gma) + " " + str(self.eps) + " " + str(self.alp) + "\n")
        f.close()

    def update_q(self, key, new_key, act, rwd):
        """Update Q value for previous action."""
        self.q_table[key][act] = self.q_table[key][act] + self.alp * (rwd + (self.gma * max(self.q_table[new_key])) - self.q_table[key][act])

    def make_key(self, game):
        # Remove rolls_left and dice_indices
        mod_t_scorecard = []
        for entry in game.c_player.t_scorecard:
            temp_entry = copy.deepcopy(entry)
            temp_entry.pop()
            temp_entry.pop()
            mod_t_scorecard.append(temp_entry)
        return str(mod_t_scorecard)
        
    def action_from_state(self, key, game):
        """Finds current state in Q-table.
        
        If the state is not in the Q-table, adds a new entry to the table
        and return the index of a random action. Otherwise, return the 
        most profitable action for that state.
        """

        if key not in self.q_table:
            self.q_table[key] = [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
        # Epsilon greedy explore/exploit
        eps_cond = random.uniform(0, 1)
        if eps_cond <= self.eps and game.c_player.rolls_left > 0:
            return random.randint(0, 25)
        # Force scoring if no rolls left
        elif eps_cond <= self.eps and game.c_player.rolls_left == 0:
            return random.randint(0, 12)
        elif eps_cond > self.eps and game.c_player.rolls_left > 0:
            return self.q_table[key].index(max(self.q_table[key]))
        # Force scoring if no rolls left
        elif eps_cond > self.eps and game.c_player.rolls_left == 0:
            return self.q_table[key].index(max(self.q_table[key][:13]))
        # TODO - Handle "equal states" when dice are just at different index