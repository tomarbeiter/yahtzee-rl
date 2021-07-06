"""V3 of the agent.

Major upgrades: SARSA update function, eps-greedy

State is the maximum possible q value chosen from the 
current theoretical scorecard given dice configuration.
"""
import random
from matplotlib import pyplot as plt

class QAgentV3:
    def __init__(self, eps, gma, alp):
        # Row in the q-table with all possible configs of dice indices to reroll
        self.q_row = [0, 1, 2, 3, 4, 5]

        # Will be used as a lookup table to find the index of the actual q_table to update reward
        self.q_ref_table = [self.q_row for _ in range(27)]
        # Initialize 27x32 table of 0's
        self.q_table = [[0 for _ in range(6)] for _ in range(27)]
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
        
    def get_action_and_state(self, game):
        eps_cond = random.uniform(0, 1)
        # Pinpoint the state
        max_state = []
        for i in range(26):
            # Don't consider states for actions that have already been scored (not allowed)
            check = i - 13 if i >= 13 else i
            if game.c_player.scorecard[check][2] != 0:
                max_state.append(0)
                continue
            max_state.append(self.q_table[i][self.q_ref_table[i].index(game.c_player.t_scorecard[i - 13 if i >= 13 else i][1].count(1))])
        max_state_val = max(max_state)
        start = 0
        state = -1
        # State is horizontal index corresponding to the max q value from the possible scores given the dice config
        for i in range(26):
            try:
                state = self.q_table[max_state.index(max_state_val, start)].index(max_state_val)
                break
            except ValueError:
                start = i

        # Determine action
        # Explore
        if eps_cond <= self.eps and game.c_player.rolls_left > 0:
            action = random.randint(0, 26)
            return (action, state)
        # Force scoring if no rolls left
        elif eps_cond <= self.eps and game.c_player.rolls_left == 0:
            action = random.randint(0, 12)
            return (action, state)
        # Exploit
        elif eps_cond > self.eps:
            # If no rolls left, score the highest (eps-greedy)
            if game.c_player.rolls_left == 0:
                action = max_state.index(max(max_state)) if max_state.index(max(max_state)) < 13 else max_state.index(max(max_state)) - 13
                return (action, state)
            else:
                action = max_state.index(max(max_state)) if max(max_state) > 0 else random.randint(0, 26)
                return (action, state)

    def update_q(self, action, state, new_action, new_state, rwd):
        """Update Q value for previous action."""
        self.q_table[action][state] = self.q_table[action][state] + \
            (self.alp * (rwd + (self.gma * (self.q_table[new_action][new_state] - self.q_table[action][state]))))

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
