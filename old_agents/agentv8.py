"""V8 of the agent.

Major upgrades: Fix random action selection to only select from possible actions
"""
import random
from matplotlib import pyplot as plt
from utilv1 import make_action_space, make_state_space


class AgentV8:
    def __init__(self, eps, gma, alp):
        # Pregenerated list of possible dice counts given there are only 5 dice.
        """State space template:
        ([count of each dice value], bool indicating if rolls remain)
        """
        # These will be used as references for finding indices in Q table.
        self.state_space = make_state_space()
        self.action_space = make_action_space()
        self.q_table = [[0 for _ in range(505)] for _ in range(77)]
        self.eps = eps
        self.alp = alp
        self.gma = gma
        self.iterations = []
        self.scores = []
        self.games_played = 0
        self.max_score = 0
        self.avg_score = 0
        self.last_avg_score = 0

    def get_action(self, game, state, possible_actions):
        eps_cond = random.uniform(0, 1)
        # Explore
        if eps_cond <= self.eps and game.c_player.rolls_left > 0:
            action = random.randint(0, 75)
            while True:
                if possible_actions[action] != -1:
                    return action
                else:
                    action = random.randint(0, 75)
        # Force scoring if no rolls left
        elif eps_cond <= self.eps and game.c_player.rolls_left == 0:
            action = random.randint(0, 12)
            return action
        # Exploit
        # Find the index of each state at each action and 
        # use that to get the Q values, then choose max Q value as action.
        elif eps_cond > self.eps:
            values = []
            for i in range(75):
                # Make sure we aren't double scoring (this should be added to API too.)
                if possible_actions[i] != -1:
                    values.append(self.q_table[i][state])
                else:
                    values.append(-1)
            # Action is the index of the maximum Q value
            # Force scoring if no rolls left
            if game.c_player.rolls_left == 0:
                action = values[:13].index(max(values[:13])) if max(values[:13]) > 0 else random.randint(0, 12)
            else:
                action = values.index(max(values)) if max(values) > 0 else random.randint(0, 26)
        return action

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
            f.write("Games Played: " + str(self.games_played) + "\n")
            f.write("Gamma, Epsilon, Alpha: " + str(self.gma) + " " + str(self.eps) + " " + str(self.alp) + "\n")
            for i in range(77):
                f.write(str(self.q_table[i]) + "\n")
            f.write(str(self.scores))
        f.close()
