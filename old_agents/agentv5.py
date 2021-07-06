"""V5 of the agent.

Major upgrades: Compressed (but not too much) the state space.
Really only provides detail for top half of card. Goal is to get the agent
to play that top half really well, and then try to extend to the rest 
in a future agent. States are now "classified" based on their characteristics 
into similar groups, as a goldilocks between V4 and V3.
"""
import random
from matplotlib import pyplot as plt

class AgentV5:
    def __init__(self, eps, gma, alp):
        """Learning table template (each row corresponds to one of the 27 actions):
        The states in each row:
        "scoring" states (no rolls left):                           rolling states:
        [1's >= 3, 2's >= 3, 3's >= 3, 4's >= 3, 5's >= 3, 6's >=3, 1's >= 1, 2's >= 1, 3's >= 1, 4's >= 1, 5's >= 1, 6's >= 1, "other" state]

        Each action has this row of states. Idea being that agent will learn what not to do when in that class of state via reward function.
        """
        # Each row has 13 states as outlined above
        self.q_table = [[0 for _ in range(13)] for _ in range(27)]
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
                # Make sure we aren't double scoring (this should be added to API too.)
                if i < 13:
                    if game.c_player.t_scorecard[i][0] != 0:
                        values.append(self.q_table[i][state])
                    else:
                        values.append(-1)
                else:
                    values.append(self.q_table[i][state])
            # Action is the index of the maximum Q value
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
            f.write("Average Score last 50: " + str(self.last_avg_score) + "\n")
            f.write("Games Played: " + str(self.games_played) + "\n")
            f.write("Gamma, Epsilon, Alpha: " + str(self.gma) + " " + str(self.eps) + " " + str(self.alp) + "\n")
            for i in range(27):
                f.write(str(self.q_table[i]) + "\n")
            f.write(str(self.scores))
        f.close()
