from yahtzee_api.game import Game
import ast

class YahtzeeSinglePlayerV4:
    """Built to work with compressed state space of agentV5."""
    def __init__(self):
        self.ytz = Game(1)      # Initialize 1-player game of Yahtzee

    def step(self, action):
        """Executes a single timestep in the environment.

        Args:
            action (int): specifies the action to take.
                Actions 0-12 score the corresponding index
                of the Player's scorecard. Actions 13-25 
                reroll the dice to pursue the score at 
                index action - 13. For example, taking action
                13 would pass the list of dice indices from
                the theoretical scorecard for the 1's item 
                to the roll function. In this way, the agent 
                can "freeze" the 1's it rolled and try to get more,
                while still being free to roll any of the dice it 
                chooses on subsequent re-rolls. Action 26 rolls all dice.
                See the Yahtzee API docs at
                https://yahtzee-api.tomarbeiter.com for more info
                about how the scorecard, dice rolling, etc. works.
        Returns:
            Tuple: The entire instance of the agent, the state,
            reward, boolean done indicator, and basic debugging info.
        """

        # Roll dice to pursue certain score
        if action > 12 and action < 26:
            reward = self._make_reward(action)
            self.ytz.c_player.roll(self.ytz.c_player.t_scorecard[action - 13][1])
            return (self._id_state(), reward, False, "Dice were rolled.")
        if action == 26:
            self.ytz.c_player.roll([0, 0, 0, 0, 0])
            return (self._id_state(), 0, False, "Dice were rolled.")
        # Score
        else:
            reward = self._make_reward(action)
            self.ytz.c_player.end_turn(action)
            # One player game, so always advance global turn
            self.ytz.next_player()
            # If the game is over, set Done flag
            return (
                self._id_state(),
                reward,
                len(self.ytz.winner) != 0,
                str(action) + " was scored."
            )

    def reset(self):
        """Starts a new game of Yahtzee in the same class instance."""
        self.ytz = Game(1)

    def _id_state(self):
        """Determines the current state of the game."""
        state = []
        for i in range(13):
            # Check if entry has not already been scored
            if self.ytz.c_player.scorecard[i][2] == 0:
                state.append(self.ytz.c_player.t_scorecard[i][1].count(1))
            # Fill in a -1 if item is already scored    
            else:
                state.append(-1)
        # Map the detailed state to its appropriate compressed value
        # Based on the number of 1's, 2's, 3's, 4's, 5's and 6's. 
        # Default compressed state is the state where top half is fully scored
        compressed_state = 12
        # If all of the top half isn't scored yet and the max has at least 3 and no rolls left
        # Rolls_left is a factor because it gives more detail for reward function
        # If we have 1 or more and rolls left, ideally agent keeps rolling - if 3 or more and no rolls, learn to score.
        # This state compression strategy gives that level of detail to allow for that learning.
        if max(state[:6]) != -1 and max(state[:6]) >= 3 and self.ytz.c_player.rolls_left == 0:
            compressed_state = state.index(max(state[:6]))
        elif max(state[:6]) != -1 and max(state[:6]) >= 1 and self.ytz.c_player.rolls_left != 0:
            compressed_state = state.index(max(state[:6])) + 6
        return compressed_state


    def _make_reward(self, action):
        """Generates rewards."""
        # Reward for scoring
        if action >= 0 and action <= 5:
            return 30 if action == self._id_state() else 1
            # possible_scores = []
            # for i in range(6):
            #     # Normalizes the values so 1's don't get washed out, and adds 30 to make them relevant and suggest the bonus
            #     possible_scores.append(self.ytz.c_player.t_scorecard[i][0]/ (i + 1) + 30)
            # max_score = max(possible_scores)
            # return max_score if self.ytz.c_player.t_scorecard[action][0] == max_score and action < 6 else 0
        
        # Reward for rolling
        # Reward choosing the action the corresponds with rolling to purse the class of state it is in.
        elif action >= 13 and action <= 18:
            return 30 if action == self._id_state() + 7 else 1
        else:
            return 1