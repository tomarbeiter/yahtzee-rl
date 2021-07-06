from yahtzee_api.game import Game
import ast

class YahtzeeSinglePlayerV3:
    """Very similar to V2, except it employs a new reward system to integrate
    with the agentV4 redesigned state space.
    """
    def __init__(self):
        """Constructor that parameterizes the rewards for rolling.

        In general, if the agent scores a value it receives that score
        as reward. When it decides to roll some or all of the dice,
        it also should receive some reward. However, it is not clear
        what the best values for that reward is, so for research
        purposes it is made configurable here. 

        """
        self.ytz = Game(1)      # Initialize 1-player game of Yahtzee
        self.state = ""

    def step(self, action, agent):
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
            self.ytz.c_player.roll(self.ytz.c_player.t_scorecard[action - 13][1])
            self._id_state()
            agent = self._insert_state(self.state, agent)
            return (agent, self.state, self._make_reward(action), False, "Dice were rolled.")
        if action == 26:
            self.ytz.c_player.roll([0, 0, 0, 0, 0])
            self._id_state()
            agent = self._insert_state(self.state, agent)
            return (agent, self.state, 0, False, "Dice were rolled.")
        # Score
        else:
            self._id_state()
            agent = self._insert_state(self.state, agent)
            self.ytz.c_player.end_turn(action)
            # One player game, so always advance global turn
            self.ytz.next_player()
            # If the game is over, set Done flag
            return (
                agent,
                self.state,
                self._make_reward(action),
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
        # Save it as a string so lookups in the ref table are faster
        self.state = str(state)

    def _insert_state(self, state, agent):
        """Checks if state is in the q table yet. 
        If not, inserts it at all possible actions for that state."""
        for i in range(13, 27):
            try:
                # If the state is found in the rolling half of the table, then stop searching.
                if agent.q_ref_table[i].index(state) >= 0:
                    return agent
            except ValueError:
                # Means that state was not found for the given action
                # Need to let the whole loop finish in case it is at another action
                continue
        
        # If the loop exits without returning, state was not found, so insert it:
        for i in range(13):
            # New states always go into the scoring actions no matter what
            if state[i] != -1:
                agent.q_table[i].append(0)
                agent.q_ref_table[i].append(state)
        
        # If there are rolls left, action also goes into corresponding rolling action
        if self.ytz.c_player.rolls_left > 0:
            for i in range(13, 27):
                if state[i] != -1:
                    agent.q_table[i].append(0)
                    agent.q_ref_table[i].append(state)
        return agent

    def _make_reward(self, action):
        """Generates rewards."""
        # Reward for scoring
        if action >= 0 and action <= 12:
            possible_scores = []
            for i in range(13):
                if i < 6:
                    # Normalizes the values so 1's don't get washed out, and adds 30 to make them relevant and suggest the bonus
                    possible_scores.append(self.ytz.c_player.t_scorecard[i][0]/ (i + 1) + 30)
                else:
                    possible_scores.append(self.ytz.c_player.t_scorecard[i][0])
            max_score = max(possible_scores)
            return max_score if self.ytz.c_player.t_scorecard[action][0] == max_score else 0
        
        # Reward for rolling
        # Reward choosing the path with the most amount of 1's (or equal to the most amount of 1's)
        else:
            state = ast.literal_eval(self.state)
            action_ones = self.ytz.c_player.t_scorecard[action - 13][1].count(1)
            return 30 if action_ones == max(state) else 0
        