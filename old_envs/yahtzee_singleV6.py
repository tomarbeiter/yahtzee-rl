from yahtzee_api.game import Game
import ast
import copy

class YahtzeeSinglePlayerV6:
    """Built to work with compressed state space of agentV6."""
    def __init__(self):
        self.ytz = Game(1)      # Initialize 1-player game of Yahtzee

    def step(self, action, action_space, state_space):
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
            action_space (list): reference list of the possible actions.
        Returns:
            Tuple: State (str), possible actions (list), reward (int), done flag (bool), debug (str)
        """

        # Roll dice to pursue certain score
        if action > 12 and action < 76:
            # Reward is based on state before dice are rolled
            state = self._id_state(state_space)
            reward = self._make_reward(action_space[action], state_space[state])
            dice_to_roll = self._translate_action(action_space[action])
            self.ytz.c_player.roll(dice_to_roll)
            new_state = self._id_state(state_space)
            actions = self._id_actions(state_space[new_state], action_space)
            return (new_state, actions, reward, False, "Dice were rolled.")
        if action == 76:
            self.ytz.c_player.roll([0, 0, 0, 0, 0])
            state = self._id_state(state_space)
            actions = self._id_actions(state_space[state], action_space)
            return (state, actions, 0, False, "Dice were rolled.")
        # Score
        else:
            state = self._id_state(state_space)
            reward = self._make_reward(action_space[action], state_space[state])
            self.ytz.c_player.end_turn(action)
            new_state = self._id_state(state_space)
            actions = self._id_actions(state_space[new_state], action_space)
            # One player game, so always advance global turn
            self.ytz.next_player()
            # If the game is over, set Done flag
            return (
                new_state,
                actions,
                reward,
                len(self.ytz.winner) != 0,
                str(action) + " was scored."
            )

    def reset(self):
        """Starts a new game of Yahtzee in the same class instance."""
        self.ytz = Game(1)

    def _translate_action(self, action):
        """Translates the chosen action in step() into dice indices for the API to roll."""   
        indices = []
        for i in range(5):
            if self.ytz.c_player.dice[i] in action:
                indices.append(1)
            else:
                indices.append(0)
        return indices

    def _id_actions(self, state, action_space):
        """Identify feasible actions (actions that are actually possible) from the given state."""
        st = ast.literal_eval(state)
        actions = []
        # First get scoring possibilities based on what has already been scored:
        # Little bit a of a hack since the state score list is compressed, I'm going to pick from the actual scorecard
        # ~technically~ it's the same thing
        for i in range(1, 14):
            if self.ytz.c_player.scorecard[i - 1][2] == 0:
                actions.append(i)
            else:
                actions.append(-1)

        # Add any rolling actions that hold dice that have actually been rolled
        dice_count = st[0]
        for i in range(13, 76):
            check = True
            for j in action_space[i]:
                if dice_count[j - 1] == 0:
                    actions.append(-1)
                    check = False
                    break
            if check:
                actions.append(action_space[i])
        return actions

    def _id_state(self, state_space):
        """Determines the current state of the game.
        
        Compressing 3/4k, Straights, and removing chance.
        This will make it much harder to chooose between them
        when scoring for the agent, but it reduces the state space
        from over 8 million to under 1 million. Tradeoffs.
        """
        state = []
        # Count how many of each die are currently rolled.
        # 1's -> 6's
        dice_count = copy.deepcopy([0, 0, 0, 0, 0, 0])
        for i in range(5):
            dice_count[self.ytz.c_player.dice[i] - 1] += 1
        state.append(dice_count)

        # Rolls remaining boolean
        if self.ytz.c_player.rolls_left == 0:
            state.append(False)
        else:
            state.append(True)
        state = str(tuple(state))
        return state_space.index(state)

    def _make_reward(self, action, state):
        """Generates rewards.
        
        Note that the values chosen for the top half scoring are meant
        to make it relevant in comparison to the bottom half scores. 
        """
        state = ast.literal_eval(state)
        # Reward for scoring Yahtzee
        if type(action) is int and 5 in state[0]:
            return 50 if action == 11 else 1

        # Reward for scoring the top half.
        # If they score it with more than 3 of the type, get 40. 
        if type(action) is int and action < 6:
            if state[0][action] >= 3:
                return 40
            elif state[0][action] == 2:
                return 10
            else:
                return 1

        # Reward for scoring Full House
        if type(action) is int and action == 8:
            if 2 in state[0] and 3 in state[0]:
                return 50
            else:
                return 1

        # Reward for scoring the bottom half
        if type(action) is int and action >= 6 and action <= 10 or action == 12:
            return self.ytz.c_player.t_scorecard[action][0] if \
                self.ytz.c_player.t_scorecard[action][0] != 0 else 1


        # Rolling rewards
        # Reward for keeping single dice corresponds to potential reward
        # for optimal score on top half (or yahtzee)
        if type(action) is list and len(action) == 1:
            return 50
        # Bad reward if it rolls when it already has 5
        if type(action) is list and len(action) == 1 and state[0][action] == 5:
            return 1
        # Keeping two dice suggests full house
        if type(action) is list and len(action) == 2:
            return 25
        # Keeping three or 4 types of dice should only be done for a straight
        # i.e., incentivize keeping them if there is one of each.
        if type(action) is list and (len(action) == 3 or len(action) == 4):
            for i in range(len(state[0])):
                if state[0][i] > 1:
                    return 1
            return 35
        return 1