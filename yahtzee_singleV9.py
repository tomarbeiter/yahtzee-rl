from yahtzee_api.game import Game
import ast
import copy

class YahtzeeSinglePlayerV9:
    """Built to work with compressed state space of agentV9.
    
    Wokring on the reward function at this point. 
    """
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
            count = 0
            for j in action_space[i]:
                count += dice_count[j - 1]
                # Check if action is keeping all 5 dice, in which case it isn't actually 
                # a rolling move and shouldn't be allowed.
                if count == 5:
                    actions.append(-1)
                    break
                # Check if it is trying to hold dice that it doesn't have.
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
        # Reward for scoring the top half.
        # If they score it with more than 3 of the type, get 50. 
        if type(action) is int and action < 6:
            if state[0][action] >= 3 and state[1] == False:
                return 50
            elif state[0][action] == 2 and state[1] == False:
                return 10
            else:
                return 1
        # Reward for scoring 3 of a kind
        if type(action) is int and action == 6:
            return 50 if 3 in state[0] else 1
        
        # Reward for scoring 4 of a kind
        if type(action) is int and action == 7:
            return 50 if 4 in state[0] else 1

        # Reward for scoring Full House
        if type(action) is int and action == 8:
            if 2 in state[0] and 3 in state[0]:
                return 50
            else:
                return 1
        
        # Reward for scoring Small Straight
        if type(action) is int and action == 9:
            if state[0].count(1) == 4:
                return 50
            else:
                return 1

        # Reward for scoring Large Straight
        if type(action) is int and action == 10:
            if state[0].count(1) == 5:
                return 50
            else:
                return 1

        # Reward for scoring Yahtzee
        if type(action) is int and action == 11:
            return 100 if 5 in state[0] else 1
        
        # Reward for scoring Chance
        if type(action) is int and action == 12:
            score = 0
            for i in range(len(state[0])):
                score += state[0][i] * i + 1
            return 50 if score > 20 else 1

        # Penalties for missing scores
        # To the extent that this can be ascertained from the state
        # Missed Full House
        if type(action) is not int or action != 8:
            # Bit of cleverness, if there are 4 zeros that means there are only 2 types of dice
            # If that is true and no type of dice has only 1, then it must be a full house.
            if state[0].count(0) == 4 and state[0].count(1) == 0:
                return 1
        
        # Missed Small Straight
        if type(action) is not int or action != 9:
            if state[0][:4].count(0) == 0 or state[0][1:5].count(0) == 0 \
                or state[0][2:6].count(0) == 0:
                return 1
        # Missed Large Straight
        if type(action) is not int or action != 10:
            if state[0][:5].count(0) == 0 or state[0][1:6].count(0) == 0:
                return 1

        # Missed Yahtzee
        if type(action) is list and len(action) == 1 and state[0][action[0] - 1] == 5:
            return 1

        # Rolling rewards
        # Reward for keeping single dice corresponds to potential reward
        # for optimal score on top half (or yahtzee)
        if type(action) is list and len(action) == 1 and \
            state[0][action[0] - 1] > 1:
            return 50

        # Keeping two dice suggests full house
        if type(action) is list and len(action) == 2:
            # If there is a two pair, keep going for full house
            for x in action:
                if state[0][x - 1] < 2:
                    return 1
            return 50

        # Keeping three or 4 types of dice should only be done for a straight
        # i.e., incentivize keeping them if there is one of each.
        if type(action) is list and (len(action) == 3 or len(action) == 4):
            for i in range(len(state[0])):
                if state[0][i] > 1:
                    return 1
            return 35
        return 1