from yahtzee_api.game import Game
import ast
import copy
import numpy as np

class YahtzeeSinglePlayerV5:
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
            reward = self._make_reward(action_space[action], state_space[state], action_space)
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
            reward = self._make_reward(action_space[action], state_space[state], action_space)
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
        indices = [0, 0, 0, 0, 0]
        for i in range(5):
            if self.ytz.c_player.dice[i] in action:
                indices[i] == 1
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
            for j in range(len(dice_count)):
                try:
                    # For each action, see if "x's" are present
                    action_space[i].index(j + 1)
                except ValueError:
                    actions.append(-1)
                    check = False
                    break
            # If we never hit a value error, then all of the dice to keep (given by action) 
            # are present in the dice that were rolled.
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
        # Determine which scores have already been scored. "True" if it is 
        # "available to score", "False" if already scored.
        # 3/4k will be True unless both are scored. Same for Straights.
        scored = []
        for i in range(13):
            if self.ytz.c_player.scorecard[i][2] != 0:
                scored.append(False)
            else:
                scored.append(True)
        # Combine 3/4k and straights
        flag1 = True
        flag2 = True
        if not scored[6] and not scored[7]:
            flag1 = False
        if not scored[9] and not scored[10]:
            flag2 = False
        scored.pop(6)
        scored.pop(6)
        scored.pop(7)
        scored.pop(7)
        scored.insert(6, flag1) 
        scored.insert(8, flag2)
        scored.pop()

            
        state.append(scored)
        # Rolls remaining boolean
        if self.ytz.c_player.rolls_left == 0:
            state.append(False)
        else:
            state.append(True)
        state = str(tuple(state))
        return state_space.index(state)

    def _make_reward(self, action, state, action_space):
        """Generates rewards.
        
        Note that the values chosen for the top half scoring are meant
        to make it relaevant in comparison to the bottom half scores. 
        3 or more in each top half entry gets the bonus, so that is the line
        in the sand of high reward. Still want some reward for less than 3 of that type,
        because it is better than 0. Just don't over incentivize scoring 2 1's rather than going
        for a full house.
        """
        state = ast.literal_eval(state)
        # Reward for scoring the top half.
        # If they score it with more than 3 of the type, get 30. 
        # If only 1 or 2, get 15 and 10 respectively.
        if type(action) is int and action < 7:
            if state[0][action - 1] >= 3:
                return 30
            elif state[0][action - 1] == 2:
                return 15
            elif state[0][action - 1] == 1:
                return 10
            else:
                return 1
        # Reward for scoring the bottom half
        if type(action) is int and action >= 7 and action <= 13:
            return self.ytz.c_player.t_scorecard[action - 1][0] if \
                self.ytz.c_player.t_scorecard[action - 1][0] != 0 else 1
        # Rolling rewards
        # These are pretty specific on a case by case basis
        # Trying to group rolling actions together; 
        # i.e. choosing to keep a single type, 2 types, 3 types, etc.
        
        # Keeping 1 type:
        if len(action) == 1:
            # If the score card of the type of dice kept is unscored, or if Yahtzee is available, or a 3/4k, give reward.
            return 30 if state[1][action[0] - 1] == True or state[1][9] == True or state[1][6] == True else 1
        elif len(action) == 2:
            # if full house is available
            return 30 if state[1][7] else 1
        # Only real reason to keep 3 types of dice is to go for chance
        # but, our state compression got rid of chance. So, I'll just give
        # a small reward to not totally disincentivize it. Not sure how this will behave.
        elif len(action) == 3:
            return 20
        elif len(action) == 4:
            # Straights
            return 30 if state[1][8] else 1
        elif len(action) > 4:
            # Keeping one of each that we have, so clearly pursuing a straight
            return 30 if state[1][8] else 1