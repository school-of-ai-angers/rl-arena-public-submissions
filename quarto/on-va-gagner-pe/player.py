# An example of a valid submission to the arena
# See more instructions in the Notebook `Train your player.ipynb`

import numpy as np
import pickle
from copy import deepcopy

rows = [np.arange(4) + 4*l for l in range(4)]
columns = [np.arange(16, step=4) + l for l in range(4)]
diagonals = [np.array([0, 5, 10, 15]), np.array([3, 6, 9, 12])]

lines = rows + columns + diagonals


def has_common_trait(pieces):
    assert len(pieces) == 4
    p1, p2, p3, p4 = pieces
    return p1 != -1 and p2 != -1 and p3 != -1 and p4 != -1 and ((p1 & p2 & p3 & p4) or ((~p1 & ~p2 & ~p3 & ~p4) & 0xF))


def get_board_status(board_state):
    """
    check board status.
    : param board_state: np array
    Return 100 is the board as a winning line
    Return 0 in case of draw
    """
    for line in lines:
        if has_common_trait(board_state[line]):
            return 100
    return 0


def play_action(state, action):
    new_state = state.copy()
    action_pos = action // 16
    action_piece = action % 16

    # Put the piece on the board
    new_state[action_pos] = new_state[16]

    # Select the next piece for the opponent
    new_state[16] = action_piece
    return new_state


def get_valid_actions(state):
    valid_actions = []

    available_pieces = set(range(16))
    for pos in range(16):
        piece = state[pos]
        if piece != -1:
            available_pieces.remove(piece)

    for pos in range(16):
        if state[pos] != -1:
            continue
        for piece in available_pieces:
            valid_actions.append(16 * pos + piece)
    return valid_actions


class QLearningPlayer:
    def __init__(self, train_mode):
        # Whether we are in training or playing mode
        # In training mode, this player will update its Q-table
        # and sometimes take a random action to explore more
        self.train_mode = train_mode

        # This agent's Q-table.
        # It is a map from state to action value pre action:
        # q_table[state][action]: float
        self.q_table = {}
        self.action_space = 256

        # Epsilon scheduling
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.99995

        # Q-table update hyperparameters
        self.alpha = 0.1
        self.gamma = 1

        # Q-table update helper variables
        self.prev_state = None
        self.prev_action = None

    def start(self, state, valid_actions):
        # First move: take the action
        return self._take_action(state, valid_actions)

    def step(self, state, valid_actions, reward):
        # Filter out actions for which we lose
        non_losing_actions = []
        for action in valid_actions:
            # If we win, we take it
            new_state = play_action(state, action)
            if get_board_status(new_state) == 100:
                return action

            # Check if the opponent can win
            we_can_lose = False
            for opp_action in get_valid_actions(new_state):
                new_new_state = play_action(new_state, opp_action)
                if get_board_status(new_new_state) == 100:
                    we_can_lose = True
                    break

            if not we_can_lose:
                non_losing_actions.append(action)

        if len(non_losing_actions) > 0:
            valid_actions = non_losing_actions

        # At every other step: update the q-table and take the next action
        # Since the game hasn't finished yet, we can use current knowledge of the q-table
        # to estimate the future reward.
        if self.train_mode:
            action_values = self._get_action_values(state, valid_actions)
            self._update_q_table(reward + self.gamma * np.max(action_values))
        return self._take_action(state, valid_actions)

    def end(self, state, reward):
        # Last step: update the q-table and schedule the next value for epsilon
        # Here, the expected action-value is simply the final reward
        if self.train_mode:
            self._update_q_table(reward)
            self.epsilon = max(
                self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _update_q_table(self, new_value):
        # Based on the reward of the previous action take at the previous step,
        # update the q-table to be closer to the desired value.
        # Note that, if `alpha` is zero, the q-table is left unchanged,
        # if `alpha` is one, the q-table will simply take the `new_value`.
        # With a value in between, one can control the tradeoff between learning too much
        # or too little from a single move
        prev_state = tuple(self.prev_state)
        q_row = self.q_table.setdefault(
            prev_state, np.zeros(self.action_space))
        q_row[self.prev_action] += self.alpha * \
            (new_value - q_row[self.prev_action])

    def _take_action(self, state, valid_actions):
        # Store the current state, copying it, otherwise the environment could mutate it afterwards
        self.prev_state = state.copy()

        if self.train_mode and np.random.random() <= self.epsilon:
            # Take a random action
            self.prev_action = np.random.choice(valid_actions)
        else:
            # Take the action that has the highest expected future reward,
            # that is, the highest action value
            action_values = self._get_action_values(state, valid_actions)
            self.prev_action = valid_actions[np.argmax(action_values)]
        return self.prev_action

    def _get_action_values(self, state, valid_actions):
        # Convert from numpy array to tuple
        state = tuple(state)
        if self.train_mode:
            # Return saved action values. If this is the first time this state is visited,
            # set all values to zero
            return self.q_table.setdefault(state, np.zeros(self.action_space))[valid_actions]
        # When not in train mode, do not change the Q-table, just return a new default for
        # every new never-visited state
        return self.q_table.get(state, np.zeros(self.action_space))[valid_actions]

    def get_freezed(self):
        # Return a copy of the player, but not in train_mode
        # This is used by the training loop, to replace the adversary from time to time
        copy = deepcopy(self)
        copy.train_mode = False
        return copy

    def save(self):
        # Save the q-table on the disk for future use
        with open('player.bin', 'wb') as fp:
            pickle.dump(dict(self.q_table), fp, protocol=4)


class Player(QLearningPlayer):
    def __init__(self, train_mode):
        super().__init__(train_mode)
        with open('player.bin', 'rb') as fp:
            self.q_table = pickle.load(fp)
