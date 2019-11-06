import numpy as np
from connect_four import Environment
from copy import deepcopy

DEPTH = 2


class Player:
    """ Use minimax as described in https://en.wikipedia.org/wiki/Minimax """

    def __init__(self, train_mode):
        """
        :param train_mode: bool
        """
        self.train_mode = train_mode

    def start(self, state, valid_actions):
        """
        :param state: np.array
        :param valid_actions: np.array 1D
        :returns: float
        """
        return self._minimax(state, valid_actions)

    def step(self, state, valid_actions, prev_reward):
        """
        :param state: np.array
        :param valid_actions: np.array 1D
        :param prev_reward: float
        :returns: float
        """
        return self._minimax(state, valid_actions)

    def end(self, state, prev_reward):
        """
        :param state: np.array
        :param prev_reward: float
        """
        pass

    def _minimax(self, state, valid_actions):
        return self._minimax_level(state, 0, False, valid_actions, DEPTH, True)[1]

    def _minimax_level(self, state, total_reward, done, valid_actions, depth, maximize):
        if depth == 0 or done:
            # Heuristic: assume future rewards of zero
            return total_reward, None
        best_action = None
        best_value = None
        if maximize:
            # This player's turn
            best_value = float('-inf')
            env = Environment.from_state(state)
            for valid_action in valid_actions:
                temp_env = deepcopy(env)
                new_state, reward, new_done, new_valid_actions = temp_env.step(valid_action)
                new_total_reward = total_reward + reward
                child = self._minimax_level(new_state, new_total_reward, new_done, new_valid_actions, depth-1, False)
                if child[0] > best_value:
                    best_value = child[0]
                    best_action = valid_action
        else:
            # The opponent's turn
            best_value = float('inf')
            env = Environment.from_state(state)
            for valid_action in valid_actions:
                temp_env = deepcopy(env)
                new_state, reward, new_done, new_valid_actions = temp_env.step(valid_action)
                new_total_reward = total_reward - reward
                child = self._minimax_level(new_state, new_total_reward, new_done, new_valid_actions, depth-1, True)
                if child[0] < best_value:
                    best_value = child[0]
                    best_action = valid_action
        return best_value, best_action
