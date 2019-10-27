import numpy as np


class Player:
    """ Play the first move """

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
        return valid_actions[0]

    def step(self, state, valid_actions, prev_reward):
        """
        :param state: np.array
        :param valid_actions: np.array 1D
        :param prev_reward: float
        :returns: float
        """
        return valid_actions[0]

    def end(self, state, prev_reward):
        """
        :param state: np.array
        :param prev_reward: float
        """
        pass
