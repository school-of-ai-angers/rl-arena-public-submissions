import numpy as np


def _pos(i, j):
    return 7 * i + j


rows = [
    [_pos(line, col_start+k) for k in range(4)]
    for line in range(6)
    for col_start in range(4)
]
columns = [
    [_pos(line_start+k, col) for k in range(4)]
    for line_start in range(3)
    for col in range(7)
]
diagonals_1 = [
    [_pos(line_start+k, col_start+k) for k in range(4)]
    for line_start in range(3)
    for col_start in range(4)
]
diagonals_2 = [
    [_pos(line_start+k, col_start-k) for k in range(4)]
    for line_start in range(3)
    for col_start in range(3, 7)
]


class Environment:

    lines = rows + columns + diagonals_1 + diagonals_2

    def __init__(self):
        # Encode the board as 42 positions with 0 (empty), 1 (first player) or -1 (second player).
        # The position at line `i` and column `j` will be at `7*i+j`.
        # The 43-th position is the current player (-1 or 1)
        self.state = np.zeros((43,))

    def reset(self):
        self.state[:42] = 0
        self.state[42] = 1
        return self.state, list(range(7))

    def step(self, action):
        assert self.state[_pos(0, action)] == 0, 'Invalid action'

        # Put the piece on the board
        for i in reversed(range(6)):
            if self.state[_pos(i, action)] == 0:
                pos = _pos(i, action)
                self.state[pos] = self.state[42]
                break

        # Check for win
        for pos_s in self.lines:
            if pos in pos_s:
                values = self.state[pos_s]
                if np.all(values == 1) or np.all(values == -1):
                    return self.state, 1, True, []

        # Check for draw
        if np.all(self.state != 0):
            return self.state, 0, True, []

        # update list of possible actions
        self.state[42] = -self.state[42]
        return self.state, 0, False, np.nonzero(self.state[[_pos(0, k) for k in range(7)]] == 0)[0]

    @classmethod
    def from_state(cls, state):
        env = cls()
        env.state = state
        return env
