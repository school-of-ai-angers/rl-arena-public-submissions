import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    The Q-network with two hidden layers, taking as input the current state
    (see details on the encoding below) and outputs the Q-value for each action,
    even the invalid ones
    """

    def __init__(self, state_size, action_size, hidden_1_size, hidden_2_size):
        super().__init__()

        self.hidden_layer = nn.Linear(state_size, hidden_1_size)
        self.hidden_layer2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.output_layer = nn.Linear(hidden_2_size, action_size)

        init_layer(self.hidden_layer, hidden_init_scale(self.hidden_layer))
        init_layer(self.hidden_layer2, hidden_init_scale(self.hidden_layer2))
        init_layer(self.output_layer, 1e-3)

    def forward(self, state):
        x = F.relu(self.hidden_layer(state))
        x = F.relu(self.hidden_layer2(x))
        return self.output_layer(x)


def hidden_init_scale(layer):
    fan_in = layer.weight.data.size()[0]
    return 1. / np.sqrt(fan_in)


def init_layer(layer, scale):
    layer.weight.data.uniform_(-scale, scale)


def encode_state(state):
    # Encode the state as a one-hot encoding of the position of each one of the 16 pieces,
    # representing each one with a 17-element vector, where the first 16 represent a position
    # in the board and the last the reserve. If the piece is not yet in game, it will be
    # represented by the zero vector, that is, 17 zeros
    piece_is_there = np.concatenate([state == piece for piece in range(16)])

    # Normalize features (measured empirically)
    feature_mean = 0.023325
    feature_std = 0.136602
    encoded_state = torch.Tensor(piece_is_there.astype('float'))
    encoded_state = (encoded_state - feature_mean) / feature_std

    return encoded_state


def decode_action_values(encoded_action_values):
    # decoded[k, 16 * position + piece] = encoded[k, position] * encoded[k, 16 + piece]
    positions = encoded_action_values[:, :16].unsqueeze(2)
    pieces = encoded_action_values[:, 16:].unsqueeze(1)
    return positions.matmul(pieces).reshape((-1, 256))


class TrainedDQAgent:
    def __init__(self, state_size, action_size, hidden_1_size, hidden_2_size, state_dict, encode_state_fn, decode_action_values_fn):
        self.qnetwork = QNetwork(state_size, action_size, hidden_1_size, hidden_2_size)
        self.qnetwork.load_state_dict(state_dict)
        self.qnetwork.eval()
        self.encode_state_fn = encode_state_fn
        self.decode_action_values_fn = decode_action_values_fn

    def start(self, state, valid_actions):
        return self._take_action(self.encode_state_fn(state), valid_actions)

    def step(self, state, valid_actions, reward):
        return self._take_action(self.encode_state_fn(state), valid_actions)

    def end(self, state, reward):
        pass

    def _take_action(self, state, valid_actions):
        # Chose the action with the highest Q-value estimated by the local network
        with torch.no_grad():
            action_values = self.decode_action_values_fn(self.qnetwork(state.unsqueeze(0)))[0]
        valid_action_values = action_values[valid_actions]
        return valid_actions[np.argmax(valid_action_values)]


class Player(TrainedDQAgent):
    def __init__(self, train_mode, weights_file='weights.pth'):
        super().__init__(17*16, 16+16, 128, 128, torch.load(weights_file), encode_state, decode_action_values)
