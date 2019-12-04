# An example of a valid submission to the arena
# See more instructions in the Notebook `Train your player.ipynb`

import numpy as np
import pickle
from copy import deepcopy
from random import shuffle

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



# import quarto rules to be able to do a MCTS

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


def estimate_values_by_MCTS(state, valid_actions, action_type='pos'):
    """
    Estimate the value of each action by Monte-Carlo Tree Search
    :returns: action_MCTS_values: numpy array
    """
    # compute valid_pos_actions
    valid_pos_actions = sorted(list(set([va // 16 for va in valid_actions])))
    
    # compute valid_piece_actions
    valid_piece_actions = sorted(list(set([va % 16 for va in valid_actions])))
    
    remaining_actions = min(len(valid_pos_actions), len(valid_piece_actions))
    # redefine the method to 
    # use Monte-Carlo Tree Search when a state is unknown in the q_table
    MC_sample = 25 if remaining_actions <= 6 else int(80 // remaining_actions) #min(25, int(200 // min(len(valid_pos_actions), len(valid_piece_actions)))) # 25

    # MCTS to eval value of each actions_to_be_estimated
    action_MCTS_values = []
    
    # select the type of action to estimate
    if action_type == 'pos':
        actions_to_be_estimated = valid_pos_actions
    elif action_type == 'piece':
        actions_to_be_estimated = valid_piece_actions   


    for estimated_act in actions_to_be_estimated:
        # init the estimate for the pos_a 
        estimate_value = 0
        for _ in range(MC_sample):
            winner_side = 1
            
            # init the MC board state
            MC_board_state = np.array(state) #deepcopy(np.array(state))  
            
            # init the valid pos actions
            MC_valid_pos_actions = valid_pos_actions.copy() #deepcopy(valid_pos_actions)
            if action_type == 'pos':
                MC_valid_pos_actions.remove(estimated_act)
            # random shuffle the pos actions
            shuffle(MC_valid_pos_actions)
                        
            # init the valid piece actions
            MC_valid_piece_actions = valid_piece_actions.copy() #deepcopy(valid_piece_actions)
            if action_type == 'piece':
                MC_valid_piece_actions.remove(estimated_act)   
            # random shuffle the piece actions
            shuffle(MC_valid_piece_actions)
            
            # do the estmated action
            if action_type == 'pos':
                MC_board_state[estimated_act] = MC_board_state[16]
                MC_board_state[16] = -1 
            
                MC_piece_act = None
            elif action_type == 'piece':
                MC_piece_act = estimated_act                    
            
            # eval board status to determine if there is a winning line
            board_status = get_board_status(MC_board_state)
            
            # play until the end of the game
            while board_status == 0 and min(len(MC_valid_pos_actions), len(MC_valid_piece_actions)) > 0:               
                # alternate the winner side
                winner_side *= -1
                
                # sample a piece and a pos to play
                if MC_piece_act is None:
                    MC_piece_act = MC_valid_piece_actions.pop()
                MC_pos_act = MC_valid_pos_actions.pop()
                
                # update the board state by playing
                MC_board_state[MC_pos_act] = MC_piece_act
                MC_piece_act = None
                
                # eval the new board status to determine if there is a winnning line
                board_status = get_board_status(MC_board_state)
            
            # add the sample_value to the estimate
            estimate_value += winner_side * board_status #* (1 + np.count_nonzero(MC_board_state == -1) / 1) if winner_side * board_status > 0 else winner_side * board_status
            
        # compute the MC estimation and add it to the MCTS_value list
        action_MCTS_values.append(estimate_value / MC_sample)
        
    return np.array(action_MCTS_values)        





# Implement a Q-learning agent

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
        self.action_space = 16 #256
        
        # Epsilon scheduling
        self.epsilon = 1
        self.min_epsilon = 0.5 #0.1 #0.8 #0.1 #0.2 #0.1
        self.epsilon_decay = 0.99995
        
        # Q-table update hyperparameters
        self.alpha = 0.02 #0.02 #0.001 #0.1
        self.gamma = 0.8 #0.5 #1
        
        # Q-table update helper variables
        self.memory_step_horizon = 2 #10 #2
        self.prev_state = [] # None
        self.prev_action = [] #None

    def start(self, state, valid_actions):
        # First move: take the action
        return self._take_action(state, valid_actions)

    def step(self, state, valid_actions, reward):
        # At every other step: update the q-table and take the next action
        # Since the game hasn't finished yet, we can use current knowledge of the q-table
        # to estimate the future reward.
        if self.train_mode: #if self.train_mode:
            pos_values = self._get_action_values(state, valid_actions, action_type='pos')
            self._update_q_table(reward, self.gamma * np.max(pos_values))
        return self._take_action(state, valid_actions)

    def end(self, state, reward):
        # Last step: update the q-table and schedule the next value for epsilon
        # Here, the expected action-value is simply the final reward
        if self.train_mode:
            # scale the reward according to the number of empty case (increase the effect of early win/loss)
            reward = reward * (1 + np.count_nonzero(state == -1) / 1) if reward > 0 else reward
            self._update_q_table(reward, 0, end=True)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _update_q_table(self, reward, delta_reward, end=False):
        # Based on the reward of the previous action take at the previous step,
        # update the q-table to be closer to the desired value.
        # Note that, if `alpha` is zero, the q-table is left unchanged,
        # if `alpha` is one, the q-table will simply take the `new_value`.
        # With a value in between, one can control the tradeoff between learning too much
        # or too little from a single move
        #prev_state = tuple(self.prev_state)
        #q_row = self.q_table.setdefault(prev_state, np.zeros(self.action_space))
        #q_row[self.prev_action] += self.alpha * (new_value - q_row[self.prev_action])
        
        new_value = reward + delta_reward
        empty_actions = [16 * pos +
                             piece for pos in range(16) for piece in range(15)]
        
        if end:
            # create or update the previous states in the memory horizon when a terminal state is found
            discount = 1
            for prev_state, prev_action in zip(self.prev_state[::-1], self.prev_action[::-1]):
                # reverse tranform to get pos and piece action and states from previous ones
                prev_state = tuple(prev_state)
                prev_pos_action = prev_action // 16
                
                updated_state = list(prev_state).copy()
                updated_state[prev_pos_action] = list(prev_state)[16]
                updated_state[16] = -1
                updated_state = tuple(updated_state)
                prev_piece_action = prev_action % 16
                
                remaining_actions = min(prev_state.count(-1), updated_state.count(-1))
                
                # updating values in reverse order
                # update PIECE action value (if this a state that is not easy to handle by MCTS)
                if remaining_actions > 6:
                    q_row = self.q_table.setdefault(updated_state, np.zeros(self.action_space))
                    q_row[prev_piece_action] += self.alpha * (new_value - q_row[prev_piece_action])

                    # update new_value for next update
                    #piece_values = self._get_action_values(updated_state, empty_actions, action_type='piece')
                    #new_value = self.gamma * np.max(piece_values)
                    new_value = self.gamma * np.max(q_row)
                else:
                    # compute an estimated new_value, just in case
                    new_value = self.gamma * new_value
                
                # update POS action value
                if remaining_actions > 6:
                    q_row = self.q_table.setdefault(prev_state, np.zeros(self.action_space))
                    q_row[prev_pos_action] += self.alpha * (new_value - q_row[prev_pos_action])

                    # update new_value for next update
                    #pos_values = self._get_action_values(prev_state, empty_actions, action_type='pos')
                    #new_value = self.gamma * np.max(pos_values)
                    new_value = self.gamma * np.max(q_row)
                else:
                    # compute an estimated new_value, just in case
                    new_value = self.gamma * new_value

                             
                
        else:
            
            new_value = reward + delta_reward
            
            # update only the last prev_state if it is not an "end"/"terminal" state
            # reverse tranform to get pos and piece action and states from previous ones            
            prev_state = tuple(self.prev_state[-1])
            prev_pos_action = self.prev_action[-1] // 16
            
            updated_state = list(prev_state).copy()
            updated_state[prev_pos_action] = list(prev_state)[16]
            updated_state[16] = -1
            updated_state = tuple(updated_state)
            prev_piece_action = self.prev_action[-1] % 16
            
            # update PIECE action value
            if tuple(updated_state) in self.q_table:
                q_row = self.q_table.setdefault(updated_state, np.zeros(self.action_space))
                q_row[prev_piece_action] += self.alpha * (new_value - q_row[prev_piece_action])
            
                # update new_value for next update
                #piece_values = self._get_action_values(updated_state, empty_actions, action_type='piece')
                #new_value = self.gamma * np.max(piece_values)
                new_value = self.gamma * np.max(q_row)
            else:
                new_value = self.gamma * new_value
                
            # update POS action value
            if tuple(prev_state) in self.q_table:
                q_row = self.q_table.setdefault(prev_state, np.zeros(self.action_space))
                q_row[prev_pos_action] += self.alpha * (new_value - q_row[prev_pos_action])
            
            
            
            
            

    def _take_action(self, state, valid_actions):
        # Store the current state, copying it, otherwise the environment could mutate it afterwards
        self.prev_state.append(state.copy())
        
        # compute valid_pos_actions nd valid_piece_actions
        valid_pos_actions = sorted(list(set([va // 16 for va in valid_actions])))
        valid_piece_actions = sorted(list(set([va % 16 for va in valid_actions])))
        
        # prune older prev state and action becoming out of the memory horizon
        if len(self.prev_state) > self.memory_step_horizon:
            del self.prev_state[0]
            del self.prev_action[0]
        
        if self.train_mode and np.random.random() <= self.epsilon: #not self.train_mode or np.random.random() <= self.epsilon:
            # Take a random action
            #self.prev_action.append(np.random.choice(valid_actions))
            
            # --------------
            
            # sample a action based on softmax proba
            # take POS action
            pos_action_values = self._get_action_values(state, valid_actions, action_type='pos')
            
            #r = np.random.random()
            if False: #r * 200 < 100 - sum(pos_action_values):
                prev_pos_action = np.random.choice(valid_pos_actions)
            else:
                # take action by sampling the softmax proba of action values
                proba_pos_action = softmax(np.array(pos_action_values))

                sample = np.random.random()

                cumul_proba = 0
                for a, p in enumerate(proba_pos_action):
                    cumul_proba += p
                    if sample <= cumul_proba:
                        break
                prev_pos_action = valid_pos_actions[a]

                   
            
            # UPDATE the STATE with the prev_pos_action
            updated_state = state.copy()
            updated_state[prev_pos_action] = state[16]
            updated_state[16] = -1
            updated_state = tuple(updated_state)
            
            # take PIECE action
            piece_action_values = self._get_action_values(updated_state, valid_actions, action_type='piece')
            
            #r = np.random.random()
            if False: #r * 200 < 100 - sum(piece_action_values):
                prev_piece_action = np.random.choice(valid_piece_actions)
            else:
                # take action by sampling the softmax proba of action values
                proba_piece_action = softmax(np.array(piece_action_values))

                sample = np.random.random()

                cumul_proba = 0
                for a, p in enumerate(proba_piece_action):
                    cumul_proba += p
                    if sample <= cumul_proba:
                        break
                prev_piece_action = valid_piece_actions[a]
                
                
            # update the prev_action pile by building the prev_action based on pos and piece actions
            self.prev_action.append(16 * prev_pos_action + prev_piece_action)
                 
            
        else:
            # Take the action that has the highest expected future reward,
            # that is, the highest action value
            # CSP: if several actions share the highest action value, take a random action among them
            
            # take POS action
            pos_action_values = self._get_action_values(state, valid_actions, action_type='pos')
            
            #if self.train_mode:
                #highest_value = max(action_values)
                #highest_value_actions = [a for a, v in zip(valid_actions, action_values) if v == highest_value]
                #if len(highest_value_actions) > 1:
                #        self.prev_action.append(np.random.choice(highest_value_actions))
                #else:
                #    self.prev_action.append(highest_value_actions[0])

                #self.prev_action.append(valid_actions[np.argmax(action_values)])
            #else:
            #r = np.random.random()
            if False: #self.train_mode: #False: #r < 0.2 - self.epsilon :
                # take action by sampling the softmax proba of action values
                proba_pos_action = softmax(np.array(pos_action_values))
                
                sample = np.random.random()
                
                cumul_proba = 0
                for a, p in enumerate(proba_pos_action):
                    cumul_proba += p
                    if sample <= cumul_proba:
                        break
                prev_pos_action = valid_pos_actions[a]
                
                # take a random action among the best valued
                #pos_highest_value = max(pos_action_values)
                #highest_value_pos_actions = [a for a, v in zip(valid_pos_actions, pos_action_values) if v == pos_highest_value]
                #if len(highest_value_pos_actions) > 1:
                #        prev_pos_action = np.random.choice(highest_value_pos_actions)
                #else:
                #    prev_pos_action = highest_value_pos_actions[0]
            else:
                # take the 1st action among the best valued
                prev_pos_action = valid_pos_actions[np.argmax(pos_action_values)]
                   
            
            # UPDATE the STATE with the prev_pos_action
            updated_state = state.copy()
            updated_state[prev_pos_action] = state[16]
            updated_state[16] = -1
            updated_state = tuple(updated_state)
            
            # take PIECE action
            piece_action_values = self._get_action_values(updated_state, valid_actions, action_type='piece')
            
            #r = np.random.random()
            if False: #self.train_mode: #False: # r < 0.2 - self.epsilon :
                # take action by sampling the softmax proba of action values
                proba_piece_action = softmax(np.array(piece_action_values))
                
                sample = np.random.random()
                
                cumul_proba = 0
                for a, p in enumerate(proba_piece_action):
                    cumul_proba += p
                    if sample <= cumul_proba:
                        break
                prev_piece_action = valid_piece_actions[a]
                
                # take a random action among the best valued
                #piece_highest_value = max(piece_action_values)
                #highest_value_piece_actions = [a for a, v in zip(valid_piece_actions, piece_action_values) if v == piece_highest_value]
                #if len(highest_value_piece_actions) > 1:
                #        prev_piece_action = np.random.choice(highest_value_piece_actions)
                #else:
                #    prev_piece_action = highest_value_piece_actions[0]
            else:
                # take the 1st action among the best valued
                prev_piece_action = valid_piece_actions[np.argmax(piece_action_values)]
                
                
            # update the prev_action pile by building the prev_action based on pos and piece actions
            self.prev_action.append(16 * prev_pos_action + prev_piece_action)
                    
        return self.prev_action[-1]
    
    def _get_action_values(self, state, valid_actions, action_type='pos'):
        # Convert from numpy array to tuple
        state = tuple(state)
        #if self.train_mode:
        #    # Return saved action values. If this is the first time this state is visited,
        #    # set all values to zero
        #    return self.q_table.setdefault(state, np.zeros(self.action_space))[valid_actions]
        # When not in train mode, do not change the Q-table, just return a new default for
        # every new never-visited state
        
        
        if action_type == 'pos':
            # compute valid_pos_actions
            valid_pos_actions = sorted(list(set([va // 16 for va in valid_actions])))
            # return self.q_table.get(state, np.zeros(self.action_space))[valid_pos_actions]
            if state in self.q_table:
                return self.q_table[state][valid_pos_actions]
            else:
                if len(valid_pos_actions) > 6: #False: #len(valid_pos_actions) > 6:
                    return np.zeros(self.action_space)[valid_pos_actions]
                else:
                    # estimate values by Monte-Carlo Tree Search
                    return estimate_values_by_MCTS(state, valid_actions, action_type)
            
        else:
            # compute valid_piece_actions
            valid_piece_actions = sorted(list(set([va % 16 for va in valid_actions])))
            #return self.q_table.get(state, np.zeros(self.action_space))[valid_piece_actions]
            if state in self.q_table:
                return self.q_table[state][valid_piece_actions]
            else:
                if len(valid_piece_actions) > 6: #False: #len(valid_piece_actions) > 6:
                    return np.zeros(self.action_space)[valid_piece_actions]
                else:
                    # estimate values by Monte-Carlo Tree Search
                    return estimate_values_by_MCTS(state, valid_actions, action_type)

    def get_freezed(self):
        # Return a copy of the player, but not in train_mode
        # This is used by the training loop, to replace the adversary from time to time
        copy = deepcopy(self)
        copy.train_mode = False
        return copy

    def save(self):
        # Save the q-table on the disk for future use
        with open('quarto/submission/player.bin', 'wb') as fp:
            pickle.dump(dict(self.q_table), fp, protocol=4)
            

class Player(QLearningPlayer):
    def __init__(self, train_mode):
        super().__init__(train_mode)
        # do NOT use any q_table, only a MCTS-based decision policy
        self.q_table = {}
        #with open('player.bin', 'rb') as fp:
        #    self.q_table = pickle.load(fp)
            
    def _get_action_values(self, state, valid_actions, action_type='pos'):
        # re-define the get_action values method to incorporate the MCTS
        
        # Convert from numpy array to tuple
        state = tuple(state)
        
        if action_type == 'pos':
            # compute valid_pos_actions
            valid_pos_actions = sorted(list(set([va // 16 for va in valid_actions])))
            # return self.q_table.get(state, np.zeros(self.action_space))[valid_pos_actions]
            if state in self.q_table:
                return self.q_table[state][valid_pos_actions]
            else:
                if False: #len(valid_pos_actions) > 6:
                    return np.zeros(self.action_space)[valid_pos_actions]
                else:
                    # estimate values by Monte-Carlo Tree Search
                    return estimate_values_by_MCTS(state, valid_actions, action_type)
            
        else:
            # compute valid_piece_actions
            valid_piece_actions = sorted(list(set([va % 16 for va in valid_actions])))
            #return self.q_table.get(state, np.zeros(self.action_space))[valid_piece_actions]
            if state in self.q_table:
                return self.q_table[state][valid_piece_actions]
            else:
                if False: #len(valid_piece_actions) > 6:
                    return np.zeros(self.action_space)[valid_piece_actions]
                else:
                    # estimate values by Monte-Carlo Tree Search
                    return estimate_values_by_MCTS(state, valid_actions, action_type)
