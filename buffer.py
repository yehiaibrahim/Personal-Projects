import numpy as np


class MyOwnBuffer():
    
    """
    A custom class of an LSTM buffer that for a robot arm that has 2 states and 3 actions 
    
    """
            
    def __init__(self, buffer_size, sequence_length, features):
        
        self.size_idx = 0
        self.seq_idx = 0
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.features = features
        self.buffer = np.zeros((self.buffer_size, self.sequence_length, self.features))

    def add(self, state_1, state_2, action, action_n, action_m, state_1_pred, state_2_pred, reward):
        
        
        """        
        Parameters
        ----------
        state_1 : ndarray
            State 1.
        state_2 : ndarray
            State 2.
        action : ndarray
            Action.
        action_n : ndarray
            Action.
        action_m : ndarray
            Action.
        state_1_pred : ndarray
            Prediction of State 1 .
        state_2_pred : ndarray
            Prediction of State 2.
        reward : ndarray
            Reward.

        Returns
        -------
        ndarray
            The sampled data.

        """
              
        if self.seq_idx == 0:
            self.state_1 = state_1
            self.state_2 = state_2
            
        self.action = action
        self.action_n = action_n
        self.action_m = action_m
        self.state_1_pred = state_1_pred
        self.state_2_pred = state_2_pred
        self.reward = reward
        features_array = np.array([self.state_1, self.state_2,
                                   self.action, self.action_n, 
                                   self.action_m, self.state_1_pred, 
                                   self.state_2_pred, self.reward], dtype=object)
        
        if self.size_idx >= self.buffer_size:
            last, self.buffer = self.buffer[-1], self.buffer[:-1]
            self.buffer = np.swapaxes(self.buffer,0,-1)
            self.buffer = np.concatenate((self.buffer, 
                                np.broadcast_to(np.array(features_array)[:, None, None], self.buffer.shape[:-1] + (1,))), 
                               axis = -1)
            self.buffer = np.swapaxes(self.buffer,0,-1)
            self.size_idx = self.size_idx -1
        
        if self.seq_idx < self.sequence_length:
            self.buffer[self.size_idx, self.seq_idx,:] = features_array
            self.seq_idx += 1
        
        if self.seq_idx == self.sequence_length:
            self.seq_idx = 0
            self.size_idx += 1

        return True

    def sample(self, samples_size):
        
        self.samples_size = samples_size
        self.sample_idx = 0
        samples = np.zeros((self.sequence_length, self.samples_size, self.features))
        if self.sample_idx < self.samples_size:
            batch_idx = np.random.choice(self.buffer_size, self.samples_size)  
            samples = self.buffer[batch_idx]              
            self.sample_idx +=1
        
        return samples
        