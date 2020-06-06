import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    """
    Vanilla DQN
    Source: https://www.datahubbs.com/deepmind-dqn/
    """
    def __init__(self, env, learning_rate = 1e-3, device='cpu',
        input_dim=(84,84), checkpoint_path=None, *args, **kwargs):
        super().__init__()
        self.device = str(torch.device(device))
        self.checkpoint_path = checkpoint_path
        self.tau = 4 # nr of stacked frames
        self.learning_rate = learning_rate
        self.actions = np.arange(env.action_space.n)
        self.n_outputs = env.action_space.n
        self.clip_val = 10

        # CNN modeled off of Mnih et al.
        self.cnn = nn.Sequential(
          nn.Conv2d(self.tau, 32, kernel_size=8, stride=4),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=4, stride=2),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1),
          nn.ReLU()
        )
        
        self.fc_layer_inputs = self.cnn_out_dim(input_dim)
        
        self.fully_connected = nn.Sequential(
          nn.Linear(self.fc_layer_inputs, 512, bias=True),
          nn.ReLU(),
          nn.Linear(512, self.n_outputs))
      
        # Set device for GPU's
        if self.device == 'cuda':
            self.cnn.cuda()
            self.fully_connected.cuda()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      
    def get_action(self, state, epsilon=0):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy_action(state)
        return action
    
    def greedy_action(self, state):
        qvals = self.get_qvals(state)
        return torch.max(qvals, dim=-1)[1].item()
      
    def get_qvals(self, state):
        state_t = torch.FloatTensor(state).to(device=self.device)
        cnn_out = self.cnn(state_t).reshape(-1, self.fc_layer_inputs)
        return self.fully_connected(cnn_out)

    def cnn_out_dim(self, input_dim):
        return self.cnn(torch.zeros(1, self.tau, *input_dim)).flatten().shape[0]

    def save_weights(self, filename='checkpoint.pth'):
        path = self.checkpoint_path+filename 
        torch.save(self.state_dict(), path)

