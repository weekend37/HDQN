import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from collections import deque

class HDQN_sep(nn.Module):
    """
    HDQN
    """
    def __init__(self, env, learning_rate = 1e-3, device='cpu',
        input_dim=(84,84), checkpoint_path=None, 
        n_options = 4,
        *args, **kwars):
        super().__init__()
        self.device = str(torch.device(device))
        self.checkpoint_path = checkpoint_path
        self.tau = 4 # nr of stacked frames
        self.learning_rate = learning_rate
        self.actions = np.arange(env.action_space.n)
        self.n_outputs = env.action_space.n
        self.clip_val = 10
        # HDQN
        self.n_options = 4
        self.current_option = 0

        # share first layer (to connect the graphs) 
        self.cnn = nn.Sequential(
          nn.Conv2d(self.tau, 32, kernel_size=8, stride=4),
          nn.ReLU()
        )

        # CNN Meta
        self.cnn_meta = nn.Sequential(
          nn.Conv2d(32, 64, kernel_size=4, stride=2),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1),
          nn.ReLU()
        )

        # CNN options
        self.cnn_options = {}
        for i in range(n_options):
            self.cnn_options[i] = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
        
        # Fully connected layers        
        self.fc_layer_inputs = self.out_dim(input_dim=input_dim)
        
        # meta
        self.fc_meta = nn.Sequential(
          nn.Linear(self.fc_layer_inputs, 512, bias=True),
          nn.ReLU(),
          nn.Linear(512, self.n_options)
        )

        # options
        self.fc_options = {}
        for i in range(n_options):
            self.fc_options[i] = nn.Sequential(
              nn.Linear(self.fc_layer_inputs, 512, bias=True),
              nn.ReLU(),
              nn.Linear(512, self.n_outputs)
            )

        # Set device for GPU's
        if self.device == 'cuda':
            self.cnn.cuda()
            self.cnn_meta.cuda()
            self.fc_meta.cuda()
            [self.cnn_options[i].cuda() for i in range(n_options)]
            [self.fc_options[i].cuda() for i in range(n_options)]
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_option(self, state, epsilon=0):
        if np.random.random() < epsilon:
            option = np.random.randint(self.n_options)
        else:
            ovals = self.get_ovals(state)
            option = torch.max(ovals, dim=-1)[1].item()
        return option

    def get_ovals(self, state):
        state_t = torch.FloatTensor(state).to(device=self.device)
        shared_cnn_out = self.cnn(state_t)
        cnn_out = self.cnn_meta(shared_cnn_out).reshape(-1, self.fc_layer_inputs)
        meta_out = self.fc_meta(cnn_out)
        return meta_out

    # actions ----------------------------------------------------------------

    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy_action(state)
        return action
    
    def greedy_action(self, state):
        qvals = self.get_qvals(state)
        return torch.max(qvals, dim=-1)[1].item()
      
    def get_qvals(self, state):
        state_t = torch.FloatTensor(np.stack(state)).to(device=self.device)
        shared_cnn_out = self.cnn(state_t)
        cnn_out = self.cnn_options[self.current_option](shared_cnn_out).reshape(-1, self.fc_layer_inputs)
        qvals = self.fc_options[self.current_option](cnn_out)
        return qvals

    # misc ----------------------------------------------------------------

    def out_dim(self, input_dim):
        return self.cnn_meta(self.cnn(torch.zeros(1, self.tau, *input_dim))).flatten().shape[0]

    def save_weights(self, filename='checkpoint.pth'):
        path = self.checkpoint_path+filename 
        torch.save(self.state_dict(), path)
