import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import deque

from DQN_utils import preprocess
from visualization import plot_rewards

import torch
import torch.nn as nn
import torch.optim as optim

import pickle 

class DQN_agent:
    def __init__(self, env, network, buffer, epsilon=0.25, batch_size=32):
        # config
        self.env = env
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.buffer = buffer
        self.window = 100
        self.skip_frames = 4
        # network
        self.network = network
        self.target_network = deepcopy(network)
        self.tau = network.tau
        # init
        self.training_rewards = []
        self.losses = []
        self.mean_training_rewards = []
        self.mean_validation_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = preprocess(self.env.reset())
        self.state_buffer = deque(maxlen=self.tau)
        self.next_state_buffer = deque(maxlen=self.tau)
        [self.state_buffer.append(np.zeros(self.s_0.shape)) for i in range(self.tau)]
        [self.next_state_buffer.append(np.zeros(self.s_0.shape)) for i in range(self.tau)]

    def take_step(self, mode='train'):
        r_reg = 0
        state_buffer = deepcopy(self.state_buffer)
        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            action = self.network.get_action(np.stack([self.state_buffer]), epsilon=self.epsilon)
            self.step_count += 1
        for i in range(self.skip_frames):
            self.state_buffer.append(self.s_0)
            s_1_raw_i, r_raw, done, _ = self.env.step(action)
            self.rewards += r_raw
            s_1_i = preprocess(s_1_raw_i)        
            self.next_state_buffer.append(s_1_i.copy())
            self.s_0 = s_1_i.copy()
            r_reg = max(r_reg, r_raw) # give max reward of 4 frames
            if done:
                break
    
        r = self.filter_reward(r_reg, done)
        self.buffer.append(state_buffer, action, r, done, deepcopy(self.next_state_buffer))

        return done

    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=10000, batch_size=32,
              network_update_frequency=4, network_sync_frequency=2000,
              network_save_frequency=100, network_evaluate_frequency=100,
              n_val_episodes=10, start_from_eps=0, checkpoint_path=None, 
              epsilon_start=None, epsilon_end=None, epsilon_final_episode=None,
              checkpoint_prefix="vanilla_dqn", plot_result=False):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_prefix = checkpoint_prefix
        self.gamma = gamma
        self.network_evaluate_frequency = network_evaluate_frequency
        self.mean_validation_rewards = {} # possible to not reset 

        # Annealing
        if not (epsilon_start is None or epsilon_end is None or epsilon_final_episode is None):
            self.epsilon = epsilon_start
            eps_incr = (epsilon_end-epsilon_start)/epsilon_final_episode

        if start_from_eps > 0:
            pth = self.network.checkpoint_path+'checkpoint_'+str(start_from_eps)+'_eps.pth'
            self.network.load_state_dict(torch.load(pth))
        
        # Populate replay buffer
        print("Populating replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode='explore')
            if done:
              self.s_0 = preprocess(self.env.reset())

        # Start learning
        print("Beginning training...")
        ep = start_from_eps
        training = True
        while training:
            self.s_0 = preprocess(self.env.reset())
            self.rewards = 0
            done = False
            while not done:
                # step
                done = self.take_step(mode='train')

                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()

                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                    self.sync_eps.append(ep)

                if done:
                    # Evaluate network
                    if ep % network_evaluate_frequency == 0:
                        self.eval_performance(n_val_episodes, ep)

                    # Save network and learning curves
                    if ep != 0 and ep % network_save_frequency == 0:
                        # network
                        filename = 'checkpoint_'+str(ep)+'_eps.pth'
                        if self.checkpoint_prefix != "":
                            filename = self.checkpoint_prefix + "_" + filename
                        self.network.save_weights(filename)
                        # Learning curves
                        self.save_learning_curves(prefix=self.checkpoint_prefix)

                    # log
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    txt = "\rEpisode {:d} Mean Rewards {:.2f}\t\t"
                    print(txt.format(ep, mean_rewards), end="")

                    if ep < epsilon_final_episode:
                        self.epsilon += eps_incr  # Anneal epsilon

                    # Terminate
                    if ep >= max_episodes:
                        print('\nEpisode limit reached.')
                        if plot_result:
                            self.plot_results()
                        return
                
    def calculate_loss(self, batch):
        dev = self.network.device
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=dev)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1,1).to(device=dev)
        dones_t = torch.ByteTensor(dones).to(dtype=torch.bool).to(device=dev)

        q_vals_raw = self.network.get_qvals(states)
        qvals = torch.gather(q_vals_raw, 1, actions_t) #.squeeze()
        q_vals_next_raw = self.target_network.get_qvals(next_states)
        qvals_next = torch.max(q_vals_next_raw, dim=-1)[0].detach()
        qvals_next[dones_t] = 0 # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        return loss
      
    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.network.clip_val)
        self.network.optimizer.step()
        if self.network.device == 'cuda':
            self.losses.append(loss.detach().cpu().numpy())
        else:
            self.losses.append(loss.detach().numpy())
    
    def filter_reward(self, r, done=False):
        if done:
            return -1
        else:
            return 0
        return max(-1,min(1, r))

    def eval_performance(self, n_val_episodes, eps):
        rewards = []
        for _ in range(n_val_episodes):
            rewards.append(float(self.play_a_game()))
        self.mean_validation_rewards[int(eps)] = np.mean(np.array(rewards))
          
    def play_a_game(self):
        state_buffer = deque(maxlen=self.tau) # init state buffer
        init_state = preprocess(self.env.reset())
        [state_buffer.append(np.zeros(init_state.shape)) for i in range(4)]
        state_buffer.append(init_state)
        r = 0.0
        done = False
        while not done:
            action = self.network.get_action([state_buffer])
            state, reward, done, info = self.env.step(action) 
            state_buffer.append(preprocess(state))
            r += reward
        return r

    def save_learning_curves(self, name="learning_curves", prefix=""):
        if self.checkpoint_path is None:
            return
        learning_curves = {
            "train": self.training_rewards,
            "mean_train": self.mean_training_rewards,
            "val": self.mean_validation_rewards,
            "losses": self.losses
        }
        path = self.checkpoint_path + name + ".pkl"
        if prefix != "":
            path = self.checkpoint_path + prefix + "_" + name + ".pkl" 
        with open(path, 'wb+') as f:
            pickle.dump(learning_curves, f, pickle.HIGHEST_PROTOCOL)

    def plot_results(self, start_from_eps=0):
        plot_rewards(self.training_rewards,
                    self.mean_training_rewards,
                    self.mean_validation_rewards,
                    start_from_eps)