import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import deque

from DQN_utils import preprocess
from visualization import plot_rewards, plot_option_ratio

import torch
import torch.nn as nn
import torch.optim as optim

import pickle 

class HDQN_agent:
    def __init__(self, env, network, buffer, epsilon=0.25, batch_size=32, option_len=10):
        # config
        self.env = env
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.window = 100
        self.skip_frames = 4
        # network
        self.network = network
        self.target_network = deepcopy(network)
        self.tau = network.tau
        # init
        self.training_rewards = []
        self.losses = {"meta":[], 0: [], 1:[], 2:[], 3:[]}
        self.mean_training_rewards = []
        self.mean_validation_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = preprocess(self.env.reset())
        self.state_buffer = deque(maxlen=self.tau)
        [self.state_buffer.append(np.zeros(self.s_0.shape)) for i in range(self.tau)]
        self.next_state_buffer = deepcopy(self.state_buffer)
        # HDQN
        self.option_len = option_len
        self.meta_step_count = 0
        self.meta_state = np.stack([deepcopy(self.state_buffer)])
        self.meta_rewards = 0
        self.meta_buffer = buffer
        self.option_buffer = deepcopy(buffer)
        self.meta_buffer.burn_in = batch_size

        # analysis
        self.train_ep_option_dist = np.zeros(self.network.n_options)
        self.train_ep_option_ratio = {}
        for o in range(self.network.n_options):
            self.train_ep_option_ratio[o] = {}
        self.eval_option_dist = np.zeros(self.network.n_options)

    def take_step(self, mode='train'):
        r_raw = 0
        state_buffer = deepcopy(self.state_buffer)
        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            action = self.network.get_action(np.stack([self.state_buffer]), epsilon=self.epsilon)
            self.step_count += 1
        for i in range(self.skip_frames):
            self.state_buffer.append(self.s_0)
            s_1_raw_i, r_raw_i, done, _ = self.env.step(action)
            s_1_i = preprocess(s_1_raw_i)        
            self.next_state_buffer.append(s_1_i.copy())
            self.s_0 = s_1_i.copy()
            r_raw = max(r_raw, r_raw_i) # give max reward of 4 frames
            if done:
                break
        
        self.rewards += r_raw
        self.meta_rewards += self.filter_reward(r_raw, done, network="meta")
        self.option_buffer.append(state_buffer, action, r_raw, done, deepcopy(self.next_state_buffer))

        return done

    def take_option(self, mode='train'):
        
        # Observe outcome
        done = False
        s_1 = np.stack([deepcopy(self.state_buffer)])
        self.meta_rewards /= self.option_len  # scale down to ~ [-1,1]
        self.meta_rewards = max(min(self.meta_rewards,1),-1) # truncate to [-1,1]
        self.meta_buffer.append(deepcopy(self.meta_state), self.network.current_option, self.meta_rewards, done, s_1)

        # Reset meta rewards and set next state
        self.meta_rewards = 0        
        self.meta_state = s_1

        # Take next option
        if mode == 'explore':
            next_option = np.random.choice(self.network.n_options)
        elif mode == 'flight':
            next_option = 0
        else:
            next_option = self.network.get_option(self.meta_state, epsilon=self.epsilon)
            self.meta_step_count += 1
        self.network.current_option = next_option

        # log it
        if mode != "eval":
            self.train_ep_option_dist[next_option] += 1
        else:
            self.eval_option_dist[next_option] += 1
        
        return next_option

    # HDQN training 
    def train(self, gamma=0.99, max_episodes=10000, batch_size=32,
              network_update_frequency=4, network_sync_frequency=2000,
              network_save_frequency=100, network_evaluate_frequency=100,
              n_val_episodes=10, start_from_eps=0, checkpoint_path=None, 
              checkpoint_prefix="hdqn", plot_result=False, 
              epsilon_start=None, epsilon_end=None, epsilon_final_episode=None,
              op_ratio_n_bins=50, meta_burn_in_ep=0):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_prefix = checkpoint_prefix
        self.gamma = gamma
        self.network_evaluate_frequency = network_evaluate_frequency
        self.mean_validation_rewards = {} # possible to not reset 
        self.options_network_update_frequency = network_update_frequency*self.network.n_options
        self.meta_network_update_frequency = network_update_frequency*self.option_len
        self.op_ratio_window = max(round((max_episodes-start_from_eps)/op_ratio_n_bins),1)
        self.meta_burn_in_ep = meta_burn_in_ep

        # Annealing
        if not (epsilon_start is None or epsilon_end is None or epsilon_final_episode is None):
            self.epsilon = epsilon_start
            eps_incr = (epsilon_end-epsilon_start)/epsilon_final_episode

        # Here it is possible to start from nonzero episode (see DQN_agent.py)

        # Populate the replay buffer
        print("Populating replay buffer...")
        p_step = 0
        while self.option_buffer.burn_in_capacity() < 1 or self.meta_buffer.burn_in_capacity() < 1:
            if p_step % self.option_len == 0:
                self.take_option(mode="explore")
            done = self.take_step(mode='explore')
            if done:
                self.s_0 = preprocess(self.env.reset())
            p_step += 1

        # Start learning
        print("Beginning training...")
        ep = start_from_eps
        training = True
        while training:
            # reset state and reward
            self.s_0 = preprocess(self.env.reset())
            [self.state_buffer.append(np.zeros(self.s_0.shape)) for i in range(self.tau)]
            self.next_state_buffer = deepcopy(self.state_buffer)
            self.rewards = 0
            done = False

            # let option-heads specialize before learning to choose between them
            meta_mode = "explore" if ep < self.meta_burn_in_ep else "train"

            # Play a game
            while not done:

                if self.step_count % self.option_len == 0:
                    self.take_option(mode=meta_mode) # Take next option

                done = self.take_step(mode='train')

                # Update options networks
                if self.step_count % (self.options_network_update_frequency+1) == 0:
                    # update all heads at the same rate
                    self.update(network='options')
                    self.update(network='meta')

                # # Update meta network
                # if self.step_count % self.meta_network_update_frequency == 0:
                #     self.update(network='meta')

                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                    self.sync_eps.append(ep)

            # Evaluate networks
            if ep % network_evaluate_frequency == 0:
                self.eval_performance(n_val_episodes, ep)

            # Save networks and learning curves
            if ep != 0 and ep % network_save_frequency == 0:
                # network
                filename = 'checkpoint_'+str(ep)+'_eps.pth'
                if self.checkpoint_prefix != "":
                    filename = self.checkpoint_prefix + "_" + filename
                self.network.save_weights(filename)
                # Learning curves
                self.save_learning_curves(prefix=self.checkpoint_prefix)

            # log stuff
            self.training_rewards.append(self.rewards)
            mean_rewards = np.mean(self.training_rewards[-self.window:])
            self.mean_training_rewards.append(mean_rewards)
            if meta_mode != "eval" and ep != 0 and ep % self.op_ratio_window == 0:
                op_ratios = self.train_ep_option_dist/max(np.sum(self.train_ep_option_dist, keepdims=True),1)
                for o in range(self.network.n_options):
                    self.train_ep_option_ratio[o][ep] = op_ratios[o]
                self.train_ep_option_dist = np.zeros(self.network.n_options)

            if ep < epsilon_final_episode:
                self.epsilon += eps_incr  # Anneal epsilon
            ep += 1
            txt = "\rEpisode {:d} Mean Rewards {:.2f}\t\t"
            print(txt.format(ep, mean_rewards), end="")
            
            if ep >= max_episodes:
                print('\nEpisode limit reached.')
                if plot_result:
                    self.plot_results()
                return
                
    def calculate_options_loss(self, batch, option):
        dev = self.network.device
        states, actions, rewards, dones, next_states = [i for i in batch]

        # filter rewards based on option head
        rewards = [self.filter_reward(rewards[i],dones[i],"options",option) for i in range(self.batch_size)]

        rewards_t = torch.FloatTensor(rewards).to(device=dev)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1,1).to(device=dev)
        dones_t = torch.ByteTensor(dones).to(dtype=torch.bool).to(device=dev)

        q_vals_raw = self.network.get_qvals(states)
        qvals = torch.gather(q_vals_raw, 1, actions_t)
        q_vals_next_raw = self.target_network.get_qvals(next_states)
        qvals_next = torch.max(q_vals_next_raw, dim=-1)[0].detach()
        qvals_next[dones_t] = 0 # Zero-out terminal states
        expected_qvals = (self.gamma * qvals_next + rewards_t).reshape(-1,1)
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss

    def calculate_meta_loss(self, batch):
        dev = self.network.device
        states, options, rewards, dones, next_states = [i for i in batch]
        states = np.stack(states).squeeze()
        next_states = np.stack(next_states).squeeze()

        rewards_t = torch.FloatTensor(rewards).to(device=dev)
        options_t = torch.LongTensor(np.array(options)).reshape(-1,1).to(device=dev)
        dones_t = torch.ByteTensor(dones).to(dtype=torch.bool).to(device=dev)

        o_vals_raw = self.network.get_ovals(states)
        ovals = torch.gather(o_vals_raw, 1, options_t)
        o_vals_next_raw = self.target_network.get_ovals(next_states)
        ovals_next = torch.max(o_vals_next_raw, dim=-1)[0].detach()
        ovals_next[dones_t] = 0 # Zero-out terminal states
        expected_ovals = (self.gamma * ovals_next + rewards_t).reshape(-1,1)
        loss = nn.MSELoss()(ovals, expected_ovals)
        return loss

    def update(self, network="options"):

        self.network.optimizer.zero_grad()
        
        # losses
        if network == "meta":
            batch = self.meta_buffer.sample_batch(batch_size=self.batch_size)
            loss = self.calculate_meta_loss(batch)
            loss.backward()
            if self.network.device == 'cuda':
                self.losses['meta'].append(loss.detach().cpu().numpy())
            else:
                self.losses['meta'].append(loss.detach().numpy())
        elif network == "options":
            losses = []
            for i in range(self.network.n_options):
                batch = self.option_buffer.sample_batch(batch_size=self.batch_size)
                losses.append(self.calculate_options_loss(batch, option=i))
                if self.network.device == 'cuda':
                    self.losses[i].append(losses[i].detach().cpu().numpy())
                else:
                    self.losses[i].append(losses[i].detach().numpy())
                
            sum(losses).backward() # an average might also be possible
                
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.network.clip_val)

        self.network.optimizer.step()

    def eval_performance(self, n_val_episodes, eps):
        rewards = []
        for _ in range(n_val_episodes):
            r = float(self.play_a_game())
            rewards.append(r)
        self.mean_validation_rewards[int(eps)] = np.mean(np.array(rewards))

    def filter_reward(self, r, done, network, opt=None):
        if network == "options":
            if opt is None:
                opt = self.network.current_option
            if (opt == 0) or (opt == 1 and r != 10) or (opt == 2 and r != 50) or (opt==3 and r <= 50):
                r = 0
            if opt == 0 and done:
                r = -1
        if network == "meta" and done:
            r = -1
        return max(-1, min(1, r))
          
    def play_a_game(self):
        self.state_buffer = deque(maxlen=self.tau) # init state buffer
        init_state = preprocess(self.env.reset())
        [self.state_buffer.append(np.zeros(init_state.shape)) for i in range(4)]
        r = 0
        done = False
        step_count = 0
        while not done:
            if step_count % self.option_len == 0:
                self.take_option(mode='eval')
            action = self.network.get_action([self.state_buffer])
            state, reward, done, _ = self.env.step(action) 
            step_count += 1
            self.state_buffer.append(preprocess(state))
            r += reward
        return r

    def save_learning_curves(self, name="learning_curves", prefix=""):
        if self.checkpoint_path is None:
          return
        learning_curves = {
          "train": self.training_rewards,
          "mean_train": self.mean_training_rewards,
          "val": self.mean_validation_rewards,
          "option_ratio": self.train_ep_option_ratio,
          "losses": self.losses
        }
        path = self.checkpoint_path + name + ".pkl"
        if prefix != "":
            path = self.checkpoint_path + prefix + "_" + name + ".pkl" 
        with open(path, 'wb+') as f:
          pickle.dump(learning_curves, f, pickle.HIGHEST_PROTOCOL)

    def plot_results(self, start_from_eps=0):
        plot_rewards(self.training_rewards, self.mean_training_rewards, self.mean_validation_rewards, start_from_eps)
        plot_option_ratio(self.train_ep_option_ratio, self.op_ratio_window)