import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import io
import base64
import pickle 

# gym
import gym
from gym import logger as gymlogger
# gymlogger.set_level(40) #error only

# Local .py files
from DQN_utils import *
from HDQN import *
from HDQN_agent import *
from DQN import *
from DQN_agent import *

# GPU config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

CHECKPOINT_FOLDERNAME = "checkpoints/"
CHECKPOINT_PREFIX="vanilla_dqn"
MEMORY_SIZE = 1e4
BATCH_SIZE = 32
BURN_IN = 1e5
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
MAX_EPISODES = 5000
INITIAL_EPSILON = 1; FINAL_EPSILON=0.1; FINAL_EPSILON_EPISODE=1e3
NETWORK_SYNC_FREQ = 10000
NETWORK_UPDATE_FREQ = 4
NETWORK_SAVE_FREQ = 100
NETWORK_EVALUATE_FREQ = 1000
N_VAL_EPISODES = 10

# Vanilla
env = gym.make("MsPacman-v0")
D = experienceReplayBuffer(memory_size=MEMORY_SIZE)
dqn = DQN(
    env, 
    learning_rate=LEARNING_RATE,
    device=device, 
    checkpoint_path=CHECKPOINT_FOLDERNAME
)
agent = DQN_agent(env, dqn, D)
agent.train(
    gamma=GAMMA,
    max_episodes= MAX_EPISODES,
    batch_size=BATCH_SIZE,
    network_update_frequency = NETWORK_UPDATE_FREQ,
    network_sync_frequency = NETWORK_SYNC_FREQ,
    network_save_frequency = NETWORK_SAVE_FREQ,
    network_evaluate_frequency=NETWORK_EVALUATE_FREQ,
    n_val_episodes=N_VAL_EPISODES,
    checkpoint_path = CHECKPOINT_FOLDERNAME,
    checkpoint_prefix = CHECKPOINT_PREFIX,
    epsilon_start=INITIAL_EPSILON,
    epsilon_end=FINAL_EPSILON,
    epsilon_final_episode=FINAL_EPSILON_EPISODE
)
