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
LEARNING_RATE = 2.5e-8
EPSILON = 0.1
MAX_EPISODES = 5 #5000
NETWORK_EVALUATE_FREQ = 100
NETWORK_SAVE_FREQ = 100
N_VAL_EPISODES = 10
META_BURN_IN_EP = 0

env = gym.make("MsPacman-v0")
D = experienceReplayBuffer(memory_size=1e4)
hdqn = HDQN(
    env, 
    learning_rate = LEARNING_RATE,
    device = device, 
    checkpoint_path = CHECKPOINT_FOLDERNAME
)
agent = HDQN_agent(env, hdqn, D, epsilon=EPSILON)
agent.train(
    max_episodes=MAX_EPISODES, 
    network_evaluate_frequency=NETWORK_EVALUATE_FREQ,
    n_val_episodes=N_VAL_EPISODES,
    meta_burn_in_ep=META_BURN_IN_EP, 
    network_save_frequency=NETWORK_SAVE_FREQ, 
    checkpoint_path=CHECKPOINT_FOLDERNAME
)
