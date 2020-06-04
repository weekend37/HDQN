import numpy as np
import matplotlib.pyplot as plt
import glob
import io
import base64
import sys
from collections import namedtuple, deque

from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
import logging

from DQN_utils import preprocess

import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

def show_video():
  """
  Utility functions to enable video recording of gym environment and displaying it
  To enable video, just do "env = wrap_env(env)""
  source: https://colab.research.google.com/drive/
  1tug_bpg8RwrFOI8C6Ed-zo0OgD3yfnWy#scrollTo=JWNVK4NUJUCl
  """
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
              </video>'''.format(encoded.decode('ascii'))))
  else: 
      print("Could not find video")
    

def wrap_env(env, video=None):
  return Monitor(env, './video', force=True, video_callable=video)

def play_with_network(env, network, option_len = 0, show_vid=True):
  options_taken = {}
  display = Display(visible=0, size=(1400, 900))
  display.start()
  env = wrap_env(env)
  state_buffer = deque(maxlen=4) # init state buffer
  state = preprocess(env.reset())
  [state_buffer.append(np.zeros(state.shape)) for i in range(4)]
  state_buffer.append(state)

  done = False
  i,r = 0, 0
  while not done:
    sys.stdout.write("\r Playing frame %i" % i)
    if option_len != 0 and i % option_len == 0:
      option = network.get_option([state_buffer])
      network.current_option = option
      options_taken[i] = option
    action = network.get_action([state_buffer])
    env.render()
    state, reward, done, info = env.step(action) 
    state_buffer.append(preprocess(state))
    r += reward
    i += 1
              
  env.close()

  if show_vid:
    show_video()

  if option_len != 0:
    """
    Displays options taken for one game
    """
    palette = np.array(['lightblue', 'orange', 'darkgreen', 'red'])
    op_ratio_data = sorted(options_taken.items())
    op_ep, op = zip(*op_ratio_data)
    placeholder = np.ones(len(op))
    plt.bar(op_ep, placeholder, color=np.take(palette, op), align="edge", alpha=0.75, width=option_len)
    plt.xlabel("Frame")
    plt.tick_params(axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) #
    plt.show()
    
  return r, options_taken

def plot_rewards(training_rewards=None, mean_training_rewards=None, mean_validation_rewards=None, start_from_eps=0):
    x_train = range(start_from_eps, start_from_eps+len(training_rewards))
    plt.plot(x_train, training_rewards, label="Train", alpha=0.5)
    plt.plot(x_train, mean_training_rewards, label="Train (Moving Average)")
    # validation
    val_data = sorted(mean_validation_rewards.items()) 
    x_val, y_val = zip(*val_data)
    plt.plot(x_val, y_val, label="Validation")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward per Episode")
    plt.show()

def plot_option_ratio(train_ep_option_ratio, window):
  """
  Displays option distribution as a function of training episodes 
  """
  n_options = len(train_ep_option_ratio)
  n_op_ratio = len(train_ep_option_ratio[0].items())
  if n_op_ratio > 0:
    bottom = np.zeros(n_op_ratio)
    for o in range(n_options):
      op_ratio_data = sorted(train_ep_option_ratio[o].items())
      op_ep, op_ratio = zip(*op_ratio_data)
      plt.bar(op_ep, op_ratio, label="option "+str(o), alpha=0.75, 
              width=window, align='edge', bottom=bottom)
      bottom += op_ratio
  plt.legend()
  plt.xlabel("Episodes")
  plt.ylabel("Option Ratio")
  plt.show()
