import numpy as np
from collections import deque, namedtuple

class experienceReplayBuffer:
  """
  class also contains functions for appending values to the buffer
  and sampling it randomly. The burn_in is used to initialize our memory.
  source: https://www.datahubbs.com/deep-q-learning-101/
  """

  def __init__(self, memory_size=1e4, burn_in=5e3):
    # colab can only hand 1e4, original was 1e6, original burn_in:5e5
    self.memory_size = memory_size
    self.burn_in = min(memory_size, burn_in)
    self.Buffer = namedtuple('Buffer', 
      field_names=['state', 'action', 'reward', 'done', 'next_state'])
    self.replay_memory = deque(maxlen=int(memory_size))

  def sample_batch(self, batch_size=32):
    samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
    # Use asterisk operator to unpack deque 
    batch = zip(*[self.replay_memory[i] for i in samples])
    return batch

  def append(self, state, action, reward, done, next_state):
    self.replay_memory.append(self.Buffer(state, action, reward, done, next_state))

  def burn_in_capacity(self):
    return len(self.replay_memory) / self.burn_in

def preprocess(img, gray=True, H=84, W=84):
  if gray:
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114]) # grayscale
    img = np.resize(img, (H, W)) # resize
    return img/255 # normalize
