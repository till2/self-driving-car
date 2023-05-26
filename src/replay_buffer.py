import os
from operator import itemgetter

import random
from collections import deque

import torch
import numpy as np

from .utils import load_config, to_np

class ReplayBuffer():
    """ This Buffer class is used for storing preprocessed observations x
        that serve at the starting point for imagination episodes. """
    def __init__(self):

        config = load_config()
        self.buffer_size = config["buffer_size"]
        self.store_on_cpu = config["store_on_cpu"]
        self.device = config["device"]
        self.buffer = deque(maxlen=self.buffer_size)

    def push(self, observation):

        # move buffer objects to cpu to avoid GPU running full
        if self.store_on_cpu and isinstance(observation, torch.Tensor):
            observation = observation.cpu()
            
        self.buffer.append(observation)

    def sample(self):
        observation = random.choice(self.buffer)

        # move buffer object to original device (probably GPU)
        if self.store_on_cpu and isinstance(observation, torch.Tensor):
            observation = observation.to(self.device)

        return observation

    def __len__(self):
        return len(self.buffer)