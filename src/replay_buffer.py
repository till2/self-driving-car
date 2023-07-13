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
    def __init__(self, buffer_size=None):

        config = load_config()
        self.buffer_size = config["buffer_size"] if buffer_size is None else buffer_size
        self.store_on_cpu = config["store_on_cpu"]
        self.device = config["device"]
        self.buffer = deque(maxlen=self.buffer_size)

    def push(self, observation):

        # move buffer objects to cpu to avoid GPU running full
        if self.store_on_cpu and isinstance(observation, torch.Tensor):
            observation = observation.cpu()
            
        self.buffer.append(observation)

    def sample(self, n=1):
        # assert that the buffer holds at least one observation
        assert len(self) > 0

        if n == 1:
            observation = random.choice(self.buffer)
        else:
            observation = torch.stack(random.sample(self.buffer, size=n, replace=True))

        # move buffer object to original device (probably GPU)

        if self.store_on_cpu and isinstance(observation, torch.Tensor):
            observation = observation.to(self.device)

        return observation

    def __len__(self):
        return len(self.buffer)