from __future__ import annotations
from typing import Dict, List, Union

import unittest

import sys
sys.path.append("/home/till/Desktop/GitHub/self-driving-car/")
sys.path.append("/home/till/Desktop/GitHub/self-driving-car/src")

import logging
import os
import random
import sys
from collections import deque
from operator import itemgetter

print(sys.version)

import gym_donkeycar
import gymnasium as gym
import imageio
import ipywidgets as widgets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from IPython.display import display
from ipywidgets import HBox, VBox
from matplotlib import pyplot as plt
from PIL import Image
from ruamel.yaml import YAML
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from tensorboard import notebook
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch import distributions as dist
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import gym.spaces as gym_spaces
import gymnasium as gym  # overwrite OpenAI gym

# suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box") # module="gymnasium"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["IMAGEIO_IGNORE_WARNINGS"] = "True"

import stable_baselines3 as sb3
from gym_donkeycar.envs.donkey_env import DonkeyEnv
from gymnasium import spaces
from gymnasium.spaces import Box
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import src
from src.actor_critic_discrete import DiscreteActorCritic
from src.actor_critic_dreamer import ActorCriticDreamer
from src.actor_critic import ContinuousActorCritic
from src.blocks import CategoricalStraightThrough, ConvBlock
from src.categorical_vae import CategoricalVAE
from src.imagination_env import make_imagination_env
from src.mlp import MLP
from src.preprocessing import transform
from src.replay_buffer import ReplayBuffer
from src.rssm import RSSM
from src.utils import (load_config, make_env, save_image_and_reconstruction,
                       to_np, symlog, symexp, twohot_encode, ExponentialMovingAvg,
                       ActionExponentialMovingAvg, MetricsTracker)
from src.vae import VAE


torch.cuda.empty_cache()
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

# Load the config
config = load_config()
for key in config:
    locals()[key] = config[key]



# Example test class:
#
#class TestAddition(unittest.TestCase):
#    
#    def addition(self,x,y):
#        return x+y
#    
#    def test_addition(self):
#        self.assertEqual(self.addition(1,2), 3)


class TestDiscreteActorCritic(unittest.TestCase):

    def setUp(self):
        # create an agent
        self.test_agent = src.actor_critic_discrete.DiscreteActorCritic()
         
    def test_critic(self):
        """
        Tests:
        - output shapes
        - that the critic softmax dist sums to 1
        """
        
        #
        # test for a single instance
        #
        sample_instance = torch.randn(config["H"] + config["Z"]).to(config["device"]) # torch.Size([1536])
        value_pred, critic_dist = self.test_agent.apply_critic(sample_instance)

        # value_pred should be a scalar without shape
        self.assertEqual(value_pred.shape, torch.Size([]))

        # critic_dist should have shape (num_buckets)
        self.assertEqual(critic_dist.shape, torch.Size([config["num_buckets"]]))

        #
        # test for a batch
        #
        sample_batch = torch.randn(32, config["H"] + config["Z"]).to(config["device"]) # torch.Size([32, 1536])
        value_pred, critic_dist = self.test_agent.apply_critic(sample_batch)

        # value_pred should have shape (batch_size)
        self.assertEqual(value_pred.shape, torch.Size([32]))

        # critic_dist should have shape (batch_size, num_buckets)
        self.assertEqual(critic_dist.shape, torch.Size([32, config["num_buckets"]]))

        #
        # test that the softmax sum is 1 for every instance in the batch
        #
        softmax_sum = critic_dist.sum(dim=-1)
        expected_sum = torch.ones(32).to(config["device"])
        self.assertTrue(torch.allclose(softmax_sum, expected_sum))


if __name__ == "__main__":
    unittest.main()