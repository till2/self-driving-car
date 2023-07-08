from __future__ import annotations
from typing import Dict, List, Union

import unittest

import logging
import os
import random
import sys
from collections import deque
from operator import itemgetter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

import gymnasium as gym

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

# Load the config
config = load_config()
for key in config:
    locals()[key] = config[key]


class TestTwoHotEncoding(unittest.TestCase):

    def setUp(self):
        print("Run two-hot encoding tests...")

    def test_critic(self):
        """
        Tests:
            - the symlog two-hot encodings for a few examples (for scalar and vector inputs)
            - that the function is reversible with symexp (for scalar and vector inputs)
        """
        
        # test the encoding for scalar example 1/2
        solution = torch.zeros(1,255).to(config["device"])
        solution[0, 74] = 0.6395
        solution[0, 75] = 0.3605
        self.assertEqual(torch.isclose(twohot_encode(-500.33), solution, atol=1e-4).all().item(), True)

        # test the encoding for scalar example 2/2
        solution = torch.zeros(1,255).to(config["device"])
        solution[0, 158] = 0.1552
        solution[0, 159] = 0.8448
        self.assertEqual(torch.isclose(twohot_encode(42), solution, atol=1e-4).all().item(), True)

        # test the shape of a vector encoding
        self.assertEqual(twohot_encode(42.0).shape, torch.Size([1, config["num_buckets"]])) # (1,255)
        self.assertEqual(twohot_encode([42.0]).shape, torch.Size([1, config["num_buckets"]])) # (1,255)
        self.assertEqual(twohot_encode([-500.33, 42, 1, 2]).shape, torch.Size([4, config["num_buckets"]])) # (4,255)

        # test whether the encoding is invertible (with symexp and dot-product with buckets) for a scalar example
        buckets = torch.linspace(config["min_bucket"], config["max_bucket"], config["num_buckets"]).to(config["device"])
        result = symexp(twohot_encode(0.01) @ buckets)
        solution = torch.tensor([0.01]).to(config["device"])
        self.assertEqual(torch.isclose(result, solution).all().item(), True)

        # test whether the encoding is invertible (with symexp and dot-product with buckets) for a vector example
        buckets = torch.linspace(config["min_bucket"], config["max_bucket"], config["num_buckets"]).to(config["device"])
        result = symexp(twohot_encode([-100_000.1, 10.25, 0.42, 500.33]) @ buckets)
        solution = torch.tensor([-100_000.1, 10.25, 0.42, 500.33]).to(config["device"])
        self.assertEqual(torch.isclose(result, solution).all().item(), True)

if __name__ == "__main__":
    unittest.main()