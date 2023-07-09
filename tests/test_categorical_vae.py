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


class TestCategoricalVAE(unittest.TestCase):

    def setUp(self):
        print("Run CategoricalVAE tests...")
        self.test_vae = CategoricalVAE()


    def test_encode(self):
        """
        Tests:
            - The output shapes.
        """
        batch_size = 5
        img_size = config["size"]
        channels = 1 if config["grayscale"] else 3

        input_tensor = torch.rand(batch_size, channels, *img_size)*255 # pixel values are in [0, 255]
        input_tensor = input_tensor.to(config["device"])

        z, z_probs = self.test_vae.encode(input_tensor)

        # Test output shapes
        self.assertEqual(z.shape, torch.Size([batch_size, self.test_vae.num_categoricals, self.test_vae.num_classes]))
        self.assertEqual(z_probs.shape, torch.Size([batch_size, self.test_vae.num_categoricals, self.test_vae.num_classes]))


    def test_decode(self):
        """
        Tests:
            - The output shape.
        """
        batch_size = 5
        img_size = config["size"]
        channels = 1 if config["grayscale"] else 3

        h = torch.randn(batch_size, config["H"]).to(config["device"])
        z_flat = torch.randn(batch_size, config["Z"]).to(config["device"])

        x_hat = self.test_vae.decode(h, z_flat)

        # Test output shape
        self.assertEqual(x_hat.shape, (batch_size, channels, *img_size))


if __name__ == "__main__":
    unittest.main()