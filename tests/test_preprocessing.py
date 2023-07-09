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


class TestDiscreteActorCritic(unittest.TestCase):

    def setUp(self):
        print("Run preprocessing tests...")
        
    def test_transform(self):
        """
        Tests:
            - The shape. It should be: (B,H,W,C) => (B,C,H,W)
            - Output properties: Tensor, Device, in [0,1]
        """

        batch_sizes = [1, 5]
        channels = 3
        img_sizes = [[64,64], [20,25], [25,20]]

        for batch_size in batch_sizes:
            for img_size in img_sizes:

                input_tensor = torch.rand(batch_size, *img_size, channels)*255 # pixel values are in [0, 255]
                print(f"Testing with input shape (B,H,W,C) = {list(input_tensor.shape)}")
                result = transform(input_tensor)
                print(f"Result shape: (B,C,H,W) = {list(result.shape)}\n")

                ## Assertions:
                # result is a tensor
                self.assertIsInstance(result, torch.Tensor)

                # test output shape
                self.assertEqual(result.shape, torch.Size([batch_size, 1 if config["grayscale"] else 3, *config["size"]]))

                # result is in [0,1]
                self.assertTrue(torch.all(torch.ge(result, 0.0)) and torch.all(torch.le(result, 1.0)))

                # result is on the set device
                self.assertEqual(result.device, config["device"])


if __name__ == "__main__":
    unittest.main()