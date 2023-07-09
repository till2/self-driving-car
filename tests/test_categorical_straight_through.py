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


class TestCategoricalStraightThrough(unittest.TestCase):

    def setUp(self):
        print("Run CategoricalStraightThrough tests...")
        self.test_categorical = CategoricalStraightThrough()

    def test_forward(self):
        """
        Tests:
            - the output shapes for valid inputs (with and without batches)
            - that faulty inputs raise an AssertionError
        """
        
        valid_inputs = [
            # (Z,)
            torch.randn(32*32), # i=0
            
            # (B, Z)
            torch.randn(1, 32*32), # i=1
            torch.randn(5, 32*32), # i=2
            
            # (NUM_CATEGORICALS, NUM_CLASSES)
            torch.randn(32, 32), # i=3

            # (B, NUM_CATEGORICALS, NUM_CLASSES)
            torch.randn(1, 32, 32), # i=4
            torch.randn(5, 32, 32), # i=5
        ]

        # test the shapes for the valid inputs
        for i, valid_input in enumerate(valid_inputs):
            print("testing valid input")
            result_z, result_probs = self.test_categorical(valid_input)

            if i == 2 or i == 5:
                batch_shape = 5
            else:
                batch_shape = 1
            
            self.assertEqual(result_z.shape, torch.Size([batch_shape, config["num_categoricals"], config["num_classes"]]))
            self.assertEqual(result_probs.shape, torch.Size([batch_shape, config["num_categoricals"], config["num_classes"]]))


        faulty_inputs = [
            # (Z,)
            torch.randn(32*31),
            torch.randn(32*2),
            torch.randn(32*64),
            torch.randn(33*32),
            torch.randn(8*8),
            torch.randn(16*16),
            
            # (B, Z)
            torch.randn(1, 32*64),
            torch.randn(5, 16*32),
            
            # (NUM_CATEGORICALS, NUM_CLASSES)
            torch.randn(32, 16),
            torch.randn(64, 32),
            torch.randn(64, 64),
            torch.randn(1, 1),

            # (B, NUM_CATEGORICALS, NUM_CLASSES)
            torch.randn(1, 64, 32),
            torch.randn(1, 32, 64),
            torch.randn(5, 64, 32),
            torch.randn(5, 32, 64),
        ]

        # test that the faulty inputs fail
        for faulty_input in faulty_inputs:
            print("testing faulty input")
            with self.assertRaises(AssertionError):
                print("faulty input detected\n")
                self.test_categorical(faulty_input)


if __name__ == "__main__":
    unittest.main()