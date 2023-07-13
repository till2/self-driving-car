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
        print("Run discrete actor critic tests...")
        # create an agent
        self.test_agent = DiscreteActorCritic()

    def test_critic(self):
        """
        Tests:
        - output shapes
        - that the critic softmax distributions sum to 1
        """
        sample_instance = torch.randn(config["H"] + config["Z"]).to(config["device"])  # torch.Size([1536])
        sample_batch = torch.randn(32, config["H"] + config["Z"]).to(config["device"]) # torch.Size([32, 1536])

        # test for a single observation as input
        value_pred, critic_dist = self.test_agent.apply_critic(sample_instance)
        self.assertEqual(value_pred.shape, torch.Size([1])) # value_pred should have shape (batch_size,)
        self.assertEqual(critic_dist.shape, torch.Size([1, config["num_buckets"]])) # critic_dist should have shape (batch_size, num_buckets)
        
        # test for the single instance that the softmax sum is 1
        softmax_sum = critic_dist.sum(dim=1)
        expected_sum = torch.ones([1]).to(config["device"])
        self.assertTrue(torch.allclose(softmax_sum, expected_sum))

        # test for a batch of observations as input
        value_pred, critic_dist = self.test_agent.apply_critic(sample_batch)
        self.assertEqual(value_pred.shape, torch.Size([32])) # value_pred should have shape (batch_size,)
        self.assertEqual(critic_dist.shape, torch.Size([32, config["num_buckets"]])) # critic_dist should have shape (batch_size, num_buckets)

        # test for the batch of observations that the softmax sum is 1
        softmax_sum = critic_dist.sum(dim=1)
        expected_sum = torch.ones([32]).to(config["device"])
        self.assertTrue(torch.allclose(softmax_sum, expected_sum))

        # test for a second batch of observations as input (to see whether the action_ema can switch batch sizes)
        sample_batch = torch.randn(5, config["H"] + config["Z"]).to(config["device"]) # torch.Size([5, 1536])

        value_pred, critic_dist = self.test_agent.apply_critic(sample_batch)
        self.assertEqual(value_pred.shape, torch.Size([5])) # value_pred should have shape (batch_size,)
        self.assertEqual(critic_dist.shape, torch.Size([5, config["num_buckets"]])) # critic_dist should have shape (batch_size, num_buckets)

        # test for the batch of observations that the softmax sum is 1 (to see whether the action_ema can switch batch sizes)
        softmax_sum = critic_dist.sum(dim=1)
        expected_sum = torch.ones([5]).to(config["device"])
        self.assertTrue(torch.allclose(softmax_sum, expected_sum))

    def test_actor(self):
        """
        Tests:
            - output shapes
            - that all logprobs are <= 0 (because log(1) = 0)
        """
        sample_instance = torch.randn(config["H"] + config["Z"]).to(config["device"]) # torch.Size([1536])
        sample_batch = torch.randn(32, config["H"] + config["Z"]).to(config["device"]) # torch.Size([32, 1536])

        # test for a single observation as input
        action, log_prob, actor_entropy = self.test_agent.get_action(sample_instance)
        self.assertEqual(action.shape, torch.Size([1, config["A"]])) # action should have shape (batch_size, A)
        self.assertEqual(log_prob.shape, torch.Size([1])) # log_prob should have shape (batch_size,)
        self.assertEqual(actor_entropy.shape, torch.Size([1])) # actor_entropy should have shape (batch_size,)
        self.assertEqual(torch.all(log_prob <= 0).item(), True)

        # test for a batch of observations as input
        action, log_prob, actor_entropy = self.test_agent.get_action(sample_batch)
        self.assertEqual(action.shape, torch.Size([32, config["A"]])) # action should have shape (batch_size, A)
        self.assertEqual(log_prob.shape, torch.Size([32])) # log_prob should have shape (batch_size,)
        self.assertEqual(actor_entropy.shape, torch.Size([32])) # actor_entropy should have shape (batch_size,)
        self.assertEqual(torch.all(log_prob <= 0).item(), True)

    def test_calculate_critic_loss(self):

        # set up a twohot_returns sample and batch_critic_dists sample
        twohot_returns = torch.zeros(16, 1, 255).to(config["device"])
        for i in range(16):
            twohot_returns[i,0,30] = 0.83
            twohot_returns[i,0,31] = 0.17
        torch.manual_seed(0)
        batch_critic_dists = torch.rand(16, 1, 255).to(config["device"])
        batch_critic_dists = F.softmax(batch_critic_dists, dim=2)

        result = self.test_agent._calculate_critic_loss(twohot_returns, batch_critic_dists)
        if config["loss_calculation"] == "gae":
            self.assertEqual(torch.isclose(result, torch.tensor(89.5558)).item(), True)

        # batch of samples
        twohot_returns = torch.zeros(16, 7, 255).to(config["device"])
        for i in range(16):
            twohot_returns[i,:,30] = 0.83 * torch.ones(7).to(config["device"])
            twohot_returns[i,:,31] = 0.17 * torch.ones(7).to(config["device"])
        batch_critic_dists = torch.ones(16, 7, 255).to(config["device"])
        batch_critic_dists = F.softmax(batch_critic_dists, dim=2)

        result = self.test_agent._calculate_critic_loss(twohot_returns, batch_critic_dists)
        if config["loss_calculation"] == "gae":
            self.assertEqual(torch.isclose(result, torch.tensor(88.6602)).item(), True)

        # test the shape
        self.assertEqual(result.shape, torch.Size([])) # should be a scalar (without shape)
    
    def test_get_loss(self):

        # test outputs for an example batch
        torch.manual_seed(42)
        episode_batches = dict()
        episode_batches["rewards"] = torch.randn(100, 16).to(config["device"])
        episode_batches["log_probs"] = - torch.randn(100, 16).to(config["device"])
        episode_batches["value_preds"] = torch.randn(100, 16).to(config["device"])
        episode_batches["critic_dists"] = torch.randn(100, 16, 255).to(config["device"])
        episode_batches["critic_dists"] = F.softmax(episode_batches["critic_dists"], dim=2)
        episode_batches["entropies"] = torch.randn(100, 16).to(config["device"])
        episode_batches["masks"] = torch.randn(100, 16).to(config["device"])
        episode_batches["masks"] = torch.where(episode_batches["masks"] > 0, 0, 1)

        last_value_pred = torch.randn(16).to(config["device"])

        critic_loss, actor_loss = self.test_agent.get_loss(episode_batches, last_value_pred)
        if config["loss_calculation"] == "gae":
            self.assertEqual(torch.allclose(critic_loss, torch.tensor(605.5697), atol=1e-4), True)
            self.assertEqual(torch.allclose(actor_loss, torch.tensor(0.0111), atol=1e-4), True)

        # test output shapes
        self.assertEqual(critic_loss.shape, torch.Size([])) # should be a scalar (without shape)
        self.assertEqual(actor_loss.shape, torch.Size([])) # should be a scalar (without shape)

if __name__ == "__main__":
    unittest.main()