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


class TestRSSM(unittest.TestCase):

    def setUp(self):
        print("Run rssm tests...")
        # create an agent
        self.test_rssm = RSSM()
        torch.manual_seed(0)
         
    def test_pre_step(self):
        """
        Tests:
        - the output shapes (and that some items are None in inference mode) of pre_step for batches.
        - that z != z_pred in training mode (test case might break in rare cases, but shouldn't happen with the current random seed)
        - that z = z_pred in inference mode
        - that continue_prob is in [0,1]
        - that continue_pred is in {0,1}
        """
        ## test batches
        batch_sizes = [1, 7]

        for batch_size in batch_sizes:

            h = torch.randn(batch_size, config["H"]).to(config["device"])
            NUM_CHANNELS = 1 if config["grayscale"] else 3
            IMG_SIZE = config["size"]
            x = torch.randn(batch_size, NUM_CHANNELS, *IMG_SIZE).to(config["device"]) #  (B, C, H, W)

            print("Testing with input img shape:", x.shape)

            # test with only x (batch)
            # => training mode
            print("- Testing with only x")
            step_dict = self.test_rssm.pre_step(x=x)

            self.assertEqual(step_dict["state"].shape, torch.Size([batch_size, config["H"] + config["Z"]]))
            self.assertEqual(step_dict["h"].shape, torch.Size([batch_size, config["H"]]))
            self.assertEqual(step_dict["z"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["z_pred"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["z_probs"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["z_pred_probs"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["reward_pred"].shape, torch.Size([batch_size]))
            self.assertEqual(step_dict["continue_prob"].shape, torch.Size([batch_size]))
            self.assertEqual(step_dict["continue_pred"].shape, torch.Size([batch_size]))
            self.assertEqual(step_dict["x"].shape, torch.Size([batch_size, NUM_CHANNELS, *IMG_SIZE]))
            self.assertEqual(step_dict["x_reconstruction"].shape, torch.Size([batch_size, NUM_CHANNELS, *IMG_SIZE]))


            # test with only h (batch)
            # => inference mode
            print("- Testing with only h")
            step_dict = self.test_rssm.pre_step(h)

            # should be None: reward_pred, continue_prob, continue_pred, x, x_reconstruction
            # z and z_pred shoud be the same
            # z_probs and z_pred_probs should be the same
            
            # test the output shapes
            self.assertEqual(step_dict["state"].shape, torch.Size([batch_size, config["H"] + config["Z"]]))
            self.assertEqual(step_dict["h"].shape, torch.Size([batch_size, config["H"]]))
            self.assertEqual(step_dict["z"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["z_pred"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["z_probs"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["z_pred_probs"].shape, torch.Size([batch_size, config["Z"]]))

            self.assertIsNotNone(step_dict["reward_pred"])
            self.assertIsNotNone(step_dict["continue_prob"])
            self.assertIsNotNone(step_dict["continue_pred"])
            
            self.assertIsNone(step_dict["x"])
            self.assertIsNone(step_dict["x_reconstruction"])

            self.assertTrue(torch.allclose(step_dict["z"], step_dict["z_pred"]))
            self.assertTrue(torch.allclose(step_dict["z_probs"], step_dict["z_pred_probs"]))


            # test with x and h (batch)
            # => training mode
            print("- Testing with h and x")
            step_dict = self.test_rssm.pre_step(h, x)

            # z and z_prior are probably not the same (test with random seed)
            # z_probs and z_pred_probs are probably not the same (test with random seed)

            self.assertEqual(step_dict["state"].shape, torch.Size([batch_size, config["H"] + config["Z"]]))
            self.assertEqual(step_dict["h"].shape, torch.Size([batch_size, config["H"]]))
            self.assertEqual(step_dict["z"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["z_pred"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["z_probs"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["z_pred_probs"].shape, torch.Size([batch_size, config["Z"]]))
            self.assertEqual(step_dict["reward_pred"].shape, torch.Size([batch_size]))
            self.assertEqual(step_dict["continue_prob"].shape, torch.Size([batch_size]))
            self.assertEqual(step_dict["continue_pred"].shape, torch.Size([batch_size]))
            self.assertEqual(step_dict["x"].shape, torch.Size([batch_size, NUM_CHANNELS, *IMG_SIZE]))
            self.assertEqual(step_dict["x_reconstruction"].shape, torch.Size([batch_size, NUM_CHANNELS, *IMG_SIZE]))

            # these should fail with the current random seed and random network init
            # (just checking that it doesn't return z_pred as z, as well as and the same probs)
            self.assertFalse(torch.allclose(step_dict["z"], step_dict["z_pred"]))
            self.assertFalse(torch.allclose(step_dict["z_probs"], step_dict["z_pred_probs"]))

            # EXTRA TESTS
            # input: h_t, output: h_t
            self.assertTrue(torch.allclose(h, step_dict["h"]))

            # test that z and z_pred are one-hot encoded
            self.assertEqual(step_dict["z"].sum(), batch_size * 32)
            self.assertEqual(step_dict["z_pred"].sum(), batch_size * 32)

            # test that continue prob is in [0,1]
            self.assertTrue((step_dict["continue_prob"] >= 0).all().item() and (step_dict["continue_prob"] <= 1).all().item())

            # test that continue pred is in {0,1}
            self.assertTrue(torch.logical_or(step_dict["continue_pred"] == 0, step_dict["continue_pred"] == 1).all().item())


    def test_step(self):
        """
        Tests:
        - the output shapes of the step function for a few batches
        - that h_in != h_out (to test that it performs the step. 
                                h_in=h_out could be True in rare cases, but not with the current random seed)
        """
        
        ## test batches
        batch_sizes = [1, 7]

        for batch_size in batch_sizes:

            action = torch.randn(batch_size, config["A"]).to(config["device"])
            h = torch.randn(batch_size, config["H"]).to(config["device"])
            z = torch.randn(batch_size, config["Z"]).to(config["device"])

            h_out = self.test_rssm.step(action, h, z)

            # Test that h_in != h_out
            self.assertFalse(torch.allclose(h, h_out))

            # Test the shape of h_out. Should be (B, H)
            self.assertEqual(h.shape, h_out.shape)
            self.assertEqual(h_out.shape, torch.Size([batch_size, config["H"]]))

    def test_get_losses(self):
        """
        Tests:
            - that all losses are scalars (meaning they don't have a shape)
        """
        # Create a step_dict for testing
        batch_size = 8
        h = torch.randn(batch_size, config["H"]).to(config["device"])
        NUM_CHANNELS = 1 if config["grayscale"] else 3
        IMG_SIZE = config["size"]
        x = torch.randn(batch_size, NUM_CHANNELS, *IMG_SIZE).to(config["device"]) #  (B, C, H, W)
        
        self.step_dict = self.test_rssm.pre_step(h, x) # careful. dependency on the pre_step function here.

        terminated = np.random.choice([True,False], size=batch_size, replace=True)
        truncated = np.random.choice([True,False], size=batch_size, replace=True)
        reward = np.random.rand(batch_size) - 0.5

        self.step_dict["continue_target"] = torch.tensor(1 - (terminated | truncated), device=config["device"], dtype=torch.float32)
        self.step_dict["reward_target"] = torch.tensor(reward, device=config["device"], dtype=torch.float32)

        # continue_target shape: (B,)
        # reward_target shape: (B,)

        self.losses = losses = self.test_rssm.get_losses(self.step_dict)

        self.assertEqual(losses["loss"].shape, torch.Size([]))
        self.assertEqual(losses["image_loss"].shape, torch.Size([]))
        self.assertEqual(losses["reward_loss"].shape, torch.Size([]))
        self.assertEqual(losses["continue_loss"].shape, torch.Size([]))
        self.assertEqual(losses["dyn_loss"].shape, torch.Size([]))
        self.assertEqual(losses["rep_loss"].shape, torch.Size([]))

    def test_update_parameters(self):
        """
        Test that all parameters of the RSSM have a gradient after two steps (so that the GRU also contributes to the loss).
        """

        # Create a step_dict for testing
        batch_size = 8
        h = torch.randn(batch_size, config["H"]).to(config["device"])
        NUM_CHANNELS = 1 if config["grayscale"] else 3
        IMG_SIZE = config["size"]
        x = torch.randn(batch_size, NUM_CHANNELS, *IMG_SIZE).to(config["device"]) #  (B, C, H, W)
        
        # first pre_step
        step_dict = self.test_rssm.pre_step(h, x)

        # first step
        action = torch.randn(batch_size, config["A"]).to(config["device"])
        h = step_dict["h"]
        z = step_dict["z"]
        h = self.test_rssm.step(action, h, z)

        # second pre_step
        step_dict = self.test_rssm.pre_step(h, x)

        terminated = np.random.choice([True,False], size=batch_size, replace=True)
        truncated = np.random.choice([True,False], size=batch_size, replace=True)
        reward = np.random.rand(batch_size) - 0.5

        step_dict["continue_target"] = torch.tensor(1 - (terminated | truncated), device=config["device"], dtype=torch.float32)
        step_dict["reward_target"] = torch.tensor(reward, device=config["device"], dtype=torch.float32)

        losses = self.test_rssm.get_losses(step_dict)

        self.test_rssm.update_parameters(losses["loss"])

        # If the test fails - for debugging: print the params where the gradient is None
        for name, param in self.test_rssm.named_parameters():
            if param.grad is None:
                print(f"Parameter '{name}' has None gradient.")

        # Test that all parameters have a gradient
        for param in self.test_rssm.parameters():
            self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()