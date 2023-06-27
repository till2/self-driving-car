import os
from operator import itemgetter

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from PIL import Image

from gymnasium.experimental.wrappers import RescaleActionV0
from gymnasium.wrappers import TimeLimit, AutoResetWrapper

from .replay_buffer import ReplayBuffer
from .preprocessing import transform
from .utils import load_config, to_np


class ImaginationEnv(gym.Env):
    """ Custom gymnasium environment for training inside the world model (RSSM). """
    def __init__(self, rssm, replay_buffer, render_mode=None):
        super(ImaginationEnv, self).__init__()
        
        config = load_config()

        self.A, self.H, self.Z = itemgetter("A", "H", "Z")(config)
        self.num_categoricals = config["num_categoricals"]
        self.num_classes = config["num_classes"]
        self.max_imagination_episode_steps = config["max_imagination_episode_steps"]
        self.device = config["device"]

        # define action and observation space
        # they must be gym.spaces objects
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.H + self.Z,), dtype=np.float32)
        self.action_space = spaces.Box(low=config["action_space_low"], high=config["action_space_high"], shape=(self.A,), dtype=np.float32)
        
        # set a counter for truncation
        self.step_counter = 0
        
        # set the rssm object
        self.rssm = rssm.to(self.device)

        # set the replay buffer object
        self.replay_buffer = replay_buffer
        
        # save the current h internally for the next step
        self.h = None
        
        # save images (optional)
        if render_mode not in [None, "rgb_array", "gif"]:
            raise Exception(f"Render mode should be None, rgb_array or gif, but got {render_mode}.")
        self.render_mode = render_mode
        if self.render_mode in ["rgb_array", "gif"]:
            self.images = []
    
    def step(self, action):

        if self.h is None:
            raise Exception("Missing h: Forgot to call env.reset()")
        
        # check whether the episode is too long and should be truncated
        self.step_counter += 1
        truncated = True if self.step_counter >= self.max_imagination_episode_steps else False
        
        # predict z from h
        z_prior = self.rssm.dynamics_mlp(self.h).view(-1, self.num_categoricals, self.num_classes) # (1,32,32) for the softmax
        z_prior = F.softmax(z_prior, -1).flatten(start_dim=1, end_dim=2) # (1, 1024)
        z = z_prior
        
        # convert the action to a tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device).view(1, self.A) # (1,A)
        
        # predict one step using the RSSM
        h_new, reward_pred, continue_prob, continue_pred, x_pred = self.rssm.step(action, self.h, z)
        terminated = bool(1 - continue_pred)
        
        # weight the reward by continue_prob to account for possible episode termination
        reward = float(continue_prob * reward_pred)
        
        # combine h and z to get the new state (observation)
        observation = to_np(torch.cat((self.h.flatten(), z.flatten()), dim=0))
        
        # update the internal h
        self.h = h_new
        
        # save reconstructed image (optional)
        if self.render_mode in ["rgb_array", "gif"]:
            if len(x_pred.shape) == 4:
                img = (255 * to_np(x_pred[0].permute(1,2,0))).astype("uint8")
            else:
                img = (255 * to_np(x_pred.permute(1,2,0))).astype("uint8")    
            self.images.append(img)
        
        info = {}
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=0, options=None):

        np.random.seed(seed)

        # reset the step counter for episode truncation
        self.step_counter = 0
        
        # reset h
        self.h = torch.zeros(self.rssm.num_rnn_layers, 1, self.H, device=self.device)
        
        # use a random x from the replay buffer
        x = self.replay_buffer.sample()

        # reset list of saved images (and take the first image if the input is a batch)

        if self.render_mode in ["rgb_array", "gif"]:
            if len(x.shape) == 4:
                img = (255 * to_np(x[0].permute(1,2,0))).astype("uint8")
            else:
                img = (255 * to_np(x.permute(1,2,0))).astype("uint8")
            self.images = [img]

        z = self.rssm.vae.encode(x)
        
        observation = to_np(torch.cat((self.h.flatten(), z.flatten()), dim=0))
            
        info = {}
        return observation, info

    def render(self):
        # return image
        if self.render_mode == "rgb_array":
            return self.images[-1]
        
        # save gif at last step of an episode
        if self.render_mode == "gif" and self.step_counter == self.max_imagination_episode_steps:
            base_path = "reconstructions/imagined_episodes/imagination_ep"
            index = 0
            while os.path.exists(f"{base_path}_{index}.gif"):
                index += 1
            imageio.mimsave(f"{base_path}_{index}.gif", self.images, duration=0.03)
    
    def close(self):
        # reset hidden state
        del self.h
        
        # reset saved images
        if self.render_mode in ["rgb_array", "gif"]:
            del self.images


def make_imagination_env(rssm, replay_buffer, render_mode):
    config = load_config()

    env = ImaginationEnv(
        rssm,
        replay_buffer,
        render_mode
    )
    print("Adding a TimeLimit wrapper with %d max imagination episode steps." % config["max_imagination_episode_steps"])
    env = TimeLimit(env, max_episode_steps=config["max_imagination_episode_steps"])

    print("Adding an AutoReset wrapper.")
    env = AutoResetWrapper(env)

    print("Adding a RescaleActionV0 wrapper.", end=" ")
    env = RescaleActionV0(env, min_action=config["action_space_low"], max_action=config["action_space_high"])
    print("Low:", env.action_space.low, end=", ")
    print("High:", env.action_space.high)

    return env