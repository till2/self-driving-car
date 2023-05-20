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

from .preprocessing import grayscale_transform as transform
from .utils import load_config, to_np


class ImaginationEnv(gym.Env):
    """ Custom gymnasium environment for training inside the world model (RSSM). """
    def __init__(self, rssm, device, max_episode_steps=50, render_mode=None):
        super(ImaginationEnv, self).__init__()
        
        config = load_config()

        self.action_clip = itemgetter("action_clip")(config)
        self.A, self.H, self.Z = itemgetter("A", "H", "Z")(config)
        self.num_categoricals, self.num_classes= itemgetter("num_categoricals", "num_classes")(config)
        
        # define action and observation space
        # they must be gym.spaces objects
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.H+self.Z,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.action_clip, high=self.action_clip, shape=(self.A,), dtype=np.float32)
        
        # set counter for truncation
        self.max_episode_steps = max_episode_steps
        self.step_counter = 0
        
        # set rssm object
        self.device = device
        self.rssm = rssm.to(self.device)
        
        # save current h and z internally for the next step
        self.h = None
        
        # save images (optional)
        if render_mode not in [None, "rgb_array", "gif"]:
            raise Exception(f"Render mode should be None, rgb_array or gif, but got {render_mode}.")
        self.render_mode = render_mode
        if self.render_mode in ["rgb_array", "gif"]:
            self.images = []
    
    def step(self, action):
        
        # check whether the episode is too long and should be truncated
        self.step_counter += 1
        truncated = True if self.step_counter >= self.max_episode_steps else False
        
        # predict z from h
        z_prior = self.rssm.dynamics_mlp(self.h).view(-1, self.num_categoricals, self.num_classes) # (1,32,32) for the softmax
        z_prior = F.softmax(z_prior, -1).flatten(start_dim=1, end_dim=2) # (1, 1024)
        z = z_prior
        
        # convert the action to a tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device).view(1,-1) # (1,3)
        
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
            self.images.append((255 * to_np(x_pred[0][0])).astype("uint8"))
        
        info = {}
        return observation, reward, terminated, truncated, info
    
    def reset(self):

        # reset the step counter for episode truncation
        self.step_counter = 0
        
        # reset h
        self.h = torch.zeros(self.rssm.num_rnn_layers, 1, self.H, device=self.device)
        
        # use a random x from the replay buffer
        initial_observation = np.load("initial_observation.npy") ### TODO: replay buffer
        x = transform(initial_observation).view(-1, 1, 128, 128).to(self.device) ### TODO: replay buffer

        z = self.rssm.vae.encode(x)
        
        observation = to_np(torch.cat((self.h.flatten(), z.flatten()), dim=0))
        
        # reset saved images
        if self.render_mode in ["rgb_array", "gif"]:
            self.images = [255 * to_np(x[0][0]).astype("uint8")]
            
        info = {}
        return observation, info

    def render(self):
        # return image
        if self.render_mode == "rgb_array":
            return self.images[-1]
        
        # save gif at last step of an episode
        if self.render_mode == "gif" and self.step_counter == self.max_episode_steps:
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