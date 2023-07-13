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
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit, AutoResetWrapper

from .replay_buffer import ReplayBuffer
from .preprocessing import transform
from .utils import load_config, to_np


class ImaginationEnv(gym.vector.VectorEnv): # ImaginationEnv(gym.Env)
    """ Custom gymnasium environment for training inside the world model (RSSM). """

    def __init__(self, rssm, replay_buffer, batch_size=1, render_mode=None):

        config = load_config()
        self.batch_size = batch_size
        self.A, self.H, self.Z = itemgetter("A", "H", "Z")(config)
        self.num_categoricals = config["num_categoricals"]
        self.num_classes = config["num_classes"]
        self.max_imagination_episode_steps = config["max_imagination_episode_steps"]
        self.batch_size = batch_size
        self.device = config["device"]

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.H + self.Z,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(self.A,), dtype=np.float32)

        super().__init__(num_envs=batch_size, observation_space=obs_space, action_space=action_space)

        # define action and observation space
        # they must be gym.spaces objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.H + self.Z,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=config["action_space_low"], high=config["action_space_high"], shape=(self.A,), dtype=np.float32)

        # set a counter for truncation
        self.step_counter = 0

        # set the rssm object
        self.rssm = rssm.to(self.device)

        # set the replay buffer object
        self.replay_buffer = replay_buffer

        # save the current h internally for the next step
        self.h = None
        self.z = None

        # save images (optional)
        if render_mode not in [None, "rgb_array", "gif"]:
            raise Exception(
                f"Render mode should be None, rgb_array or gif, but got {render_mode}.")
        self.render_mode = render_mode
        if self.render_mode in ["rgb_array", "gif"]:
            self.images = []


    def reset(self, seed=42, options=None):
        """
        Resets the imagination env.
        Samples a random observation from the replay buffer and encodes the posterior z as the first state.
        Resets the step counter used for truncation.
        Internally saves the initial h and z.

        Args:
            seed (int): Seed used for the pseudo-random sampling operations
            options (None): Not used in this implementation.
        
        Returns:
            observation (torch.Tensor): Batched rssm state (with encoded posterior z)
                Shape: (IMAG_BATCHSIZE, STATE) with STATE=Z+H
            info (dict): Not used in this implementation.
        """
        np.random.seed(seed)

        # reset the step counter for episode truncation
        self.step_counter = 0

        # use a random x from the replay buffer
        x = self.replay_buffer.sample() # => (B, C, H, W)

        # reset list of saved images (and take the first image if the input is a batch)
        if self.render_mode in ["rgb_array", "gif"]:
            if len(x.shape) == 4:
                # batched
                img = (255 * to_np(x[0].permute(1, 2, 0))).astype("uint8")
            else:
                # not batched
                img = (255 * to_np(x.permute(1, 2, 0))).astype("uint8")
            self.images = [img]

        step_dict = self.rssm.pre_step(x=x)
        self.h = step_dict["h"]
        self.z = step_dict["z"]
        observation = step_dict["state"]

        info = {}
        return observation, info


    def step(self, action):
        """
        Uses the internally saved hidden state h that was output of the prev step
        to first perform a rssm.step to get h_new and then use it to get the next
        observation and head outputs with rssm.pre_step. 

        Args:
            action (torch.Tensor | np.ndarray):
                Shape: (IMAG_BATCHSIZE, A)
        
        Returns:W
            observation (torch.Tensor):
                Shape: (IMAG_BATCHSIZE, STATE) with STATE=H+Z
        """
        if self.h is None or self.z is None:
            raise Exception("Missing h: Forgot to call env.reset()")
        
        # check the action shape
        assert len(action.shape) == 2
        # assert action.shape[0] == self.batch_size
        # assert action.shape[1] == self.A
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device)
        # do later: action = action.view(self.batch_size, self.A)  # (IMAG_BATCHSIZE, A)

        # check whether the episode is too long and should be truncated
        self.step_counter += 1
        truncated = torch.tensor([self.step_counter >= self.max_imagination_episode_steps]*self.batch_size).to(self.device)

        h_new = self.rssm.step(action, self.h, self.z)
        step_dict = self.rssm.pre_step(h=h_new, return_reconstruction=True if self.render_mode in ["rgb_array", "gif"] else False)

        continue_pred = step_dict["continue_pred"]
        continue_prob = step_dict["continue_prob"]
        reward_pred = step_dict["reward_pred"]


        # predict one step using the RSSM
        terminated = (1 - continue_pred).bool()

        # weight the reward by continue_prob to account for possible episode termination
        reward = (continue_prob * reward_pred).float()

        observation = step_dict["state"]

        # update the internal h
        self.h = h_new

        # save reconstructed image (optional)
        if self.render_mode in ["rgb_array", "gif"]:
            x_pred = step_dict["x_reconstruction"]
            if len(x_pred.shape) == 4:
                img = (255 * to_np(x_pred[0].permute(1, 2, 0))).astype("uint8")
            else:
                img = (255 * to_np(x_pred.permute(1, 2, 0))).astype("uint8")
            self.images.append(img)

        info = {}
        return observation, reward, terminated, truncated, info


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
            imageio.mimsave(f"{base_path}_{index}.gif",
                            self.images, duration=0.03)

    def close(self):
        # reset hidden state
        del self.h

        # reset saved images
        if self.render_mode in ["rgb_array", "gif"]:
            del self.images


def make_imagination_env(rssm, replay_buffer, batch_size=1, render_mode=None):
    config = load_config()

    env = ImaginationEnv(
        rssm,
        replay_buffer,
        batch_size,
        render_mode
    )

    # print("Adding a TimeLimit wrapper with %d max imagination episode steps." %
    #       config["max_imagination_episode_steps"])
    # env = TimeLimit(
    #     env, max_episode_steps=config["max_imagination_episode_steps"])

    # print("Adding an AutoReset wrapper.")
    # env = AutoResetWrapper(env)

    print("Adding a RescaleActionV0 wrapper.", end=" ")
    env = RescaleActionV0(
        env, min_action=config["action_space_low"], max_action=config["action_space_high"])
    print("Low:", env.action_space.low, end=", ")
    print("High:", env.action_space.high)

    # print("Adding a Gymnasium RecordEpisodeStatistics wrapper.")
    # env = RecordEpisodeStatistics(env, deque_size=config["n_model_updates"])

    return env
