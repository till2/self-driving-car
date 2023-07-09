import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from ruamel.yaml import YAML

import logging
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
from gymnasium.wrappers import TimeLimit, AutoResetWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from typing import Dict, List, Union

from torch.utils.tensorboard import SummaryWriter

def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def symlog(r):
    # note: log1p = log(1+x)
    if isinstance(r, torch.Tensor):
        return torch.sign(r) * torch.log1p(torch.absolute(r))
    else:
        return np.sign(r) * np.log1p(np.absolute(r))


def symexp(r):
    if isinstance(r, torch.Tensor):
        return torch.sign(r) * (torch.exp(torch.absolute(r)) - 1)
    else:
        return np.sign(r) * (np.exp(np.absolute(r)) - 1)


def twohot_encode(x):
    """
    Computes the twohot encoded symlog returns.
    
    Args:
        x (torch.Tensor): the original returns (scalar or vector of values) to be encoded
            Shape: (B,)
    
    Returns:
        encoded (torch.Tensor): the encoded symlog returns as a vector
            Shape: (B, NUM_BUCKETS)
    
    Notes:
        - note that the returns are automatically symlogged before the two-hot encoding
    """
    config = load_config()
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x = x.to(config["device"])
    
    # add an empty dim for scalar inputs
    if not len(x.shape):
        x = x.view(-1)
        
    buckets = torch.linspace(config["min_bucket"], config["max_bucket"], config["num_buckets"]).to(config["device"])
    
    # symlog transform the values
    values = symlog(x)

    # clip values at the borders to ensure they lie within the bucket range
    clipped_values = torch.clamp(values, buckets[0], buckets[-1])

    # calculate the index of the right bucket (k+1) and left bucket (k) for each value
    index = torch.searchsorted(buckets, clipped_values, side="right")
    left_bucket_index = torch.clamp(index - 1, 0, buckets.shape[0] - 1)
    right_bucket_index = torch.clamp(index, 0, buckets.shape[0] - 1)

    # calculate the weights
    weight = (buckets[right_bucket_index] - clipped_values) / (buckets[right_bucket_index] - buckets[left_bucket_index])
    weight = torch.clamp(weight, 0.0, 1.0)  # Ensure weights are within [0, 1]

    # create the one-hot encoding for each value
    encoded = torch.zeros((x.shape[0], buckets.shape[0])).to(config["device"])
    weight = weight.to(encoded.dtype)
    encoded.scatter_add_(1, left_bucket_index.unsqueeze(1), weight.unsqueeze(1))
    encoded.scatter_add_(1, right_bucket_index.unsqueeze(1), 1 - weight.unsqueeze(1))

    return encoded



def load_config():
    yaml = YAML(typ='safe')
    # file_path = os.path.expanduser('~/Desktop/self-driving-car/src/config.yaml')
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(parent_dir, "src/config.yaml")
    with open(file_path, "r") as file:
        static_params = yaml.load(file)

    # Set dependent variables
    device = torch.device("cuda:0" if torch.cuda.is_available() and static_params["device"] == "cuda:0" else "cpu")
    A = 3 if static_params["toy_env"] else 2
    Z = static_params["num_categoricals"] * static_params["num_classes"]

    config = {
        "device": device,
        "A": A,
        "Z": Z
    }

    for key in static_params:
        if key in config:
            continue
        config[key] = static_params[key]

    return config


class ExponentialMovingAvg():
    
    def __init__(self):
        config = load_config()
        self.quantile_05 = torch.tensor(0.0).to(config["device"])
        self.quantile_95 = torch.tensor(0.0).to(config["device"])
        self.decay = torch.tensor(config["moving_avg_decay_rate"]).to(config["device"]) # 0.99
        self.device = config["device"]
            
    def scale_batch(self, batch):
        """ Returns the scaled batch and updates the moving averages. """
        if not isinstance(batch, torch.Tensor):
            batch = torch.Tensor(batch).to(self.device)
        batch = torch.flatten(batch.detach())
        
        # calculate and update the 05 and 95 quantiles
        qs = torch.quantile(batch, q=torch.tensor([0.05, 0.95]).to(dtype=batch.dtype).to(self.device))
        self.quantile_05 = self.decay * self.quantile_05 + (1-self.decay) * qs[0]
        self.quantile_95 = self.decay * self.quantile_95 + (1-self.decay) * qs[1]
        
        moving_scale = torch.max(torch.tensor(1.0), self.quantile_95 - self.quantile_05)

        scaled_batch = batch / moving_scale.detach()
        return scaled_batch


class ActionExponentialMovingAvg():
    
    def __init__(self):
        config = load_config()
        self.decay = torch.tensor(config["action_mov_avg_decay"]).to(config["device"]) # fraction of prev action that stays
        self.action = None
        self.device = config["device"]
    
    def reset(self):
        """ Resets the EMA action. 
            Calling this method is required when the batch size changes (e.g. from training to inference) 
            to avoid shape mismatches between previous actions and the current action. """
        self.action = None
            
    def get_smoothed_action(self, action):
        """ Returns the EMA action and updates the moving average. 
            Automatically resets the EMA when shape mismatches occur to avoid throwing an error. """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        action = action.to(self.device)

        if action is None:
            raise TypeError(f"Action should be torch.tensor or array-like, but got {action}")
        
        # reset the EMA automatically when the batch sizes changes
        if self.action is not None and self.action.shape != action.shape:
            self.reset()

        # update the EMA
        if self.action is None:
            self.action = action
        else:
            self.action = self.decay * self.action + (1-self.decay) * action
        return self.action


class PIDController():
    
    def __init__(self, n_actions=None, kp=0.7, ki=0.0, kd=0.0):
        config = load_config()
        if n_actions is None:
            n_actions = config["A"]
        self.action = torch.zeros(n_actions).to(config["device"])
        self.target_action = torch.zeros(n_actions).to(config["device"])
        self.prev_error = torch.zeros(n_actions).to(config["device"])
        self.integral = torch.zeros(n_actions).to(config["device"])
        self.kp = kp # proportional gain
        self.ki = ki # integral gain
        self.kd = kd # derivative gain
        self.device = config["device"]
            
    def get_smoothed_action(self, target_action):
        """ Returns the smoothed action using a PID controller. """
        if not isinstance(target_action, torch.Tensor):
            target_action = torch.tensor(target_action)
        target_action.to(self.device)
        
        error = target_action - self.action
        self.integral += error
        derivative = error - self.prev_error
        
        self.action = self.action + self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        
        return self.action


class MetricsTracker():
    
    def __init__(self, training_metrics: List[str] = None, episode_metrics: List[str] = None):
        config = load_config()
        self.log_interval = config["log_interval"]
        self.log_dir = config["log_dir"]
        self.device = config["device"]
        
        self.training_metrics = dict()
        self.episode_metrics = dict()
        
        # init training metrics dict
        if training_metrics:
            for name in training_metrics:
                self.training_metrics[name] = list()
        
        # init episode metrics dict
        if episode_metrics:
            for name in episode_metrics:
                self.episode_metrics[name] = list()

        # tensorboard setup        
        self.writer = SummaryWriter(self.log_dir)        
        
    def add(self, 
            training_metrics: Dict[str, torch.Tensor] = None, 
            episode_metrics: Dict[str, torch.Tensor] = None
           ):
        """ This method is called every step. Adds metrics to the tracker. 
            Note: All dtypes are torch.tensor """
        
        if training_metrics:
            for name, value in training_metrics.items():
                # convert to tensor
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value).to(self.device)
                
                # add to list
                self.training_metrics[name].append(value)

        if episode_metrics:
            for name, value in episode_metrics.items():
                # convert to tensor
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value).to(self.device)
                
                # add to list
                self.episode_metrics[name].append(value)
            
    def get_episode_batches(self, reduction=None) -> Dict[str, torch.Tensor]:
        """ Returns a dictionary of all tracked episode metrics as a stacked tensor and then resets the batches."""
        episode_metrics = dict()
        
        for name, batch in self.episode_metrics.items():
            # stack the batch as a tensor
            episode_metrics[name] = torch.stack(batch)

            if reduction == "mean":
                episode_metrics[name] = torch.mean(episode_metrics[name])
            if reduction == "sum":
                episode_metrics[name] = torch.sum(episode_metrics[name])
            
            # reset the batch
            self.episode_metrics[name] = []

        return episode_metrics
    
    def log_to_tensorboard(self, step):
        """ This method is called every log_interval steps. Logs the metrics to tensorboard. """
        for name, batch in self.training_metrics.items():
            self.writer.add_scalar(name, batch[-1].item(), global_step=step)


def save_image_and_reconstruction(x, x_pred, episode):

    config = load_config()

    # take the first image if the input is a batch
    if len(x.shape) == 4:
        x = x[0]
    if len(x_pred.shape) == 4:
        x_pred = x_pred[0]

    # imshow expects channels at the last dim
    original_image = to_np(x.permute(1,2,0))
    reconstructed_image = to_np(x_pred.permute(1,2,0))

    # Create a figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image on the left side
    cmap = "gray" if config["grayscale"] else None
    axes[0].imshow(original_image, cmap=cmap)
    axes[0].set_title("Original Image")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot reconstructed image on the right side
    axes[1].imshow(reconstructed_image, cmap=cmap)
    axes[1].set_title("Reconstructed Image")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout(pad=2.0)
    plt.savefig(f"reconstructions/episode_{episode}_comparison.png")
    plt.close(fig)


def make_env():
    """
    Creates environments with observation shapes:
        Shape in vectorized=False mode: (*OBS_SHAPE)
        Shape in vectorized=True mode: (NUM_ENVS,*OBS_SHAPE)
    """

    config = load_config()
    
    print("Creating environment with properties:")

    if config["toy_env"]:
        print("- type: toy env")
        make_one_env = lambda: gym.make(
            "CarRacing-v2", 
            render_mode="rgb_array",
            continuous=True,
        )

    else:
        print("- type: donkey sim env")

        sim_config = {
            # sim: fixed settings
            "exe_path": config["exe_path"],
            "port": config["port"],
            "host": "localhost",
            "log_level": logging.INFO,
            "start_delay": 5.0,
            "cam_resolution": (120, 160, 3),

            # sim: hyperparameters
            "max_cte": config["max_cte"],
            "frame_skip": config["frame_skip"],
            "steer_limit": config["steer_limit"],
            "throttle_min": config["throttle_min"],
            "throttle_max": config["throttle_max"],

            # sim: cosmetics
            "body_style": config["body_style"],
            "body_rgb": config["body_rgb"],
            "car_name": config["car_name"],
            "font_size": config["font_size"],
        }

        make_one_env = lambda: gym.make(
            "GymV21Environment-v0", 
            env_id=config["env_id"],
            make_kwargs={
                "conf": sim_config
            })
    
    # helper functions to automatically reset individual envs after max_episode_steps
    print("- adding a TimeLimit wrapper with %d max episode steps" % config["max_episode_steps"])
    print("- adding an AutoReset wrapper")
    make_time_limit_env = lambda: gym.wrappers.TimeLimit(make_one_env(), max_episode_steps=config["max_episode_steps"])
    make_auto_reset_env = lambda: gym.wrappers.AutoResetWrapper(make_time_limit_env())

    if config["vectorized"]:
        n_envs = config["n_envs"]

        if config["sb3_monitor"]:
            print("- wrapping the env in a SB3 Monitor wrapper to record episode statistics")
            env = DummyVecEnv([lambda: Monitor(make_auto_reset_env())] * n_envs)
        else:
            print(f"- making a AsyncVectorEnv with {n_envs} envs")
            env = gym.vector.AsyncVectorEnv([lambda: make_auto_reset_env() for i in range(n_envs)])
    else:
        raise NotImplementedError("This implementation only supports vectorized environments. Use n_envs=1 for a single environment.")
        print("- making a non-vectorized env")
        env = make_auto_reset_env()

    print("- adding a RescaleActionV0 wrapper", end=" ")
    env = RescaleActionV0(env, min_action=config["action_space_low"], max_action=config["action_space_high"])
    print("Low:", config["action_space_low"], end=", ")
    print("High:", config["action_space_high"])

    print("- adding a Gymnasium RecordEpisodeStatistics wrapper")
    env = gym.wrappers.RecordEpisodeStatistics(
        env, 
        deque_size=config["n_envs"] * config["n_updates"] if config["vectorized"] else config["n_updates"],
    )

    return env