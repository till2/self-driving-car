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


to_np = lambda x: x.detach().cpu().numpy()

symlog = lambda r: np.sign(r) * np.log(abs(r) + 1)
symexp = lambda r: np.sign(r) * (np.exp(abs(r)) - 1)


def load_config():
    yaml = YAML(typ='safe')
    file_path = os.path.expanduser('~/Desktop/self-driving-car/src/config.yaml')
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

    config = load_config()
    
    if config["toy_env"]:
        print("Making a toy env.")
        make_one_env = lambda: gym.make(
            "CarRacing-v2", 
            render_mode="rgb_array",
            continuous=True,
        )

    else:
        print("Making a donkey sim env.")

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
    
    if config["vectorized"]:
        n_envs = config["n_envs"]

        print(f"Making {n_envs} vectorized envs.")
        if config["sb3_monitor"]:
            print("Wrapping the env in a SB3 Monitor wrapper to record episode statistics.")
            env = DummyVecEnv([lambda: Monitor(make_one_env())] * n_envs)
        else:
            print("Adding a Gymnasium RecordEpisodeStatistics wrapper.")
            env = gym.wrappers.RecordEpisodeStatistics(
                make_one_env(), 
                deque_size=config["n_envs"] * config["n_updates"],
            )
        
    else:
        print("Making a non-vectorized env.")
        print("Adding a Gymnasium RecordEpisodeStatistics wrapper.")
        env = gym.wrappers.RecordEpisodeStatistics(
            make_one_env(), 
            deque_size=config["n_updates"],
        )
    
    print("Adding a TimeLimit wrapper with %d max episode steps." % config["max_episode_steps"])
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config["max_episode_steps"])

    print("Adding an AutoReset wrapper.")
    env = gym.wrappers.AutoResetWrapper(env)

    print("Adding a RescaleActionV0 wrapper.", end=" ")
    env = RescaleActionV0(env, min_action=config["action_space_low"], max_action=config["action_space_high"])
    print("Low:", env.action_space.low, end=", ")
    print("High:", env.action_space.high)

    # Maybe do the symlog reward scaling after env.step() to log the real reward 
    # if config["symlog_rewards"]:
    #     print("Adding symlog reward scaling.")
    #     env = gym.wrappers.TransformReward(env, symlog)

    return env