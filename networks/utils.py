import torch
import numpy as np
from ruamel.yaml import YAML


def to_np(x):
    return x.detach().cpu().numpy() 

def load_config():
    yaml = YAML(typ='safe')
    with open("./networks/config.yaml", "r") as file:
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
