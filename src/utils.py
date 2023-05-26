import numpy as np
import torch
from matplotlib import pyplot as plt
from ruamel.yaml import YAML


def to_np(x):
    return x.detach().cpu().numpy() 


def load_config():
    yaml = YAML(typ='safe')
    with open("./src/config.yaml", "r") as file:
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