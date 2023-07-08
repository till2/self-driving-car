import torch
import torchvision
from torchvision import transforms

from .utils import load_config

config = load_config()

device = config["device"]
grayscale = config["grayscale"]
height, width = config["size"]

""" 
Scales to [0,1], resizes to [H_new,W_new] and optionally grayscales a batch of images

Args:
    x (torch.Tensor): input tensor => use torch.tensor(obs) as the input
        Shape: (B,H,W,C)

Returns:
    x (torch.Tensor): transformed batch of images
        Shape: (B,H,W,C)

Notes:
    - only works on tensors as input, otherwise wouldn't work with batches
"""
transform = transforms.Compose([
    #transforms.ToTensor(), # devides by 255 to scale the input to the range [0,1]
    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)), # => (B,C,H,W)
    transforms.Lambda(lambda x: x / 255),
    transforms.Lambda(lambda x: x.to(device)),
    transforms.Resize((height, width)),
    transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),
    transforms.Lambda(lambda x: x.permute(0, 2, 3, 1)),
])

# result = preprocess(obs)
# print("min:", result.min())
# print("max:", result.max())
# print("mean:", result.mean())
# print("std:", result.std())

# plt.hist(torch.ravel(result).numpy(), bins=50, density=True);