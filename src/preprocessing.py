import torch
import torchvision
from torchvision import transforms

from .utils import load_config

config = load_config()

device = config["device"]
grayscale = config["grayscale"]
height, width = config["size"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(device)),
    transforms.Resize((height, width)),
    transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),
])

# result = preprocess(obs)
# print("min:", result.min())
# print("max:", result.max())
# print("mean:", result.mean())
# print("std:", result.std())

# plt.hist(torch.ravel(result).numpy(), bins=50, density=True);