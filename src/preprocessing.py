import torch
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

grayscale_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(device)),
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
])

# result = preprocess(obs)
# print("min:", result.min())
# print("max:", result.max())
# print("mean:", result.mean())
# print("std:", result.std())

# plt.hist(torch.ravel(result).numpy(), bins=50, density=True);