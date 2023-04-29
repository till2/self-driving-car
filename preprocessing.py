import torch
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" normalize to ImageNet mean and std, resize to 128x128 and grayscale (-> output dims: 1 x 128 x 128) """
grayscale_transform = transforms.Compose([
    transforms.ToTensor(), # -> scaled to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
    transforms.Lambda(lambda x: x.to(device)), # Move to device
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
])