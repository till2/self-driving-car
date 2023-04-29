import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Normal, Categorical

import torchvision
from torchvision import transforms

from .blocks import ConvBlock, TransposeConvBlock, ResConvBlock, CategoricalStraightThrough


class CategoricalVAE(nn.Module):
    def __init__(self, greyscale=True, vae_ent_coeff=0.0):
        super(CategoricalVAE, self).__init__()

        if greyscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
            
        self.vae_ent_coeff = vae_ent_coeff
        
        self.encoder = nn.Sequential(
            ConvBlock(self.input_channels, 32),
            ResConvBlock(32, 32),
            ConvBlock(32, 32, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(32, 64),
            ResConvBlock(64, 64),
            ConvBlock(64, 64, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(64, 128),
            ResConvBlock(128, 128),
            ConvBlock(128, 128, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(128, 256),
            ResConvBlock(256, 256),
            ConvBlock(256, 256, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(256, 512),
            ResConvBlock(512, 512),
            
            ConvBlock(512, 16),
            ResConvBlock(16, 16),
        )
        
        self.categorical = CategoricalStraightThrough(num_classes=32)
        
        self.decoder = nn.Sequential(
            TransposeConvBlock(1, 1024),
            ResConvBlock(1024, 1024),
            
            TransposeConvBlock(1024, 512),
            ResConvBlock(512, 512),
            
            ConvBlock(512, 256),
            ResConvBlock(256, 256),
            
            ConvBlock(256, 128),
            ResConvBlock(128, 128),
            
            ConvBlock(128, 64),
            ResConvBlock(64, 64),
            
            ConvBlock(64, 16),
            nn.Conv2d(16, self.input_channels, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        logits = self.encoder(x).view(-1, 32, 32)
        z = self.categorical(logits)
        return z
    
    def decode(self, z):
        x = self.decoder(z.view(-1, 1, 32, 32))
        return x

    def forward(self, x):
        z = self.encode(x).view(-1, 32, 32)
        
        # reconstruct x
        xhat = self.decode(z)
        return xhat

    def get_loss(self, x, xhat):
        
        # image reconstruction loss
        reconstruction_loss = F.mse_loss(x, xhat, reduction="mean")
        
        # total loss
        entropy_loss = - self.vae_ent_coeff * self.categorical.entropy.mean()
        loss = reconstruction_loss + entropy_loss
        return loss, reconstruction_loss, entropy_loss

    def save_weights(self):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        torch.save(self.state_dict(), "weights/CategoricalVAE")
    
    def load_weights(self, path="weights/CategoricalVAE", eval_mode=True):
        self.load_state_dict(torch.load(path))
        if eval_mode:
            print("Set CategoricalVAE to evaluation mode.")
            self.eval()
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def info(self):
        """
        Prints useful information, such as the device, 
        number of parameters, 
        input-, hidden- and output shapes. 
        """
        title = f"| {self.__class__.__name__} info |"
        print(title)
        print("-" * len(title))

        if next(self.parameters()).is_cuda:
            print("device: cuda")
            batch_tensor_dummy = torch.rand(8, 1, 128, 128).cuda()
        else:
            print("device: cpu")
            batch_tensor_dummy = torch.rand(8, 1, 128, 128).cpu()

        print(f"number of parameters: {self.get_num_params():_}")
        print("input shape :", list(batch_tensor_dummy.shape))
        print("hidden shape:", list(self.encode(batch_tensor_dummy).shape))
        print("output shape:", list(self(batch_tensor_dummy).shape))
        print("vae_ent_coeff:", self.vae_ent_coeff)

# batch_size = 8

# vae = CategoricalVAE(
#     greyscale=True,
#     vae_ent_coeff=0.000001
# ).to(device)

# vae_optim = optim.Adam(
#     vae.parameters(), 
#     lr=1e-2,
#     weight_decay=1e-5 # l2 regularization
# )

# vae_scheduler = ReduceLROnPlateau(vae_optim, 'min', patience=100, factor=0.5)

# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group["lr"]

# print(vae.get_num_params())