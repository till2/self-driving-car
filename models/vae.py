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

from .blocks import ConvBlock, TransposeConvBlock, ResConvBlock

    
class VAE(nn.Module):
    def __init__(self, greyscale=True, beta=4):
        super(VAE, self).__init__()

        if greyscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
            
        self.beta = 50 # kl-multiplier (from beta-VAE paper)
        
        self.encoder = nn.Sequential(
            ConvBlock(self.input_channels, 16),
            ResConvBlock(16, 16),
            ConvBlock(16, 16, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(16, 32),
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
            ConvBlock(256, 256, kernel_size=4, padding=2, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ResConvBlock(256, 256),
            # ConvBlock(256, 4, kernel_size=4, padding=1, stride=1),
            
            ConvBlock(256, 8, kernel_size=4, padding=1, stride=1),
            nn.Flatten(),
            
        )
        
        self.mu = nn.Linear(8*4*4, 32)
        self.logvar = nn.Linear(8*4*4, 32)

        self.decoder = nn.Sequential(
            TransposeConvBlock(2, 32),
            ResConvBlock(32, 32),
            
            TransposeConvBlock(32, 64),
            ResConvBlock(64, 64),
            
            TransposeConvBlock(64, 128),
            ResConvBlock(128, 128),
            
            TransposeConvBlock(128, 256),
            ResConvBlock(256, 256),
            
            TransposeConvBlock(256, 512),
            ResConvBlock(512, 512),
            
            ResConvBlock(512, 512),
            ConvBlock(512, 32),
            nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

        
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        return mu, logvar
    
    def decode(self, z):
        x = self.decoder(z)
        return x
        
    def forward(self, x):
        # get the distribution of z
        mu, logvar = self.encode(x)
        
        # reparameterization
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(logvar)
        
        # sample z
        z = mu + std * epsilon
        
        # reconstruct x
        xhat = self.decode(z.view(-1, 2, 4, 4))
        return xhat, mu, logvar

    def get_loss(self, x, xhat, mu, logvar):
        
        # image reconstruction loss
        reconstruction_loss = F.mse_loss(x, xhat, reduction='sum')
        
        # KL divergence between the latent distribution and the standard normal distribution
        var = torch.exp(logvar)
        kl_divergence = 0.5 * torch.sum(var -logvar -1 +mu.pow(2))
        
        # total loss
        loss = reconstruction_loss + self.beta * kl_divergence
        return loss, reconstruction_loss, kl_divergence

    def save_weights(self):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        torch.save(self.state_dict(), "weights/VAE")
    
    def load_weights(self, path="weights/VAE", eval_mode=True):
        self.load_state_dict(torch.load(path))
        if eval_mode:
            print("Set VAE to evaluation mode.")
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
        print("mu shape:", list(self.encode(batch_tensor_dummy)[0].shape))
        print("logvar shape:", list(self.encode(batch_tensor_dummy)[1].shape))
        print("output shape:", list(self(batch_tensor_dummy)[0].shape))

# vae = VAE(greyscale=True).to(device)

# vae_optim = optim.Adam(
#     vae.parameters(), 
#     lr=3e-4, 
#     weight_decay=1e-5 # l2 regularization
# )

# vae_scheduler = ReduceLROnPlateau(vae_optim, 'min')


# print(vae.get_num_params())