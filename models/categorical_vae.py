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
    def __init__(self, grayscale=True, vae_ent_coeff=0.0):
        super(CategoricalVAE, self).__init__()

        if grayscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
        self.vae_ent_coeff = vae_ent_coeff
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.categorical = CategoricalStraightThrough(num_classes=32)

        # settings
        kernel_size = 3
        stride = 2
        padding = 1

        # channels
        input_channels = self.input_channels
        channels = [16, 32, 64, 128, 256, 512, 1024]

        print("Initializing encoder:")
        height, width = 128, 128
        for i, out_channels in enumerate(channels):
            
            height = (height + 2*padding - kernel_size) // stride + 1
            width = (width + 2*padding - kernel_size) // stride + 1

            print(f"- adding ConvBlock({input_channels, out_channels}) \
                  ==> output shape: ({out_channels}, {height}, {width}) ==> prod: {out_channels * height * width}")
            conv_block = ConvBlock(input_channels, out_channels, kernel_size, stride, 
                                   padding, height, width)
            self.encoder.add_module(f"conv_block_{i}", conv_block)
            
            input_channels = out_channels
        
        print("\nInitializing decoder:")
        height, width = 1, 1
        padding=1
        for i, out_channels in enumerate(reversed(channels)):
            
            height = (height - 1)*stride - 2*padding + kernel_size + 1
            width = (width - 1)*stride - 2*padding + kernel_size + 1
            
            # last layer
            if i == len(channels)-1:
                out_channels = self.input_channels
            
            print(f"- adding transpose ConvBlock({input_channels}, {out_channels}) \
                  ==> output shape: ({out_channels}, {height}, {width}) ==> prod: {out_channels * height * width}")
            transpose_conv_block = ConvBlock(input_channels, out_channels, kernel_size, stride, 
                                             padding, height, width, transpose_conv=True)
            self.decoder.add_module(f"transpose_conv_block_{i}", transpose_conv_block)
            
            input_channels = out_channels

        self.decoder.add_module("output_activation", nn.Sigmoid())


    def encode(self, x):
        logits = self.encoder(x).view(-1, 32, 32)
        z = self.categorical(logits)
        return z
    
    def decode(self, z):
        x = self.decoder(z.view(-1, 32*32, 1, 1))
        return x

    def forward(self, x):
        z = self.encode(x).view(-1, 32, 32)
        x_hat = self.decode(z)
        return x_hat
        
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