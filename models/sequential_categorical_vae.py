import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import distributions as dist
from torch.distributions import Normal, Categorical

import torchvision
from torchvision import transforms

from .blocks import ConvBlock, TransposeConvBlock, ResConvBlock, CategoricalStraightThrough
from .mlp import MLP


class SeqCatVAE(nn.Module):
    """
    The sequential categorical VAE first feeds x through the CNN encoder, 
    then concatenates the result with h and feeds concat(h,enc(x))
    through a MLP to get the logits for the categorical distribution. 
    Finally samples z from the categorical with straight-through gradients.  

    Input: h (batch of determininistic hidden states) and x (batch of transformed input observations)
    Output: z (sampled stochastic hidden state)
    """
    def __init__(self, H=512, Z=32*32, grayscale=True, vae_ent_coeff=0.0):
        super(SeqCatVAE, self).__init__()

        if grayscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
        
        self.H = H
        self.Z = Z
        self.vae_ent_coeff = vae_ent_coeff
        
        self.encoder = nn.Sequential(
            ConvBlock(self.input_channels, 32),
            ResConvBlock(32, 32),
            ConvBlock(32, 32, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(32, 32),
            ResConvBlock(32, 32),
            ConvBlock(32, 32, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(32, 64),
            ResConvBlock(64, 64),
            ConvBlock(64, 64, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(64, 64),
            ResConvBlock(64, 64),
            ConvBlock(64, 64, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(64, 32),
            ResConvBlock(32, 32),
            
            ConvBlock(32, 16),
            ResConvBlock(16, 16),
        )
        
        self.encoder_mlp = MLP(input_dims=self.H + 16*8*8, output_dims=self.Z) # H+16*8*8 -> Z
        self.categorical = CategoricalStraightThrough(num_classes=32)
        self.decoder_mlp = MLP(input_dims=self.H+self.Z, output_dims=self.Z)
        
        self.decoder = nn.Sequential(
            TransposeConvBlock(1, 128),
            ResConvBlock(128, 128),
            
            TransposeConvBlock(128, 64),
            ResConvBlock(64, 64),
            
            ConvBlock(64, 64),
            ResConvBlock(64, 64),
            
            ConvBlock(64, 32),
            ResConvBlock(32, 32),
            
            ConvBlock(32, 32),
            ResConvBlock(32, 32),
            
            ConvBlock(32, 16),
            nn.Conv2d(16, self.input_channels, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

    def encode(self, h, x):
        h_and_encoding = torch.cat((h.view(-1, self.H), self.encoder(x).view(-1, 16*8*8)), dim=1)
        logits = self.encoder_mlp(h_and_encoding).view(-1, 32, 32)
        z = self.categorical(logits)
        return z
    
    def decode(self, h, z):
        h_and_z = torch.cat((h.view(-1, self.H), z.view(-1, self.Z)), dim=1)
        decoder_cnn_input = self.decoder_mlp(h_and_z).view(-1, 1, 32, 32)
        x = self.decoder(decoder_cnn_input)
        return x

    def forward(self, h, x):
        # encode
        z = self.encode(h, x).view(-1, 32, 32)
        # decode
        x_reconstruction = self.decode(h, z)
        return x_reconstruction

    def get_loss(self, x, x_reconstruction):
        
        # image reconstruction loss
        reconstruction_loss = F.mse_loss(x, x_reconstruction, reduction="mean")
        
        # total loss
        entropy_loss = - self.vae_ent_coeff * self.categorical.entropy.mean()
        loss = reconstruction_loss + entropy_loss
        return loss, reconstruction_loss, entropy_loss

    def save_weights(self):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        torch.save(self.state_dict(), "weights/SeqCatVAE")
    
    def load_weights(self, path="weights/SeqCatVAE", eval_mode=True):
        self.load_state_dict(torch.load(path)) 
        if eval_mode:
            print("Set SequentialCategoricalVAE to evaluation mode.")
            self.eval()
        
# batch_size = 8
# vae = SeqCatVAE(
#     H=512,
#     Z=32*32,
#     grayscale=True,
#     vae_ent_coeff=0.000001
# ).to(device)