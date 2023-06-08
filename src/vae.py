import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .blocks import CategoricalStraightThrough, ConvBlock
from .utils import load_config


class VAE(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        config = load_config()

        self.input_channels = 1 if config["grayscale"] else 3 # for the encoder
        self.decoder_start_channels = config["channels"][-1] # for the decoder

        self.hidden_dims = config["Z"]
        self.beta = 1
        
        self.mu = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.logvar = nn.Linear(self.hidden_dims, self.hidden_dims)
        
        self.entropyloss_coeff = config["entropyloss_coeff"]
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # settings
        kernel_size = config["kernel_size"]
        stride = config["stride"]
        padding = config["padding"]

        # channels
        input_channels = self.input_channels
        channels = config["channels"]

        print("Initializing encoder:")
        height, width = config["size"]
        for i, out_channels in enumerate(channels):
            
            height = (height + 2*padding - kernel_size) // stride + 1
            width = (width + 2*padding - kernel_size) // stride + 1

            print(f"- adding ConvBlock({input_channels, out_channels}) \
                  ==> output shape: ({out_channels}, {height}, {width}) ==> prod: {out_channels * height * width}")
            conv_block = ConvBlock(input_channels, out_channels, kernel_size, stride, 
                                   padding, height, width)
            self.encoder.add_module(f"conv_block_{i}", conv_block)
            input_channels = out_channels
        
        # save the shape of the encoded image
        self.decoder_start_height, self.decoder_start_width = height, width
        self.linear = nn.Linear(self.hidden_dims, self.decoder_start_channels * height * width)
        
        # Add linear layer after the encoder
        print(f"- adding Flatten()")
        self.encoder.add_module("flatten", nn.Flatten())

        # predict mu and logvar
        print(f"- adding Linear() for Mu: {self.hidden_dims} and Logvar: {self.hidden_dims}")
            
        print("\nInitializing decoder:")
        print(f"- adding Reshape: (*,{self.hidden_dims}) => (*,{self.decoder_start_channels},{height},{width})") # reshape in self.decode function
        padding = config["padding"]
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

        if config["decoder_final_activation"].lower() == "sigmoid":
            self.decoder.add_module("output_activation", nn.Sigmoid())
        
        self.to(config["device"])

            
    def encode(self, x):
        logits  = self.encoder(x).flatten()
        mu, logvar = self.mu(logits), self.logvar(logits)
        
        # reparameterization
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(logvar)
        z = mu + std * epsilon
        return z, mu, logvar
    
    
    def decode(self, z):
        dec_inp = self.linear(z).view(self.decoder_start_channels, self.decoder_start_height, self.decoder_start_width)
        x_hat = self.decoder(dec_inp)
        return x_hat
    
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat, mu, logvar
        

    def get_loss(self, x, x_hat, mu, logvar):
        
        # image reconstruction loss
        reconstruction_loss = F.mse_loss(x, x_hat, reduction="mean")
        
        # KLD between latent distribution and standard normal distribution
        var = torch.exp(logvar)
        KLD = 0.5 * torch.sum(var -logvar -1 + mu.pow(2))
        
        # total loss
        kld_loss = self.beta * KLD
        loss = reconstruction_loss + kld_loss
        return loss, reconstruction_loss, kld_loss

    
    def save_weights(self):
        os.makedirs("weights", exist_ok=True)
        base_path = "weights/VAE"
        index = 0
        while os.path.exists(f"{base_path}_{index}"):
            index += 1
        torch.save(self.state_dict(), f"{base_path}_{index}")

        
    def load_weights(self, path="weights/VAE", eval_mode=True):
        self.load_state_dict(torch.load(path))
        if eval_mode:
            print("Set VAE to evaluation mode.")
            self.eval()
    
    
    @classmethod
    def get_num_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)