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

from .blocks import ConvBlock, CategoricalStraightThrough


class CategoricalVAE(nn.Module):
    def __init__(self, features=512+32*32, grayscale=True, entropyloss_coeff=0.0, uniform_ratio=0.01):
        super(CategoricalVAE, self).__init__()

        if grayscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
        self.entropyloss_coeff = entropyloss_coeff
        self.uniform_ratio = uniform_ratio
        
        self.linear = nn.Linear(features, 64*4*4)
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.categorical = CategoricalStraightThrough(num_classes=32)

        # settings
        kernel_size = 3
        stride = 2
        padding = 1

        # channels
        input_channels = self.input_channels
        channels = [32, 64, 128, 256, 64]

        print("Initializing encoder:")
        height, width = 128, 128
        for i, out_channels in enumerate(channels):
            
            height = (height + 2*padding - kernel_size) // stride + 1
            width = (width + 2*padding - kernel_size) // stride + 1

            print(f"- adding ConvBlock({input_channels, out_channels}) \
                  ==> output shape: ({out_channels}, {height}, {width}) ==> prod: {out_channels * height * width}")
            conv_block = ConvBlock(input_channels, out_channels, kernel_size, stride, 
                                   padding, height, width, bias=False)
            self.encoder.add_module(f"conv_block_{i}", conv_block)
            input_channels = out_channels
        
        # Add linear layer after the encoder
        print(f"- adding Flatten()")
        self.encoder.add_module("flatten", nn.Flatten())
        print(f"- adding Reshape: (*,{32 * 32}) => (*,32,32)") # reshape in self.encode function
            
        print("\nInitializing decoder:")
        print(f"- adding Reshape: (*,{32 * 32}) => (*,64,4,4)") # reshape in self.decode function
        height, width = 4, 4
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
                                             padding, height, width, bias=False, transpose_conv=True)
            self.decoder.add_module(f"transpose_conv_block_{i}", transpose_conv_block)
            
            input_channels = out_channels

        self.decoder.add_module("output_activation", nn.Sigmoid())


    def encode(self, x, uniform_ratio=0.01):
        logits = self.encoder(x).view(-1,32,32)
        z = self.categorical(logits, uniform_ratio)
        return z
    
    def decode(self, h, z_flat):
        zh = torch.cat((h, z_flat), dim=-1)
        dec_inp = self.linear(zh).view(-1,64,4,4)
        x_hat = self.decoder(dec_inp.view(-1,64,4,4))
        return x_hat

    def get_loss(self, x, xhat):
        
        # image reconstruction loss
        reconstruction_loss = F.mse_loss(x, xhat, reduction="mean")
        
        # total loss
        entropy_loss = - self.entropyloss_coeff * self.categorical.entropy.mean()
        loss = reconstruction_loss + entropy_loss
        return loss, reconstruction_loss, entropy_loss

    def save_weights(self):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        base_path = "weights/CategoricalVAE"
        index = 0
        while os.path.exists(f"{base_path}_{index}"):
            index += 1
        torch.save(self.state_dict(), f"{base_path}_{index}")

    def load_weights(self, path="weights/CategoricalVAE", eval_mode=True):
        self.load_state_dict(torch.load(path))
        if eval_mode:
            print("Set CategoricalVAE to evaluation mode.")
            self.eval()
    
    @classmethod
    def get_num_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def info(self):
        """
        Prints useful information, such as the device, 
        number of parameters, 
        input-, hidden- and output shapes. 
        """
        title = f"\n| {self.__class__.__name__} info |"
        print(title)
        print("-" * len(title))

        if next(self.parameters()).is_cuda:
            print("device: cuda")
            batch_tensor_dummy = torch.rand(8, 1, 128, 128).cuda()
            h_dummy = torch.rand(8, 512).cuda()
        else:
            print("device: cpu")
            batch_tensor_dummy = torch.rand(8, 1, 128, 128).cpu()
            h_dummy = torch.rand(8, 512).cpu()

        print(f"number of parameters: {self.get_num_params(self):,} (encoder: {self.get_num_params(self.encoder):,}, decoder: {self.get_num_params(self.decoder):,})")
        print("input shape :", list(batch_tensor_dummy.shape))
        enc_dummy = self.encode(batch_tensor_dummy).detach()
        print("hidden shape:", list(enc_dummy.shape))
        print("output shape:", list(self.decode(h_dummy, enc_dummy.flatten(start_dim=1, end_dim=2)).shape))
        print("entropyloss_coeff:", self.entropyloss_coeff)
        print("uniform_ratio:", self.uniform_ratio)