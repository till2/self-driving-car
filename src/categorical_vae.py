import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .blocks import CategoricalStraightThrough, ConvBlock
from .utils import load_config

class CategoricalVAE(nn.Module):
    def __init__(self):
        super(CategoricalVAE, self).__init__()
        config = load_config()

        self.input_channels = 1 if config["grayscale"] else 3 # for the encoder
        self.decoder_start_channels = config["channels"][-1] # for the decoder
        self.num_categoricals = config["num_categoricals"]
        self.num_classes = config["num_classes"]

        self.n_features = config["H"] + config["Z"]
        self.entropyloss_coeff = config["entropyloss_coeff"]
        
        self.linear = nn.Linear(self.n_features, self.decoder_start_channels*4*4)
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.categorical = CategoricalStraightThrough()

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
        
        # Add linear layer after the encoder
        print(f"- adding Flatten()")
        self.encoder.add_module("flatten", nn.Flatten())
        print(f"- adding Reshape: (*,{self.num_categoricals * self.num_classes}) => (*,{self.num_categoricals},{self.num_classes})") # reshape in self.encode function
            
        print("\nInitializing decoder:")
        print(f"- adding Reshape: (*,{self.num_categoricals * self.num_classes}) => (*,{self.decoder_start_channels},{height},{width})") # reshape in self.decode function
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

    def encode(self, x):
        logits = self.encoder(x).view(-1,self.num_categoricals,self.num_classes)
        z = self.categorical(logits)
        return z
    
    def decode(self, h, z_flat):
        zh = torch.cat((h, z_flat), dim=-1)
        dec_inp = self.linear(zh).view(-1, self.decoder_start_channels, self.decoder_start_height, self.decoder_start_width)
        x_hat = self.decoder(dec_inp.view(-1, self.decoder_start_channels, self.decoder_start_height, self.decoder_start_width))
        return x_hat

    def get_loss(self, x, xhat):
        
        # image reconstruction loss
        reconstruction_loss = F.mse_loss(x, xhat, reduction="mean")
        
        # total loss
        entropy_loss = - self.entropyloss_coeff * self.categorical.entropy.mean()
        loss = reconstruction_loss + entropy_loss
        return loss, reconstruction_loss, entropy_loss

    def save_weights(self):
        os.makedirs("weights", exist_ok=True)
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