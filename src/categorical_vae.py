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
        super().__init__()
        config = load_config()

        self.input_channels = 1 if config["grayscale"] else 3 # for the encoder
        self.decoder_start_channels = config["channels"][-1] # for the decoder
        self.num_categoricals = config["num_categoricals"]
        self.num_classes = config["num_classes"]
        self.device = config["device"]

        self.n_features = config["H"] + config["Z"]
        self.entropyloss_coeff = config["entropyloss_coeff"]
        
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
        self.linear = nn.Linear(self.n_features, self.decoder_start_channels * height * width)
        
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
        
        self.to(self.device)

    def encode(self, x):
        """
        Encodes a given preprocessed observation to the posterior stochastic state z. 
        Also returns the probs for the categorical distribution that are required for the KL loss.

        Args:
            x (torch.Tensor): Preprocessed observation (channels-first)
                Shape: (B,C,H,W)
        
        Returns:
            z (torch.Tensor):
                Shape: (B, NUM_CATEGORICALS, NUM_CLASSES)
            z_probs (torch.Tensor):
                Shape: (B, NUM_CATEGORICALS, NUM_CLASSES)
        """
        batch_size = x.shape[0]
        logits = self.encoder(x) # (B, Z) with Z = NUM_CATEGORICALS*NUM_CLASSES
        logits = logits.view(batch_size, self.num_categoricals, self.num_classes) # (B, NUM_CATEGORICALS, NUM_CLASSES)
        z, z_probs = self.categorical(logits) # => (B, NUM_CATEGORICALS, NUM_CLASSES)
        return z, z_probs
    
    def decode(self, h, z_flat):
        """
        Decodes the state, consisting of h and z, into the reconstructed image.
        The decoders final activation is a sigmoid, so it reconstructs the to [0,1] scaled pixel values.
        The decoder is currently deterministic and doesn't sample the output image.

        Args:
            h (torch.Tensor): Deterministic recurrent hidden state (B, H)
            z_flat (torch.Tensor): Flattened categorical variables (B, NUM_CATEGORICALS * NUM_CLASSES)
        
        Returns:
            x_hat (torch.Tensor): Reconstructed image (B, C, H, W)
        """
        assert len(h.shape) == 2
        batch_size = h.shape[0]

        if len(z_flat.shape) == 2:
            assert z_flat.shape[0] == batch_size
            assert z_flat.shape[1] == self.num_categoricals * self.num_classes
        elif len(z_flat.shape) == 3:
            assert z_flat.shape[0] == batch_size
            assert z_flat.shape[1] == self.num_categoricals
            assert z_flat.shape[2] == self.num_classes
        else:
            raise AssertionError(f"Decoder z input shape should be [B, NUM_CATEGORICALS * NUM_CLASSES], but got {list(z_flat.shape)}")


        zh = torch.cat((h, z_flat), dim=1) # (B, H) + (B, Z) => (B, H+Z)
        dec_inp = self.linear(zh)
        dec_inp = dec_inp.view(batch_size, self.decoder_start_channels, self.decoder_start_height, self.decoder_start_width)
        x_hat = self.decoder(dec_inp)
        return x_hat

    def get_loss(self, x, xhat):
        """
        Args:
            x (torch.Tensor): Input images (B, C, H, W)
            xhat (torch.Tensor): Reconstructed images (B, C, H, W)
        
        Returns:
            loss (torch.Tensor): Total loss
            reconstruction_loss (torch.Tensor): Image reconstruction loss
            entropy_loss (torch.Tensor): Entropy loss
        """
        
        raise NotImplementedError("This method should not be called in the current code.")

        # # image reconstruction loss
        # reconstruction_loss = F.mse_loss(x, xhat, reduction="mean")
        
        # # total loss
        # entropy_loss = - self.entropyloss_coeff * self.categorical.entropy.mean()
        # loss = reconstruction_loss + entropy_loss
        # return loss, reconstruction_loss, entropy_loss
        


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