import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions as dist

from .utils import load_config


class ConvBlock(nn.Module):
    """ Use this block to perform a convolution and change the number of channels. """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, 
                    height=None, width=None, transpose_conv=False):
        super().__init__()
        config = load_config()

        conv_bias = config["conv_bias"]

        # conv
        if transpose_conv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=conv_bias, output_padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=conv_bias)

        # norm
        if height and width:
            self.norm = nn.LayerNorm([out_channels, height, width])
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        # activation (DreamerV2: ELU, DreamerV3: SiLU)
        if config["activation"].lower() == "silu":
            self.activation = nn.SiLU(inplace=True)
        elif config["activation"].lower() == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

  
class CategoricalStraightThrough(nn.Module):
    """
    A Categorical distribution that returns one-hot categorical samples, with a
    stright-through gradient estimator to get gradients through the discrete sampling operation.  

    Note:
        The entropy of the latest forward pass distribution gets saved as an attribute.
    """
    def __init__(self):
        super().__init__()
        config = load_config()
        
        self.num_categoricals = config["num_categoricals"]
        self.num_classes = config["num_classes"]
        self.uniform_ratio = config["uniform_ratio"]
        self.entropy = None
        self.device = config["device"]

    def forward(self, logits):
        """
        Calculates the categorical input probabilities as a softmax over the classes

        Args:
            logits (torch.Tensor):
                Shape:
                    (Z,) or
                    (B, Z) or
                    (NUM_CATEGORICALS, NUM_CLASSES) or
                    (B, NUM_CATEGORICALS, NUM_CLASSES)
        
        Returns:
            sample (torch.Tensor): One-hot encoded samples from NUM_CATEGORICALS distributions over NUM_CLASSES discrete classes.
                Shape: (B, NUM_CATEGORICALS, NUM_CLASSES)
            probs (torch.Tensor): The input probabilities for the one-hot categorical distribution. (these are required for the KL loss)
                Shape: (B, NUM_CATEGORICALS, NUM_CLASSES)

        Note:
            Reshapes any valid input to the correct format to always return the same output shape.
        """
        # move to device
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        logits = logits.to(self.device)

        # prepare shape and potential error message (for wrong input shapes)
        input_shape = logits.shape
        error_message = f"Invalid input shape {list(input_shape)} to Categorical"

        # Convert: (Z,) => (B, NUM_CATEGORICALS, NUM_CLASSES)
        if len(logits.shape) == 1:  
            batch_shape = 1
            assert logits.shape[0] == self.num_categoricals * self.num_classes, error_message
            logits = logits.view(batch_shape, self.num_categoricals, self.num_classes)

        # (B, Z) or (NUM_CATEGORICALS, NUM_CLASSES)
        elif len(logits.shape) == 2:

            # Convert: (NUM_CATEGORICALS, NUM_CLASSES) => (B, NUM_CATEGORICALS, NUM_CLASSES)
            if logits.shape[0] == self.num_categoricals and logits.shape[1] == self.num_classes:
                batch_shape = 1
                logits = logits.unsqueeze(0)
            
            # Convert: (B, Z) => (B, NUM_CATEGORICALS, NUM_CLASSES)
            else:
                batch_shape = logits.shape[0]
                assert logits.shape[1] == self.num_categoricals * self.num_classes, error_message
                logits = logits.view(batch_shape, self.num_categoricals, self.num_classes)

        # (B, NUM_CATEGORICALS, NUM_CLASSES)
        elif len(logits.shape) == 3:
            batch_shape = logits.shape[0]
            assert logits.shape[1] == self.num_categoricals and logits.shape[2] == self.num_classes, error_message

        else:
            raise AssertionError(error_message)
        
        # shape should be (B, NUM_CATEGORICALS, NUM_CLASSES)
        assert len(logits.shape) == 3
        assert logits.shape[0] >= 1
        assert logits.shape[1] == self.num_categoricals
        assert logits.shape[2] == self.num_classes
        
        # Compute the softmax probabilities (softmax over classes)
        probs = F.softmax(logits, dim=-1)

        # from the DreamerV3 paper: parameterize as 1% uniform and 99% logits to avoid near-deterministic distributions
        probs = self.uniform_ratio * (torch.ones_like(probs) / self.num_classes) + (1-self.uniform_ratio) * probs

        # Sample from the categorical distribution
        m = dist.OneHotCategorical(probs=probs)
        sample = m.sample()

        # Compute the straight-through gradient estimator
        grad = m.probs - m.probs.detach()
        sample = sample + grad # has the gradient of probs
        
        self.entropy = m.entropy()
        return sample, probs