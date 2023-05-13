import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions as dist

class ConvBlock(nn.Module):
    """ Use this block to perform a convolution and change the number of channels. """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, 
                    height=None, width=None, bias=False, transpose_conv=False):
        super(ConvBlock, self).__init__()
        
        if transpose_conv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, output_padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if height and width:
            self.norm = nn.LayerNorm([out_channels, height, width])
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        self.activation = nn.SiLU(inplace=True) # DreamerV2: ELU, DreamerV3: SiLU

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

  
class CategoricalStraightThrough(nn.Module):
    """
    Given a tensor of logits, this module samples from a categorical distribution,
    and computes the straight-through gradient estimator of the sampling operation.
    The entropy of the latest forward pass distribution gets saved as an attribute.
    """
    def __init__(self, num_classes):
        super(CategoricalStraightThrough, self).__init__()
        self.num_classes = num_classes
        self.entropy = None

    def forward(self, logits, uniform_ratio=0.01):
        
        # Compute the softmax probabilities
        probs = F.softmax(logits.view(-1, self.num_classes, self.num_classes), -1)

        # from the DreamerV3 paper: parameterize as 1% uniform and 99% logits to avoid near-deterministic distributions
        probs = uniform_ratio * torch.ones_like(probs) / probs.shape[-1] + (1-uniform_ratio) * probs

        # Sample from the categorical distribution
        m = dist.OneHotCategorical(probs=probs)
        sample = m.sample()

        # Compute the straight-through gradient estimator
        grad = m.probs - m.probs.detach()
        sample = sample + grad # has the gradient of probs
        
        self.entropy = m.entropy()
        return sample