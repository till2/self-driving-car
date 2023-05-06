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


class ConvBlock(nn.Module):
    """ Use this block to change the number of channels. """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

    
class TransposeConvBlock(nn.Module):
    """ Use this block to change the number of channels and perform a deconvolution
        followed by batchnorm and a relu activation. """
    def __init__(self, in_channels, out_channels):
        super(TransposeConvBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

    
class ResConvBlock(nn.Module):
    """ This block needs the same number input and output channels.
        It performs three convolutions with batchnorm, relu 
        and then adds a skip connection. """
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # uncomment later. commenting stuff out to save parameters for now.
        # residual = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        # x = self.conv2(x)        
        # x += residual
        # x = self.relu(x)
        # x = self.bn2(x)
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

    def forward(self, logits):
        
        # Compute the softmax probabilities
        probs = F.softmax(logits.view(-1, self.num_classes, self.num_classes), -1)

        # from the DreamerV3 paper: parameterize as 1% uniform and 99% logits to avoid near-deterministic distributions
        probs = 0.01 * torch.ones_like(probs) / probs.shape[-1] + 0.99 * probs

        # Sample from the categorical distribution
        m = dist.OneHotCategorical(probs=probs)
        sample = m.sample()

        # Compute the straight-through gradient estimator
        grad = m.probs - m.probs.detach()
        sample = sample + grad # has the gradient of probs
        
        self.entropy = m.entropy()
        return sample