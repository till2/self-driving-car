import os
from operator import itemgetter

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from .categorical_vae import CategoricalVAE
from .mlp import MLP
from .utils import load_config, to_np


class RSSM(nn.Module):
    def __init__(self):
        super(RSSM, self).__init__()

        config = load_config()
        self.A, self.H, self.Z = itemgetter("A", "H", "Z")(config)
        self.num_categoricals, self.num_classes = itemgetter("num_categoricals", "num_classes")(config)
        A, H, Z = self.A, self.H, self.Z

        # init the VAE
        self.vae = CategoricalVAE(features=H+Z)
        
        # init the RNN
        self.num_rnn_layers = 1
        self.rnn = nn.GRU(input_size=A+H+Z, hidden_size=H, num_layers=self.num_rnn_layers)
        
        # init MLPs
        self.dynamics_mlp = MLP(input_dims=H, output_dims=Z) # H -> Z
        self.reward_mlp = MLP(input_dims=H+Z, output_dims=1) # state (H+Z) -> 1
        self.continue_mlp = MLP(input_dims=H+Z, output_dims=1, out_type="sigmoid") # state (H+Z)->1 # add sigmoid and BinaryCE  
    
    def step(self, action, h, z):
        A, H, Z = self.A, self.H, self.Z
        h = h.view(-1, H)

        # convert the action to a tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=h.device).view(1,-1) # (1,A)

        # reconstruct the image
        x_reconstruction = self.vae.decode(h, z)

        state = torch.cat((h, z), dim=1)
        
        # predict the reward and continue flag
        reward_pred = self.reward_mlp(state)
        continue_prob = self.continue_mlp(state)
        continue_pred = torch.bernoulli(continue_prob)

        # rssm step:
        # concatenate the rnn_input and apply RNN to obtain the next hidden state
        rnn_input = torch.cat((action, h, z), 1)
        # linear in
        _, h = self.rnn(rnn_input, h)
        # linear out  
        
        return h, reward_pred, continue_prob, continue_pred, x_reconstruction
    
    def get_losses(self,
                   x_target, x_pred,
                   reward_target, reward_pred,
                   continue_target, continue_prob,
                   z_pred, z):
        
        image_loss = F.mse_loss(x_target, x_pred, reduction="mean")
        reward_loss = F.mse_loss(reward_target.squeeze(), reward_pred.squeeze(), reduction="mean")
        continue_loss = F.binary_cross_entropy(continue_prob.squeeze(), continue_target.squeeze())
        
        # DreamerV3 KL losses: regularize the posterior (z) towards the prior (z_pred)
        kld = dist.kl.kl_divergence
        
        # define the distributions with grad
        dist_z = dist.OneHotCategorical(probs=z.view(-1, self.num_categoricals, self.num_classes))
        dist_z_pred = dist.OneHotCategorical(probs=z_pred.view(-1, self.num_categoricals, self.num_classes))
        
        # define the distributions without grad
        dist_z_sg = dist.OneHotCategorical(probs=z.detach().view(-1, self.num_categoricals, self.num_classes))
        dist_z_pred_sg = dist.OneHotCategorical(probs=z_pred.detach().view(-1, self.num_categoricals, self.num_classes))

        # calculate the mean KL-divergence across the categoricals
        
        dyn_loss = torch.max(torch.tensor(1), torch.mean(kld(dist_z_sg, dist_z_pred)))
        rep_loss = torch.max(torch.tensor(1), torch.mean(kld(dist_z, dist_z_pred_sg)))
     
        # calculate the combined loss
        loss = 1.0 * (image_loss + reward_loss + continue_loss) + 0.5 * dyn_loss + 0.1 * rep_loss
        
        return {"loss": loss, "image_loss": image_loss, "reward_loss": reward_loss, 
                "continue_loss": continue_loss, "dyn_loss": dyn_loss, "rep_loss": rep_loss}
    
    def save_weights(self):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        base_path = "weights/RSSM"
        index = 0
        while os.path.exists(f"{base_path}_{index}"):
            index += 1
        torch.save(self.state_dict(), f"{base_path}_{index}")

    def load_weights(self, path="weights/RSSM", eval_mode=True):
        self.load_state_dict(torch.load(path))
        if eval_mode:
            print("Set RSSM to evaluation mode.")
            self.eval()