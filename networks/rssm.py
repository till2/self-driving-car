import os
from operator import itemgetter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from .categorical_vae import CategoricalVAE
from .mlp import MLP
from .utils import to_np, load_config



class RSSM(nn.Module):
    def __init__(self):
        super(RSSM, self).__init__()

        config = load_config()
        A, H, Z = itemgetter("A", "H", "Z")(config)
        self.num_categoricals, self.num_classes = itemgetter("num_categoricals", "num_classes")(config)

        # init the VAE
        self.vae = CategoricalVAE(features=H+Z)
        
        # init the RNN
        self.num_rnn_layers = 1
        self.rnn = nn.GRU(input_size=A+H+Z, hidden_size=H, num_layers=self.num_rnn_layers)
        
        # init MLPs
        self.dynamics_mlp = MLP(input_dims=H, output_dims=Z) # H -> Z
        self.reward_mlp = MLP(input_dims=H+Z, output_dims=1) # state (H+Z) -> 1
        self.continue_mlp = MLP(input_dims=H+Z, output_dims=1) # state (H+Z)->1 # add sigmoid and BinaryCE  
    
    def step(self, action, h, z):

        # concatenate the rnn_input and apply RNN to obtain the next hidden state
        rnn_input = torch.cat((action, h.view(-1, H), z), 1)
        _, h = self.rnn(rnn_input, h.view(-1, H))
        
        state = torch.cat((h.view(-1, H), z), 1)
        
        # predict the reward and continue flag
        reward_pred = self.reward_mlp(state)
        continue_prob = torch.sigmoid(self.continue_mlp(state)) # binary classification
        continue_pred = bool(continue_prob > 0.5)
        
        x_reconstruction = self.vae.decode(h, z)
        
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