import os
from operator import itemgetter

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .blocks import CategoricalStraightThrough
from .categorical_vae import CategoricalVAE
from .mlp import MLP
from .utils import load_config, to_np, symlog, symexp, twohot_encode, ExponentialMovingAvg


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config = load_config()

        self.A, self.H, self.Z = itemgetter("A", "H", "Z")(config)
        self.num_categoricals = config["num_categoricals"]
        self.num_classes = config["num_classes"]
        self.img_size = config["size"]
        self.grayscale = config["grayscale"]
        self.device = config["device"]

        # loss hyperparameters
        self.pred_loss_coeff = config["pred_loss_coeff"]
        self.dyn_loss_coeff = config["dyn_loss_coeff"]
        self.rep_loss_coeff = config["rep_loss_coeff"]
        self.free_nats = config["free_nats"]
        self.max_grad_norm = config["max_grad_norm"]

        # init the VAE
        self.vae = CategoricalVAE()
        self.categorical = CategoricalStraightThrough()
        
        # init the GRU
        self.num_rnn_layers = config["num_rnn_layers"]
        self.rnn = nn.GRU(input_size=self.A + self.H + self.Z, hidden_size=self.H, num_layers=self.num_rnn_layers)

        # init pre- and postprocessing layers before and after the GRU
        self.pre_gru_linear = MLP(input_dims=self.A + self.H + self.Z, output_dims=self.A + self.H + self.Z, 
                                    n_layers=1, hidden_dims=512)

        # init MLPs
        print("\nInitializing dynamics_mlp.")
        self.dynamics_mlp = MLP(input_dims=self.H, hidden_dims=self.Z, output_dims=self.Z) # H -> Z
        print("\nInitializing reward_mlp.")
        self.reward_mlp = MLP(input_dims=self.H + self.Z, output_dims=1, weight_init="final_layer_zeros") # state (H+Z) -> 1
        print("\nInitializing continue_mlp.")
        self.continue_mlp = MLP(input_dims=self.H + self.Z, output_dims=1, out_type="sigmoid") # state (H+Z)->1 into bernoulli

        # init the optimizer
        self.rssm_lr = config["rssm_lr"]
        self.rssm_l2_regularization = config["rssm_l2_regularization"]
        self.optim = optim.Adam(self.parameters(), lr=self.rssm_lr, weight_decay=self.rssm_l2_regularization)

        self.to(self.device)
    
    def pre_step(self, h=None, x=None):
        """
        Performs one step with the RSSM model given an action, the previous hidden and stochastic state.

        Args:
            h (torch.Tensor): deterministic hidden state
                Shape: (B, H)
            x (torch.Tensor): preprocessed observation (required in training mode to get z)
                Shape: (B, C, H, W)
        
        Returns:
        step_dict (dict): Dictionary containing the following:
            state (torch.Tensor): combined state consisting of deterministic hidden state and stochastic state
                Shape: (B, H+Z)
            h (torch.Tensor):
                Shape: (B, H)
            z (torch.Tensor): one-hot sample from the posterior distribution (training mode) or the prior distribution (inference mode)
                Shape: (B, Z)
            z_pred (torch.Tensor): one-hot sample from the prior distribution
                Shape: (B, Z)
            z_probs (torch.Tensor):
                Shape: (B, Z)
            z_pred_probs (torch.Tensor):
                Shape: (B, Z)
            reward_pred (torch.Tensor): predicted rewards
                Shape: (B,)
            continue_prob (torch.Tensor): predicted continue probabilities
                Shape: (B,)
            continue_pred (torch.Tensor): predicted continue values
                Shape: (B,)
            x (torch.Tensor): preprocessed observation
                Shape: (B, C, H, W)
            x_reconstruction (torch.Tensor): reconstructed observation
                Shape: (B, C, H, W)
        
        Note:
            All inputs need to have a batch dimension.

            If an observation is given, use it to get the posterior stochastic state z. 
            Otherwise, use the approximation z_prior (predicted from h) in the combined state.

            Either h or x has to be given. Usage:
                Use only x for the first step in training (use x from env.reset).
                Use only x for the first step in inference (use an x from the replay buffer).
                Use only h for steps > 0 in inference (use h from rssm.step).
                Use both x and h for steps > 0 in training (use x from env.step and h from rssm.step).

            Algo - Correct order:
                1) state, z, z_prior, head_outputs = rssm.pre_step(h, x if training else None)
                2) action = agent.get_action(state)
                3) h_new = rssm.step(action, h, z)
        """
        # assert input shapes
        assert h is not None or x is not None
        if h is not None:
            assert len(h.shape) == 2
            assert h.shape[1] == self.config["H"]
        if x is not None:
            assert len(x.shape) == 4
            assert x.shape[1] == 1 if self.config["grayscale"] else 3
            assert x.shape[2] == self.config["size"][0]
            assert x.shape[3] == self.config["size"][1]

        batch_size = x.shape[0] if x is not None else h.shape[0]

        # init the deterministic hidden state to zeros if it's not given
        if h is None:
            h = torch.zeros(batch_size, self.H)
        h = h.to(self.device).view(batch_size, self.H) # (B, H)
        
        # NEW:
        # in training mode: get posterior from the current obs
        training = True if x is not None else False

        if training: # (also true for first step in inference where x is given. but this is fine.)
            z, z_probs = self.vae.encode(x) # (B, NUM_CATEGORICALS, NUM_CLASSES)
            z = z.flatten(start_dim=1, end_dim=2) # => (B, Z)
            z_probs = z_probs.flatten(start_dim=1, end_dim=2) # => (B, Z)

        # in training and inference mode: get prior from h
        z_prior_logits = self.dynamics_mlp(h) # => (B, Z)
        z_prior, z_prior_probs = self.categorical(z_prior_logits) # => (B, NUM_CATEGORICALS, NUM_CLASSES)
        z_prior = z_prior.flatten(start_dim=1, end_dim=2) # => (B, Z)
        z_prior_probs = z_prior_probs.flatten(start_dim=1, end_dim=2) # => (B, Z)

        # use the posterior z in training mode and prior in inference
        state = torch.cat((h, z if training else z_prior), dim=1) # (B, STATE) with STATE=H+Z

        # apply heads
        if training:
            x_reconstruction = self.vae.decode(h, z) # (B, C, H, W)
            reward_pred = self.reward_mlp(state).view(batch_size) # (B,)
            continue_prob = self.continue_mlp(state).view(batch_size) # (B,)
            continue_pred = torch.bernoulli(continue_prob).view(batch_size) # (B,)
        
        ### return state, h, z, reward_pred, continue_prob, continue_pred, x_reconstruction
        return {
            "state": state, # (h_t, z_t or z_prior_t in inference)
            "h": h, # h_t
            "z": z if training else z_prior, # one-hot sample
            "z_pred": z_prior, # one-hot sample
            "z_probs": z_probs if training else z_prior_probs, # for the KL loss
            "z_pred_probs": z_prior_probs, # for the KL loss
            "reward_pred": reward_pred if training else None,
            "continue_prob": continue_prob if training else None,
            "continue_pred": continue_pred if training else None,
            "x": x if training else None,
            "x_reconstruction": x_reconstruction  if training else None,
            }
        
    
    def step(self, action, h, z):
        """
        Performs a step with the GRU to get the next hidden state.

        Args:
            action (torch.Tensor):
                Shape: (B, A)
            h (torch.Tensor): The deterministic recurrent hidden state h_{t}
                Shape: (B, H)
            z (torch.Tensor): The stochastic hidden state z_{t}
                Shape: (B, Z)
        
        Returns:
            h (torch.Tensor): The deterministic recurrent hidden state h_{t+1} as input for the next time step
                Shape: (B, H)
        """
        # convert the action to a tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        action = action.to(self.device).view(-1, self.A) # (B, A)
        h = h.to(self.device)
        z = z.to(self.device)

        batch_size = action.shape[0]
        assert action.shape == torch.Size([batch_size, self.A])
        assert h.shape == torch.Size([batch_size, self.H])
        assert z.shape == torch.Size([batch_size, self.Z])

        # concatenate the rnn_input and apply RNN to obtain the next hidden state
        rnn_input = torch.cat((action, h, z), 1).float() # (B, RNN_INPUT) with RNN_INPUT=A+H+Z

        # NEW: Linear In
        rnn_input = self.pre_gru_linear(rnn_input)

        rnn_input = rnn_input.view(1, batch_size, -1)  # (1, B, A+H+Z)
        h = h.unsqueeze(0) # (1, B, H)

        _, h = self.rnn(rnn_input, h)
        h = h.squeeze(0) # (B, H)

        # test the shapes of the stoch and deter hidden states
        assert h.shape == torch.Size([batch_size, self.H]) # should be (B, H)
        return h


    
    def get_losses(self, step_dict):
        """
        Computes the loss for one time step for the RSSM model given the step dictionary.

        Args:
            step_dict (dict): Dictionary that should include:
                x (torch.Tensor): Preprocessed observation.
                    Shape: (B, C, H, W)
                x_reconstruction (torch.Tensor): Reconstructed observation.
                    Shape: (B, C, H, W)
                reward_target (torch.Tensor):
                    Shape: (B,)
                reward_pred (torch.Tensor):
                    Shape: (B,)
                continue_target (torch.Tensor):
                    Shape: (B,)
                continue_prob (torch.Tensor):
                    Shape: (B,)
                z_probs (torch.Tensor):
                    Shape: (B, Z)
                z_pred_probs (torch.Tensor):
                    Shape: (B, Z)
        
        Returns:
        losses (dict): Dictionary containing the following losses:
            loss (torch.Tensor): Total RSSM loss
            image_loss (torch.Tensor): Reconstruction loss
            reward_loss (torch.Tensor): MSE loss for the predicted rewards
            continue_loss (torch.Tensor): BCE loss for the predicted continue probabilities
            dyn_loss (torch.Tensor): KL divergence loss between the posterior and prior probabilities
            rep_loss (torch.Tensor): KL divergence loss between the prior and posterior probabilities
        """
        ### x_target, x_pred, reward_target, reward_pred, continue_target, continue_prob, z_pred, z

        x_target = step_dict["x"]
        x_pred = step_dict["x_reconstruction"]
        reward_target = step_dict["reward_target"]
        reward_pred = step_dict["reward_pred"]
        continue_target = step_dict["continue_target"]
        continue_prob = step_dict["continue_prob"]
        z_probs = step_dict["z_probs"]
        z_pred_probs = step_dict["z_pred_probs"]

        batch_size = x_target.shape[0]

        # assert that all values are present in the state_dict
        assert x_target.shape == torch.Size([batch_size, 1 if self.grayscale else 3, *self.img_size])
        assert x_pred.shape == torch.Size([batch_size, 1 if self.grayscale else 3, *self.img_size])
        assert reward_target.shape == torch.Size([batch_size])
        assert reward_pred.shape == torch.Size([batch_size])
        assert continue_target.shape == torch.Size([batch_size])
        assert continue_prob.shape == torch.Size([batch_size])
        assert z_pred_probs.shape == torch.Size([batch_size, self.Z])
        assert z_probs.shape == torch.Size([batch_size, self.Z])
        
        image_loss = F.mse_loss(x_target, x_pred, reduction="mean")
        reward_loss = F.mse_loss(reward_target.squeeze(), reward_pred.squeeze(), reduction="mean")
        continue_loss = F.binary_cross_entropy(continue_prob.squeeze(), continue_target.squeeze())
        
        # DreamerV3 KL losses: regularize the posterior (z) towards the prior (z_pred)
        kld = dist.kl.kl_divergence
        
        # define the distributions with grad
        dist_z = dist.OneHotCategorical(probs=z_probs.view(-1, self.num_categoricals, self.num_classes))
        dist_z_pred = dist.OneHotCategorical(probs=z_pred_probs.view(-1, self.num_categoricals, self.num_classes))
        
        # define the distributions without grad
        dist_z_sg = dist.OneHotCategorical(probs=z_probs.detach().view(-1, self.num_categoricals, self.num_classes))
        dist_z_pred_sg = dist.OneHotCategorical(probs=z_pred_probs.detach().view(-1, self.num_categoricals, self.num_classes))

        # calculate the mean KL-divergence across the categoricals (and apply the "free bits" method)
        dyn_loss = torch.max(torch.tensor(self.free_nats), torch.mean(kld(dist_z_sg, dist_z_pred)))
        rep_loss = torch.max(torch.tensor(self.free_nats), torch.mean(kld(dist_z, dist_z_pred_sg)))
     
        # calculate the combined loss
        loss = self.pred_loss_coeff * (image_loss + reward_loss + continue_loss) + \
                self.dyn_loss_coeff * dyn_loss + self.rep_loss_coeff * rep_loss
        
        return {"loss": loss, "image_loss": image_loss, "reward_loss": reward_loss, 
                "continue_loss": continue_loss, "dyn_loss": dyn_loss, "rep_loss": rep_loss}
    
    def update_parameters(self, loss):
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm, norm_type=2)
        self.optim.step()
    
    def save_weights(self, filename):
        os.makedirs("weights", exist_ok=True)
        if filename:
            torch.save(self.state_dict(), f"weights/{filename}")
        else:
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