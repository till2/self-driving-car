import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions as dist

from .mlp import MLP
from .utils import load_config, to_np

class ContinuousActorCritic(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        config = load_config()

        # if config["toy_env"]:
        #     self.n_features = config["Z"]
        # else:
        #     self.n_features = config["H"] + config["Z"]

        self.n_features = config["Z"] # for using with an autoencoder
        self.n_actions = config["A"]

        # For testing on Pendulum-v1:
        # self.n_features = 3
        # self.n_actions = 1

        # hyperparameters
        self.gamma = config["gamma"]
        self.lam = config["lam"]
        self.ent_coef = config["ent_coef"]
        self.n_envs = config["n_envs"]
        self.action_clip = config["action_clip"]
        self.critic_lr = config["critic_lr"]
        self.actor_lr = config["actor_lr"]
        self.max_grad_norm = config["max_grad_norm"]
        self.device = config["device"]

        # define actor and critic nets
        self.critic = MLP(input_dims=self.n_features, output_dims=1, out_type="linear")
        self.actor = MLP(input_dims=self.n_features, output_dims=self.n_actions, out_type="gaussian")
        
        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.to(self.device)
    
    def get_action(self, x):

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device)
        
        mu, var = self.actor(x)
        var = torch.clamp(var, min=1e-8)
        std = torch.sqrt(var)
        std = torch.clamp(std, min=1e-8, max=1)
        
        action_pd = dist.Normal(mu, std)
        actions = action_pd.sample()

        action = torch.tanh(actions)
        actor_entropy = action_pd.entropy()
        log_probs = action_pd.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + 1e-8) # update logprob because of tanh after sampling
        log_prob = log_probs.sum(0, keepdim=True) # reduce to a scalar (the probs would multiply, but logprobs add)

        return action, log_prob, actor_entropy
    
    def get_loss(
        self,
        ep_rewards: torch.Tensor,
        ep_log_probs: torch.Tensor,
        ep_value_preds: torch.Tensor,
        last_value_pred: torch.Tensor,
        ep_entropies: torch.Tensor,
        ep_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of actor and critic using GAE.
        """

        # append the last value pred to the value preds tensor
        ep_value_preds = torch.cat((ep_value_preds, last_value_pred.unsqueeze(1).detach()), dim=0)

        # set up tensors for the advantage calculation
        returns = torch.zeros_like(ep_rewards).to(self.device)
        advantages = torch.zeros_like(ep_rewards).to(self.device)
        next_advantage = torch.zeros_like(last_value_pred)

        # calculate advantages using GAE
        for t in reversed(range(len(ep_rewards))):
            returns[t] = ep_rewards[t] + self.gamma * ep_value_preds[t+1] * ep_masks[t]
            td_error = returns[t] - ep_value_preds[t]
            advantages[t] = next_advantage = td_error + self.gamma * self.lam * next_advantage * ep_masks[t]
        
        # calculate the critic loss
        critic_loss = advantages.pow(2).mean()

        # calculate the actor loss using the policy gradient theorem and give an entropy bonus
        actor_loss = -(ep_log_probs * advantages.detach()).mean() - self.ent_coef * ep_entropies.mean()
        return critic_loss, actor_loss

    
    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:

        config = load_config()

        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm, norm_type=2)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm, norm_type=2)
        self.actor_optim.step()

    def save_weights(self):
        os.makedirs("weights", exist_ok=True)
        base_path = "weights/ContinuousActorCritic"
        index = 0
        while os.path.exists(f"{base_path}_{index}"):
            index += 1
        torch.save(self.state_dict(), f"{base_path}_{index}")

        
    def load_weights(self, path="weights/ContinuousActorCritic", eval_mode=False):
        self.load_state_dict(torch.load(path))
        if eval_mode:
            print("Set Agent to eval mode.")
            self.eval()