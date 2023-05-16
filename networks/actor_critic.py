import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions as dist

from .mlp import MLP

class ContinuousActorCritic(nn.Module):
    
    def __init__(self, 
                 n_features=3, 
                 n_actions=1,
                 n_envs=1,
                 gamma=0.999, 
                 lam=0.95, # lam=1 is Monte-Carlo (no bias, high variance), lam=0 is TD (high bias, no variance)
                 entropy_coeff=0.01, 
                 critic_lr=0.005,
                 actor_lr=0.001,
                 action_clip_min=-2,
                 action_clip_max=2
                ):
        super().__init__()
        
        # hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.entropy_coeff = entropy_coeff
        self.n_envs = n_envs
        self.action_clip_min = action_clip_min
        self.action_clip_max = action_clip_max
        
        # define actor and critic nets
        self.critic = MLP(input_dims=n_features, output_dims=1)
        self.actor = MLP(input_dims=n_features, output_dims=n_actions, out_type="gaussian")

        # define optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
    
    def get_action(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        mu, var = self.actor(x)
        std = torch.sqrt(var)
        
        action_pd = dist.Normal(mu, std)
        action = action_pd.sample()
        action = torch.clip(action, self.action_clip_min, self.action_clip_max)
        
        log_prob = action_pd.log_prob(action)
        actor_entropy = action_pd.entropy()
        
        return action, log_prob, actor_entropy
    
    def get_loss(
        self,
        ep_rewards: torch.Tensor,
        ep_log_probs: torch.Tensor,
        ep_value_preds: torch.Tensor,
        ep_entropies: torch.Tensor,
        ep_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of actor and critic using GAE.
        """
        T = len(ep_rewards)
        advantages = torch.zeros(T, self.n_envs, device=ep_rewards.device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                ep_rewards[t] + self.gamma * ep_masks[t] * ep_value_preds[t + 1] - ep_value_preds[t]
            )
            gae = td_error + self.gamma * self.lam * ep_masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = -(advantages.detach() * ep_log_probs).mean() - self.entropy_coeff * ep_entropies.mean()
        
        return critic_loss, actor_loss
    
    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0, norm_type=2)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0, norm_type=2)
        self.actor_optim.step()