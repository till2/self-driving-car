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
                 vf_loss_coeff=0.5,
                 critic_lr=0.005,
                 actor_lr=0.001,
                 action_clip=1,
                ):
        super().__init__()
        
        # hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.entropy_coeff = entropy_coeff
        self.vf_loss_coeff = vf_loss_coeff
        self.n_envs = n_envs
        self.action_clip = action_clip # action range: [-action_clip, action_clip]

        # discretize the action space
        # self.possible_actions = [-2, -0.5, -0.1, 0, 0.1, 0.5, 2]
        # self.n_buckets = 7
        # self.action_min = -2
        # self.action_max = 2
        
        # define actor and critic nets
        self.critic = MLP(input_dims=n_features, output_dims=1, out_type="linear")
        self.actor = MLP(input_dims=n_features, output_dims=n_actions, out_type="gaussian")
        # self.actor = MLP(input_dims=n_features, output_dims=n_actions*self.n_buckets, out_type="linear")

        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
    
    def get_action(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        mu, var = self.actor(x)
        mu = torch.clamp(mu, min=-2, max=2)
        var = torch.clamp(var, min=1e-8, max=1)
        std = torch.sqrt(var)
        
        action_pd = dist.Normal(mu, std)
        action = action_pd.sample() # rsample for reparameterization trick

        # action = torch.tanh(action)
        action = torch.clip(action, -self.action_clip, self.action_clip) # you could also put another tanh here
        
        log_prob = action_pd.log_prob(action)

        # SAC
        # log_prob -= torch.log(1 - action.pow(2) + 1e-8) # SAC
        # log_prob.sum(1, keepdim=True)

        actor_entropy = action_pd.entropy()
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
        # T = len(ep_rewards)
        # advantages = torch.zeros_like(ep_rewards)

        # # compute the advantages using GAE
        # gae = 0.0
        # for t in reversed(range(T - 1)):
        #     td_error = (
        #         ep_rewards[t] + self.gamma * ep_masks[t] * ep_value_preds[t + 1] - ep_value_preds[t]
        #     )
        #     gae = td_error + self.gamma * self.lam * ep_masks[t] * gae
        #     advantages[t] = gae

        # # calculate the loss of the minibatch for actor and critic
        # critic_loss = advantages.pow(2).mean()

        # # give a bonus for higher entropy to encourage exploration
        # actor_loss = -(advantages.detach() * ep_log_probs).mean() - self.entropy_coeff * ep_entropies.mean()
        
        # return critic_loss, actor_loss

        #
        ###
        #

        # # append the next_value_pred to value preds tensor
        ep_value_preds = torch.cat((ep_value_preds, last_value_pred.T.detach()), dim=0)

        # # set up tensors for the advantage calculation
        returns = torch.zeros_like(ep_rewards)
        advantages = torch.zeros_like(ep_rewards)
        next_advantage = torch.zeros_like(last_value_pred).T

        # calculate advantages using GAE
        for t in reversed(range(len(ep_rewards))):
            returns[t] = ep_rewards[t] + self.gamma * ep_value_preds[t+1] * ep_masks[t]
            td_error = returns[t] -  ep_value_preds[t]
            advantages[t] = next_advantage = td_error + self.gamma * self.lam * next_advantage * ep_masks[t]

        # # calculate the critic loss (without the additional value pred that was needed for the advantage calculation)
        # # critic_loss = F.mse_loss(ep_value_preds[:-1], returns)
        critic_loss = advantages.pow(2).mean()

        # calculate the actor loss using the policy gradient theorem and give an entropy bonus
        actor_loss = -(ep_log_probs * advantages.detach()).mean() - self.entropy_coeff * ep_entropies.mean()

        return critic_loss, actor_loss
    
    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:

        #total_loss = self.vf_loss_coeff * critic_loss + actor_loss
        #total_loss.backward()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0, norm_type=2)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0, norm_type=2)
        self.actor_optim.step()