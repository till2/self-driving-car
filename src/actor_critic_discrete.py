import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions as dist

from .mlp import MLP
from .utils import load_config, to_np, symlog, symexp, twohot_encode, ExponentialMovingAvg, ActionExponentialMovingAvg

class DiscreteActorCritic(nn.Module):
    
    def __init__(self, n_features=None, n_actions=None):
        super().__init__()
        
        config = load_config()

        # For using with the world model:
        self.n_features = config["H"] + config["Z"] if n_features is None else n_features
        self.n_actions = config["A"] if n_actions is None else n_actions
        
        print(f"Initializing agent with {self.n_features} features and {self.n_actions} actions.")

        # hyperparameters
        self.gamma = config["gamma"]
        self.lam = config["lam"]
        self.ent_coef = config["ent_coef"]
        self.n_envs = config["n_envs"]
        self.critic_lr = config["critic_lr"]
        self.actor_lr = config["actor_lr"]
        self.max_grad_norm = config["max_grad_norm"]
        self.device = config["device"]

        # critic buckets
        self.min_bucket = config["min_bucket"]
        self.max_bucket = config["max_bucket"]
        self.num_buckets = config["num_buckets"]
        
        # use discrete action targets with EMA for smooth control
        self.use_action_ema = config["use_action_ema"]
        if self.use_action_ema:
            self.action_ema = ActionExponentialMovingAvg()
            self.action_buckets = torch.tensor([-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0]).to(self.device)
            self.n_action_buckets = len(self.action_buckets)

        # define actor and critic nets
        print("Initializing critic.")
        self.critic = MLP(input_dims=self.n_features, output_dims=config["num_buckets"], out_type="softmax", weight_init="final_layer_zeros")
        print("Initializing actor.")
        self.actor = MLP(input_dims=self.n_features, output_dims=self.n_actions*self.n_action_buckets, out_type="linear", weight_init="final_layer_zeros")
        
        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.to(self.device)
    
    def get_action(self, x):
        """
        Selects a continuous action between -1 and 1 based on the given observation. 

        Args:
            x (torch.Tensor or np.ndarray): The preprocessed observation.
                Shape: (B, NUM_FEATURES)

        Returns:
            action (torch.Tensor): The sampled action (with a stochastic policy).
                Shape: (B, A)
            log_prob (torch.Tensor): The logarithm of the probability of the generated action (required for the loss calculation).
                Shape: (B,)
            actor_entropy (torch.Tensor): The entropy of the actor's distribution (useful for encouraging exploration in the loss).
                Shape: (B,)

        Notes:
            - The generated action is smoothed using an exponential moving average to avoid jitter in the controls.
            - The batch size should stay consistent over the duration of training, 
              because new actions have to match the shape of old actions for the exponential moving average
        """

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)

        # get stochastic action
        action_logits = self.actor(x)
        action_logits = action_logits.view(-1, self.n_actions, self.n_action_buckets)
        action_pd = torch.distributions.Categorical(logits=action_logits) # implicitly uses softmax
        action_idxs = action_pd.sample()
        action_target = self.action_buckets[action_idxs] # is a vector
        actor_entropy = action_pd.entropy().sum(dim=1) # sum over entries of the action-vector
        log_prob = action_pd.log_prob(action_idxs).sum(dim=1) # sum over entries of the action-vector

        # update exponential moving average action for smooth control
        if self.use_action_ema:
            action = self.action_ema.get_smoothed_action(action_target)
        action = action.view(-1, self.n_actions) # (B,A)

        return action, log_prob, actor_entropy
        
    def apply_critic(self, x):
        """
        Applies the critic to a preprocessed observation.

        Args:
            x (torch.Tensor or np.ndarray): The preprocessed observation.
                Shape: (B, NUM_FEATURES)

        Returns:
            value_pred (torch.Tensor): The predicted original return. The prediction is symexp'd.
                Shape: (B,)
            critic_dist (torch.Tensor): The distribution over buckets predicted by the critic.
                Shape: (B, NUM_BUCKETS)

        Notes:
            - The critic is trained to predict symlog returns, so the prediction needs to be symexp'd.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)

        buckets = torch.linspace(self.min_bucket, self.max_bucket, self.num_buckets).to(self.device)
        value_pred = symexp(self.critic(x) @ buckets)
        value_pred = value_pred.view(-1) # (B,)
        critic_dist = self.critic(x)
        critic_dist = critic_dist.view(-1, self.num_buckets) # (B, num_buckets)
        return value_pred, critic_dist
    
    def get_loss(
        self,
        episode_batches: dict[torch.Tensor],
        last_value_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of critic and actor using GAE.

        Args:
            episode_batches: dict that includes:
                rewards (torch.Tensor):
                    Shape: (SEQ_LEN, B)
                log_probs (torch.Tensor):
                    Shape: (SEQ_LEN, B)
                value_preds (torch.Tensor):
                    Shape: (SEQ_LEN, B)
                critic_dists (torch.Tensor):
                    Shape: (SEQ_LEN, B, NUM_BUCKETS)
                entropies (torch.Tensor):
                    Shape: (SEQ_LEN, B)
                masks (torch.Tensor):
                    Shape: (SEQ_LEN, B)

            last_value_pred (torch.Tensor):
                Shape: (B,)

        Returns:
            critic_loss (torch.Tensor):
                Shape: Scalar (without shape)
            actor_loss (torch.Tensor):
                Shape: Scalar (without shape)
        """

        ep_rewards = episode_batches["rewards"]
        ep_log_probs = episode_batches["log_probs"]
        ep_value_preds = episode_batches["value_preds"]
        batch_critic_dists = episode_batches["critic_dists"]
        ep_entropies = episode_batches["entropies"]
        ep_masks = episode_batches["masks"]

        # FROM ORIGINAL:
        # append the last value pred to the value preds tensor
        last_value_pred = last_value_pred.view(1,-1).detach() # (1, B)
        ep_value_preds = torch.cat((ep_value_preds, last_value_pred), dim=0) # (SEQ_LEN+1, B)

        # set up tensors for the advantage calculation
        returns = torch.zeros_like(ep_rewards).to(self.device) # (SEQ_LEN, B)
        advantages = torch.zeros_like(ep_rewards).to(self.device) # (SEQ_LEN, B)
        next_advantage = torch.zeros_like(last_value_pred) # (1, B)

        # calculate advantages using GAE
        for t in reversed(range(len(ep_rewards))):
            returns[t] = ep_rewards[t] + self.gamma * ep_masks[t] * ep_value_preds[t+1]
            td_error = returns[t] - ep_value_preds[t]
            advantages[t] = next_advantage = td_error + self.gamma * self.lam * ep_masks[t] * next_advantage

        # categorical crossentropy (should be fine, I checked.)
        twohot_returns = torch.stack([twohot_encode(r) for r in returns]) # (SEQ_LEN, B, NUM_BUCKETS)        
        critic_loss = self._calculate_critic_loss(twohot_returns, batch_critic_dists)

        # calculate the actor loss using the policy gradient theorem and give an entropy bonus
        actor_loss = -(ep_log_probs * advantages.detach()).mean() - self.ent_coef * ep_entropies.mean()
        return critic_loss, actor_loss
    
    def _calculate_critic_loss(self, twohot_returns, batch_critic_dists):
        """
        Calculates the mean categorical cross-entropy loss of a batch of gradient-detached
        targets (twohot_returns) and the predictions (batch_critic_dists).

        Args:
            twohot_returns (torch.Tensor):
                Shape: (SEQ_LENGTH, B, NUM_BUCKETS)
        
        Returns:
            critic_loss (torch.Tensor):
                Shape: Scalar (without shape)
        """
        twohot_returns = twohot_returns.permute(1, 0, 2) # => (B, SEQ_LENGTH, NUM_BUCKETS)
        batch_critic_dists = batch_critic_dists.permute(1, 0, 2) # => (B, SEQ_LENGTH, NUM_BUCKETS)
        critic_loss = -torch.sum(twohot_returns.detach() * torch.log(batch_critic_dists), dim=(1, 2)) # => (B,)
        critic_loss = torch.mean(critic_loss) # (,)
        return critic_loss
    
    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:

        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm, norm_type=2)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm, norm_type=2)
        self.actor_optim.step()

    def save_weights(self, filename=None):
        os.makedirs("weights", exist_ok=True)
        if filename:
            print(f"Saving agent weights to weights/{filename}")
            torch.save(self.state_dict(), f"weights/{filename}")
        else:
            base_path = "weights/DiscreteActorCritic"
            index = 0
            while os.path.exists(f"{base_path}_{index}"):
                index += 1
            print(f"Saving agent weights to {base_path}_{index}")
            torch.save(self.state_dict(), f"{base_path}_{index}")
        
    def load_weights(self, path="weights/DiscreteActorCritic", eval_mode=False):
        self.load_state_dict(torch.load(path))
        if eval_mode:
            print("Set Agent to eval mode.")
            self.eval()