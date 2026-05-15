import math
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):

    def layer_init(self, layer, gain=math.sqrt(2)):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0)
        return layer

    def __init__(self, obs_dim, num_actions, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.policy_head = self.layer_init(nn.Linear(hidden_dim, num_actions), gain=0.01)
        self.value_head = self.layer_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, obs):
        x = self.trunk(obs)
        return self.policy_head(x), self.value_head(x).squeeze(-1)
    
    def act(self, obs, mask):
        policy_logits, value = self(obs)
        policy_logits = policy_logits.masked_fill(~mask.bool(), float("-inf"))
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, action, mask):
        policy_logits, value = self(obs)
        policy_logits = policy_logits.masked_fill(~mask.bool(), float("-inf"))
        dist = Categorical(logits=policy_logits)
        entropy = dist.entropy()
        return entropy, dist.log_prob(action), value

