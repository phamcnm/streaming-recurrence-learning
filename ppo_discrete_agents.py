import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from recurrent_wrappers.create_models import create_model

def create_agent(envs, env_type, **kwargs):
    if env_type == 'ale_py':
        return AgentAtari(n_actions=envs.single_action_space.n, **kwargs)
    elif env_type == 'mujoco':
        return AgentContinuous(n_obs=np.array(envs.single_observation_space.shape).prod(), n_actions=np.prod(envs.single_action_space.shape), **kwargs)
    else:
        return AgentDiscrete(n_obs=np.prod(envs.single_observation_space.shape), n_actions=envs.single_action_space.n, **kwargs)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def init_weights(m, std=1.0, bias_const=0.0):
    if isinstance(m, nn.LayerNorm):
        if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        if m.bias   is not None: nn.init.constant_(m.bias,   0.0)
        return
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, std)
        if m.bias is not None:
            nn.init.constant_(m.bias, bias_const)
        return
    for _, p in m.named_parameters(recurse=False):
        if p is None: continue
        if p.ndim < 2:
            nn.init.constant_(p, bias_const)
        else:
            nn.init.orthogonal_(p, std)

class AgentDiscrete(nn.Module):
    def __init__(self, n_obs, n_actions, rnn_unit="glru", arch='mynet', d_model=64, d_state=64, rnn_mode="sequential", ponder_eps=0.01, **kwargs):
        super().__init__()
        self.rnn_type = rnn_unit
        self.rnn_mode = rnn_mode

        self.actor_embedder = layer_init(nn.Linear(n_obs, d_model), std=1)
        self.actor, self.outdim = create_model(
            name=rnn_unit,
            embed_dim=d_model,
            hidden_dim=d_state,
            arch=arch,
            rnn_mode=rnn_mode,
            use_layernorm=True)
        self.actor_readout = layer_init(nn.Linear(self.outdim, n_actions), std=0.01)
        self.actor.apply(init_weights)

        self.critic_embedder = layer_init(nn.Linear(n_obs, d_model), std=1)
        self.critic, self.outdim = create_model(
            name=rnn_unit,
            embed_dim=d_model,
            hidden_dim=d_state,
            arch=arch,
            rnn_mode=rnn_mode,
            use_layernorm=True)
        self.critic_readout = layer_init(nn.Linear(self.outdim, 1), std=1.0)
        self.critic.apply(init_weights)
        
    def get_states(self, x, rnn_state, done, network='actor', rnn_burnin=0.0, batch_size=1, **kwargs):
        if network == 'actor':
            embedder, rnn = self.actor_embedder, self.actor
        elif network == 'critic':
            embedder, rnn = self.critic_embedder, self.critic
        x = embedder(x)
        x = F.relu(x) if self.rnn_type in ["gru", "rnn"] else x
        if self.rnn_type == 'mlp':
            return rnn(x)
        seq_len = x.shape[0] // batch_size
        x = x.reshape((seq_len, batch_size, -1))
        done_reshaped = done.reshape((seq_len, batch_size))
        x, rnn_state, aux_data = rnn(x, hidden=rnn_state, done=done_reshaped, **kwargs)
        x = x.reshape(-1, self.outdim)
        return x, rnn_state, aux_data

    def get_value(self, x, rnn_state=None, done=None, rnn_burnin=0.0, batch_size=1, **kwargs):
        x, critic_rnn_state, _ = self.get_states(
            x,
            rnn_state,
            done,
            network='critic',
            rnn_burnin=rnn_burnin,
            batch_size=batch_size,
            **kwargs,
        )
        return self.critic_readout(x), critic_rnn_state

    def get_action(self, x, rnn_state=None, done=None, action=None, rnn_burnin=0.0, batch_size=1, **kwargs):
        x, rnn_state, aux_data = self.get_states(
            x,
            rnn_state,
            done,
            network='actor',
            rnn_burnin=rnn_burnin,
            batch_size=batch_size,
            **kwargs,
        )
        logits = self.actor_readout(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        else:
            action = action.long()
        return action, probs.log_prob(action), probs.entropy(), rnn_state, aux_data

class AgentAtari(nn.Module):
    def __init__(self, n_actions, **kwargs):
        super().__init__()
        self.embdder = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x, **kwargs):
        x = self.embdder(x / 255.0)
        return self.critic(x), None

    def get_action(self, x, action=None, **kwargs):
        x = self.embdder(x / 255.0)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        else:
            action = action.long()
        return action, probs.log_prob(action), probs.entropy(), None, None

class AgentContinuous(nn.Module):
    def __init__(self, n_obs, n_actions, **kwargs):
        super().__init__()
        # envs.single_observation_space.shape
        # envs.single_action_space.shape
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(n_actions)))

    def get_value(self, x, **kwargs):
        return self.critic(x), None

    def get_action(self, x, action=None, **kwargs):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), None, None
