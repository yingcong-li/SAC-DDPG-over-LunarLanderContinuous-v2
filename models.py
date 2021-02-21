import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import Tuple


class ValueNetwork(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        if not isinstance(dim_obs, int):
            TypeError('dimension of observation must be int')
        if not isinstance(dims_hidden_neurons, tuple):
            TypeError('dimensions of hidden neurons must be tuple of int')

        super(ValueNetwork, self).__init__()
        self.num_layers = len(dims_hidden_neurons)

        n_neurons = (dim_obs,) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, observation: torch.Tensor):
        x = observation
        for i in range(self.num_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        return self.output(x)


class QNetwork(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        if not isinstance(dim_obs, int):
            TypeError('dimension of observation must be int')
        if not isinstance(dim_action, int):
            TypeError('dimension of action must be int')
        if not isinstance(dims_hidden_neurons, tuple):
            TypeError('dimensions of hidden neurons must be tuple of int')

        super(QNetwork, self).__init__()
        self.num_layers = len(dims_hidden_neurons)
        self.dim_obs = dim_obs
        self.dim_action = dim_action

        n_neurons = (dim_obs + dim_action,) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, observation: torch.Tensor, action):
        x = torch.cat([observation.view(-1, self.dim_obs), action.view(-1, self.dim_action)], dim=1)
        for i in range(self.num_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        return self.output(x)


class PolicyNetwork(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        if not isinstance(dim_obs, int):
            TypeError('dimension of observation must be int')
        if not isinstance(dims_hidden_neurons, tuple):
            TypeError('dimensions of hidden neurons must be tuple of int')

        super(PolicyNetwork, self).__init__()
        self.num_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action
        self.reparam_noise = 1e-6

        n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.mean = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        self.var = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.uniform_(self.mean.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.mean.bias, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.var.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.var.bias, a=-3e-3, b=3e-3)

    def forward(self, observation: torch.Tensor):
        x = observation
        for i in range(self.num_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        mean = self.mean(x)
        var = torch.clamp(self.var(x), min=self.reparam_noise, max=1)
        return mean, var

    def sample_normal(self, state, reparam=False):
        mean, var = self.forward(state)
        probs = Normal(mean, var)

        if reparam:
            actions = probs.rsample()
        else:
            actions = probs.sample()

        action = torch.tanh(actions)
        log_probs = probs.log_prob(actions)
        log_probs -= torch.log((1 - torch.tanh(actions).pow(2)) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs


class Actor(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        if not isinstance(dim_obs, int):
            TypeError('dimension of observation must be int')
        if not isinstance(dims_hidden_neurons, tuple):
            TypeError('dimensions of hidden neurons must be tuple of int')

        super(Actor, self).__init__()
        self.num_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action

        self.mu, self.sigma, self.theta, self.dt = .0, .2, .15, 1e-2
        self.x = torch.zeros(self.dim_action)

        n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.mean = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.uniform_(self.mean.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.mean.bias, a=-3e-3, b=3e-3)

    def forward(self, observation: torch.Tensor):
        x = observation
        for i in range(self.num_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        mean = torch.tanh(self.mean(x))
        return mean

    def sample_normal(self, state):
        mean = self.forward(state).data.clone()
        noise = self.x + self.theta*self.dt*(self.mu-self.x)+self.sigma*torch.normal(0, self.dt, size=(self.dim_action,))
        self.x = noise
        return  mean + noise

    def process_reset(self):
        self.x = torch.zeros(self.dim_action)

