import torch
import torch.nn as nn
from models import ValueNetwork, QNetwork, PolicyNetwork

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)


class SAC():

    def __init__(self, config):

        torch.manual_seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.tau = config['tau']  # target smoothing coefficient
        self.discount = config['discount']  # discount factor
        self.batch_size = config['batch_size']  # mini batch size
        self.reward_scale = config['reward_scale']  # reward scale

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.V = ValueNetwork(dim_obs=self.dim_obs,
                              dims_hidden_neurons=self.dims_hidden_neurons)
        self.V_tar = ValueNetwork(dim_obs=self.dim_obs,
                              dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q1 = QNetwork(dim_obs=self.dim_obs,
                           dim_action=self.dim_action,
                           dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q2 = QNetwork(dim_obs=self.dim_obs,
                           dim_action=self.dim_action,
                           dims_hidden_neurons=self.dims_hidden_neurons)
        self.Policy = PolicyNetwork(dim_obs=self.dim_obs,
                                    dim_action=self.dim_action,
                                    dims_hidden_neurons=self.dims_hidden_neurons)

        self.optimizer_V = torch.optim.Adam(self.V.parameters(), lr=self.lr)
        self.optimizer_Q1 = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.optimizer_Q2 = torch.optim.Adam(self.Q2.parameters(), lr=self.lr)
        self.optimizer_Policy = torch.optim.Adam(self.Policy.parameters(), lr=self.lr)
        self.training_step = 0
        self.loss_func = nn.MSELoss()

        self.V_tar.load_state_dict(self.V.state_dict())

    def update(self, buffer):
        t = buffer.sample(self.batch_size)

        done = t.done
        s = t.obs
        a = t.action

        sp = t.next_obs
        r = t.reward

        self.training_step += 1

        self.update_V(s)
        self.update_Q(s, a, sp, r, done)
        self.update_Policy(s)

    def update_V(self, s):
        value = self.V(s)
        a, log = self.take_action(s, reparam=False)
        q_value = torch.min(self.Q1(s, a), self.Q2(s, a))


        loss = 0.5 * self.loss_func(value, q_value -log)
        self.optimizer_V.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer_V.step()

        state_dict = self.V.state_dict().copy()
        state_dict_ = self.V_tar.state_dict().copy()
        for n, p in state_dict.items():
            state_dict_[n] = self.tau * p + (1-self.tau) * state_dict_[n]
        self.V_tar.load_state_dict(state_dict_)

    def update_Q(self, s, a, sp, r, done):
        q1, q2 = self.Q1(s, a), self.Q2(s, a)
        value_ = self.V_tar(sp)
        loss1 = 0.5 * self.loss_func(q1, self.reward_scale * r + ~done * self.discount * value_)
        loss2 = 0.5 * self.loss_func(q2, self.reward_scale * r + ~done * self.discount * value_)
        self.optimizer_Q1.zero_grad()
        loss1.backward(retain_graph=True)
        self.optimizer_Q1.step()
        self.optimizer_Q2.zero_grad()
        loss2.backward()
        self.optimizer_Q2.step()

    def update_Policy(self, s):
        action, log = self.take_action(s, reparam=True)
        q_value = torch.min(self.Q1(s, action), self.Q2(s, action))
        loss = torch.mean(log - q_value)
        self.optimizer_Policy.zero_grad()
        loss.backward()
        self.optimizer_Policy.step()

    def take_action(self, state, reparam=False):
        state = torch.tensor(state, dtype=torch.double)
        return self.Policy.sample_normal(state, reparam=reparam)