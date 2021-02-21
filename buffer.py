from collections import namedtuple
from collections import deque
import torch
import numpy.random as nr

Transitions = namedtuple('Transitions', ['obs', 'action', 'reward', 'next_obs', 'done'])


class ReplayBuffer:
    def __init__(self, config):
        replay_buffer_size = config['replay_buffer_size']
        seed = config['seed']
        nr.seed(seed)

        self.replay_buffer_size = replay_buffer_size
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)

    def append_memory(self, obs, action, reward, next_obs, done: bool):
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)

    def sample(self, batch_size):
        buffer_size = len(self.obs)

        idx = nr.choice(buffer_size,
                        size=min(buffer_size, batch_size),
                        replace=False)
        t = Transitions
        t.obs = torch.stack(list(map(self.obs.__getitem__, idx)))
        t.action = torch.stack(list(map(self.action.__getitem__, idx)))
        t.reward = torch.stack(list(map(self.reward.__getitem__, idx)))
        t.next_obs = torch.stack(list(map(self.next_obs.__getitem__, idx)))
        t.done = torch.tensor(list(map(self.done.__getitem__, idx)))[:, None]
        return t

    def clear(self):
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)
