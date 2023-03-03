"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import math
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Linear

from shared_adam import SharedAdam
from utils import v_wrap, set_init, push_and_pull, record

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GA3C_GAE"] = "1"

# CPU_COUNT = 1
CPU_COUNT = mp.cpu_count()
UPDATE_GLOBAL_ITER = 500
GAMMA = 0.9
MAX_EP = 1000
MAX_EP_STEP = 200

ENV = 'Pendulum-v0'
# ENV = 'Humanoid-v3'
# ENV = 'HumanoidStandup-v2'
# ENV = 'Ant-v3'

env = gym.make(ENV)
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
U_BOUND = float(env.action_space.high[0])
L_BOUND = float(env.action_space.low[0])
env.close()


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(Net, self).__init__()

        self.layer_1: Linear = Linear(input_size, hidden_size)
        self.layer_2: Linear = Linear(hidden_size, hidden_size)
        self.mu: Linear = Linear(hidden_size, output_size)
        self.sigma: Linear = Linear(hidden_size, output_size)
        self.v = nn.Linear(hidden_size, 1)
        set_init([self.layer_1, self.layer_2, self.mu, self.sigma, self.v])

        self.gam = 0.9
        self.lam = 0.9

    def forward(self, x):
        x: torch.Tensor = F.relu(self.layer_1(x))
        x: torch.Tensor = F.relu(self.layer_2(x))
        mu: torch.Tensor = torch.tanh(self.mu(x))
        sigma: torch.Tensor = F.softplus(self.sigma(x)) + 0.0001  # Don't want a zero here
        values: torch.Tensor = self.v(x)

        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        if ENV == "Pendulum-v0":
            m = Normal(mu.view(1, ).data, sigma.view(1, ).data)
        else:
            m = Normal(mu, sigma)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        if os.environ["GA3C_GAE"] == "1":
            td = self.compute_gae(v_t, values)
        else:
            td = v_t - values
        c_loss = td.pow(2)

        m = Normal(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss

    def compute_gae(self,
                    values_seen: torch.Tensor,
                    values_predicted: torch.Tensor) -> torch.Tensor:

        advantage = values_seen.detach() - values_predicted.detach()
        for i in range(len(advantage) - 2, -1, -1):
            advantage[i] = advantage[i] + self.gam * self.lam * advantage[i + 1]

        return advantage


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = gym.make(ENV).unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                # if self.name == 'w0':
                #     self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a.clip(L_BOUND, U_BOUND))
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, f"w{i:02}") for i in range(CPU_COUNT)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
