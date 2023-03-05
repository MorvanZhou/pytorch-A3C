"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import math
import os

import gym
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Linear

from shared_adam import SharedAdam
from utils import v_wrap, set_init, push_and_pull, record, pickle_results, name_from_config

NAME_PENDULUM = "Pendulum-v0"

H_ENV_NAME = "ENV_NAME"
H_STATE_SIZE = "N_S"
H_ACTION_SIZE = "N_A"
H_GAMMA = "GAMMA"
H_LAMBDA = "LAMBDA"
H_USE_GAE = "USE_GAE"
H_UPPER_BOUND = "U_BOUND"
H_LOWER_BOUND = "L_BOUND"
H_UPDATE_GLOBAL_ITER = "UPDATE_GLOBAL_ITER"
H_MAX_EP_STEP = "MAX_EP_STEP"
H_MAX_EP = "MAX_EP"
H_RENDER = "render"


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        hidden_size = 64
        self.config = config
        self.layer_1: Linear = Linear(config[H_STATE_SIZE], hidden_size)
        self.layer_2: Linear = Linear(hidden_size, hidden_size)
        self.mu: Linear = Linear(hidden_size, config[H_ACTION_SIZE])
        self.sigma: Linear = Linear(hidden_size, config[H_ACTION_SIZE])
        self.v = nn.Linear(hidden_size, 1)
        set_init([self.layer_1, self.layer_2, self.mu, self.sigma, self.v])

    def forward(self, x):
        x: torch.Tensor = F.relu(self.layer_1(x))
        x: torch.Tensor = F.relu(self.layer_2(x))
        mu: torch.Tensor = torch.tanh(self.mu(x))
        sigma: torch.Tensor = F.softplus(self.sigma(x)) + 0.0001  # Don't want a zero here
        values: torch.Tensor = self.v(x)

        return mu, sigma, values

    def choose_action(self, s):
        self.eval()
        mu, sigma, _ = self.forward(s)
        if self.config[H_ENV_NAME] == NAME_PENDULUM:
            mu = mu.view(1, ).data
            sigma = sigma.view(1, ).data
        m = Normal(mu, sigma)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        if self.config[H_USE_GAE]:
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
            advantage[i] = advantage[i] + self.config[H_GAMMA] * self.config[H_LAMBDA] * advantage[i + 1]

        return advantage


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, config):
        super(Worker, self).__init__()
        self.config = config
        self.name = name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(config)           # local network
        self.env = gym.make(config[H_ENV_NAME]).unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < self.config[H_MAX_EP]:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(self.config[H_MAX_EP_STEP]):
                if self.config[H_RENDER] and self.name == "w00":
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a.clip(self.config[H_LOWER_BOUND], self.config[H_UPPER_BOUND]))
                if t == self.config[H_MAX_EP_STEP] - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.config[H_UPDATE_GLOBAL_ITER] == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, self.config[H_GAMMA], self.config[H_UPPER_BOUND])
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


def init_config(gamma, lam, max_ep, env_name, use_gae, render):
    config = dict()
    config[H_GAMMA] = gamma
    config[H_LAMBDA] = lam
    config[H_MAX_EP] = max_ep
    config[H_ENV_NAME] = env_name
    config[H_USE_GAE] = use_gae
    config[H_UPDATE_GLOBAL_ITER] = 200
    config[H_MAX_EP_STEP] = 500
    config[H_RENDER] = render

    env = gym.make(env_name)
    config[H_STATE_SIZE] = env.observation_space.shape[0]
    config[H_ACTION_SIZE] = env.action_space.shape[0]
    config[H_UPPER_BOUND] = float(env.action_space.high[0])
    config[H_LOWER_BOUND] = float(env.action_space.low[0])
    env.close()
    return config


def a3c(cpu_count, gamma, lam, max_ep, env_name, use_gae, render, plot=False):

    os.environ["OMP_NUM_THREADS"] = "1"

    config = init_config(gamma, lam, max_ep, env_name, use_gae, render)

    gnet = Net(config)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, f"w{i:02}", config) for i in range(cpu_count)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    if plot:
        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()

    filename = name_from_config(config[H_ENV_NAME],
                                config[H_MAX_EP],
                                config[H_USE_GAE],
                                config[H_GAMMA],
                                config[H_LAMBDA])
    pickle_results(res, filename)
