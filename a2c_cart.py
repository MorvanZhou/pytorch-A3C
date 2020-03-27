"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

"""

import torch
import torch.nn as nn
from utils import v_wrap, set_init
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import numpy as np
import gym
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import matplotlib.pyplot as plt
import time

GAMMA = 0.9
MAX_EP = 5000



env = gym.make('CartPole-v0').unwrapped
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


def plotter():
    plt.plot(res)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()


def handleArguments():
    """Handles CLI arguments and saves them globally"""
    parser = argparse.ArgumentParser(
        description="Switch between modes in A2C or loading models from previous games")
    parser.add_argument("--demo_mode", "-d", help="Renders the gym environment", action="store_true")
    parser.add_argument("--load_model", "-l", help="Loads the model of previously gained training data", action="store_true")
    global args
    args = parser.parse_args()


def push_and_pull(opt, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = gnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = gnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    opt.step()

    # pull global parameters
    gnet.load_state_dict(gnet.state_dict())

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 24)
        self.pi2 = nn.Linear(24, 24)
        self.pi3 = nn.Linear(24, a_dim)
        self.v1 = nn.Linear(s_dim, 24)
        self.v2 = nn.Linear(24, 24)
        self.v3 = nn.Linear(24, 1)
        set_init([self.pi1, self.pi2, self.pi3, self.v1, self.v2, self.v3])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        pi2 = F.relu(self.pi2(pi1))
        logits = self.pi3(pi2)
        v1 = F.relu(self.v1(x))
        v2 = F.relu(self.v2(v1))
        values = self.v3(v2)
        return logits, values

    def set_init(layers):
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight, nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(layer.bias, nn.init.calculate_gain('relu'))

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


if __name__ == "__main__":
    handleArguments()
    # load global network
    if args.load_model:
        gnet = Net(N_S, N_A)
        gnet = torch.load("./save_model/a3c_cart.pt")
        gnet.eval()
    else:
        gnet = Net(N_S, N_A)

    opt = SharedAdam(gnet.parameters(), lr=0.003, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r = 1, 0.

    res = []  # record episode reward to plot

    name = 'w00'
    total_step = 1
    stop_processes = False
    scores = []
    while global_ep < MAX_EP and stop_processes is False:
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0.
        while True:
            if name == 'w00' and args.demo_mode:
                env.render()
            a = gnet.choose_action(v_wrap(s[None, :]))
            s_, r, done, _ = env.step(a)
            if done: r = -1
            ep_r += r
            buffer_a.append(a)
            buffer_s.append(s)
            buffer_r.append(r)
            if done or ep_r == 700:  # update network
                # sync
                push_and_pull(opt, gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                buffer_s, buffer_a, buffer_r = [], [], []

                global_ep += 1

                if global_ep_r == 0.:
                    global_ep_r = ep_r
                else:
                    global_ep_r = global_ep_r * 0.99 + ep_r * 0.01
                print("w00 Ep:", global_ep, "| Ep_r: %.0f" % global_ep_r)
                scores.append(int(global_ep_r))
                if np.mean(scores[-min(10, len(scores)):]) >= 500:
                    stop_processes = True
                break
            s = s_
            total_step += 1

    if np.mean(scores[-min(10, len(scores)):]) >= 300:
        print("Save model")
        torch.save(gnet, "./save_model/a3c_cart.pt")
    else:
        print("Failed to train agent. Model was not saved")
    plotter()
    sys.exit()
