"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

"""

import torch
import torch.nn as nn
from utils import v_wrap, set_init, plotter_ep_rew, handleArguments, optimize, plotter_ep_time, confidence_intervall
import matplotlib.pyplot as plt
import torch.nn.functional as F
from shared_adam import SharedAdam
import numpy as np
import gym
from datetime import datetime
import time
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"

GAMMA = 0.9
MAX_EP = 3000

env = gym.make('CartPole-v0').unwrapped
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


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
    # load global network
    print ("Starting A2C Agent for Cartpole-v0")
    time.sleep(3)
    timedelta_sum = datetime.now()
    timedelta_sum -= timedelta_sum
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    actions = []
    # Reinitialize Agent to make 5 trials
    for i in range(3):
        starttime = datetime.now()

        if handleArguments().load_model:
            gnet = Net(N_S, N_A)
            gnet = torch.load("./save_model/a3c_cart.pt")
            gnet.eval()
        else:
            gnet = Net(N_S, N_A)

        opt = SharedAdam(gnet.parameters(), lr=0.005, betas=(0.92, 0.999))      # global optimizer

        # Global variables for episodes
        durations = []
        scores = []
        global_ep, global_ep_r = 1, 0.
        name = 'w00'
        total_step = 1
        stop_processes = False

        while global_ep < MAX_EP and stop_processes is False:
            s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                # Initialize stopwatch for average episode duration
                start = time.time()
                if name == 'w00' and handleArguments().demo_mode:
                    env.render()
                a = gnet.choose_action(v_wrap(s[None, :]))
                actions.append(a)
                s_, r, done, _ = env.step(a)

                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if done or ep_r == 600:  # update network
                    # sync
                    optimize(opt, gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    global_ep += 1

                    if global_ep_r == 0.:
                        global_ep_r = ep_r
                    else:
                        global_ep_r = global_ep_r * 0.99 + ep_r * 0.01

                    end = time.time()
                    duration = end - start
                    durations.append(duration)

                    print("w00 Ep:", global_ep, "| Ep_r: %.0f" % global_ep_r, "| Duration:", round(duration, 5))
                    scores.append(int(global_ep_r))
                    if handleArguments().load_model:
                        if np.mean(scores[-min(100, len(scores)):]) >= 400 and global_ep >= 100:
                            stop_processes = True
                    else:
                        if np.mean(scores[-min(10, len(scores)):]) >= 400 and global_ep >= 10:
                            stop_processes = True
                    break

                s = s_
                total_step += 1

        if np.mean(scores[-min(10, len(scores)):]) >= 300 and not handleArguments().load_model:
            print("Save model")
            torch.save(gnet, "./save_model/a2c_cart.pt")
        elif handleArguments().load_model:
            print ("Testing! No need to save model.")
        else:
            print("Failed to train agent. Model was not saved")
        endtime = datetime.now()
        timedelta = endtime - starttime
        print("Number of Episodes: ", global_ep, " | Finished within: ", timedelta)

        timedelta_sum += timedelta/3

        # Get results for confidence intervall
        if handleArguments().load_model:
            confidence_intervall(actions, True)
        else:
            confidence_intervall(actions)

        # Plot results
        plotter_ep_time(ax1, durations)
        plotter_ep_rew(ax2, scores)
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 8
            }
    plt.text(0, 650, f"Average Training Duration: {timedelta_sum}", fontdict=font)
    plt.title("Vanilla A2C-Cartpole", fontsize = 16)
    plt.show()

    sys.exit()
