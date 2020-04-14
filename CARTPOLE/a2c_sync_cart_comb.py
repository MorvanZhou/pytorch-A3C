"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

"""

import torch
import torch.nn as nn
from cart_utils import v_wrap, set_init, plotter_ep_rew, plotter_ep_rew_norm, handleArguments, push_and_pull, record, plotter_ep_time_norm, plotter_ep_time, confidence_intervall
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import numpy as np
import gym
import time
from datetime import datetime
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"

GAMMA = 0.9
MAX_EP = 2000

env = gym.make('CartPole-v0').unwrapped
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 80)
        self.pi2 = nn.Linear(80, 60)
        self.pi3 = nn.Linear(60, a_dim)
        self.v2 = nn.Linear(80, 60)
        self.v3 = nn.Linear(60, 1)
        set_init([self.pi1, self.pi2, self.pi3, self.v2, self.v3])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        pi2 = F.relu(self.pi2(pi1))
        logits = self.pi3(pi2)
        v2 = F.relu(self.v2(pi1))
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


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, global_time_done, res_queue, time_queue, action_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.g_time = global_ep, global_ep_r, global_time_done
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.res_queue, self.time_queue, self.action_queue = res_queue, time_queue, action_queue
        self.env = gym.make("CartPole-v0").unwrapped

    def run(self):
        total_step = 1
        stop_processes = False
        scores = []
        while self.g_ep.value < MAX_EP and stop_processes is False:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                start = time.time()
                if self.name == 'w00' and handleArguments().demo_mode:
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if done or ep_r >= 450:  # update global and assign to local net
                    # sync
                    end = time.time()
                    time_done = end - start

                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA, True,
                                  self.g_ep)

                    if not handleArguments().load_model:
                        time.sleep(0.5)

                    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.time_queue, self.g_time, time_done, a,
                           self.action_queue, self.name)

                    scores.append(int(self.g_ep_r.value))
                    if handleArguments().load_model and handleArguments().normalized_plot:
                        if np.mean(scores[-min(100, len(scores)):]) >= 400 and self.g_ep.value >= 100:
                            stop_processes = True
                    elif handleArguments().normalized_plot:
                        if np.mean(scores[-min(10, len(scores)):]) >= 400 and self.g_ep.value >= mp.cpu_count():
                            stop_processes = True
                    else:
                        stop_processes = False
                    break

                s = s_
                total_step += 1
        self.res_queue.put(None)
        self.time_queue.put(None)
        self.action_queue.put(None)


if __name__ == "__main__":
    # load global network
    print("Starting Synchronous A2C Agent for Cartpole-v0")
    time.sleep(3)
    timedelta_sum = datetime.now()
    timedelta_sum -= timedelta_sum
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if handleArguments().normalized_plot and not handleArguments().save_data:
        runs = 3
    else:
        runs = 1

    # Reinitialize Agent to make 5 trials
    for i in range(runs):
        starttime = datetime.now()
        if handleArguments().load_model:
            gnet = Net(N_S, N_A)
            gnet = torch.load("./CARTPOLE/cart_save_model/a2c_sync_cart_comb.pt")
            gnet.eval()
        else:
            gnet = Net(N_S, N_A)

        gnet.share_memory()  # share the global parameters in multiprocessing
        opt = SharedAdam(gnet.parameters(), lr=0.001, betas=(0.92, 0.999))  # global optimizer
        global_ep, global_ep_r, global_time_done = mp.Value('i', 0), mp.Value('d', 0.), mp.Value('d', 0.)
        res_queue, time_queue, action_queue = mp.Queue(), mp.Queue(), mp.Queue()
        # parallel training
        if handleArguments().load_model:
            workers = [
                Worker(gnet, opt, global_ep, global_ep_r, global_time_done, res_queue, time_queue, action_queue, i) for
                i in range(1)]
            [w.start() for w in workers]
        else:
            workers = [
                Worker(gnet, opt, global_ep, global_ep_r, global_time_done, res_queue, time_queue, action_queue, i) for
                i in range(mp.cpu_count())]
            [w.start() for w in workers]

        # record episode-reward and episode-duration to plot
        res = []
        durations = []
        actions = []
        while True:
            r = res_queue.get()
            t = time_queue.get()
            a = action_queue.get()
            if r is not None:
                res.append(r)
            if t is not None:
                durations.append(t)
            if a is not None:
                actions.append(a)
            else:
                break

        [w.join() for w in workers]

        if np.mean(res[-min(mp.cpu_count(), len(res)):]) >= 200 and not handleArguments().load_model:
            print("Save model")
            torch.save(gnet, "./CARTPOLE/cart_save_model/a2c_sync_cart_comb.pt")
        elif handleArguments().load_model:
            print("Testing! No need to save model.")
        else:
            print("Failed to train agent. Model was not saved")
        endtime = datetime.now()
        timedelta = endtime - starttime
        print("Number of Episodes: ", global_ep.value, " | Finished within: ", timedelta)

        timedelta_sum += timedelta / 3

        # Get results for confidence intervall

        if handleArguments().load_model:
            confidence_intervall(actions, True)
        else:
            confidence_intervall(actions)

        # Plot results
        if handleArguments().normalized_plot:
            plotter_ep_time_norm(ax1, durations)
            plotter_ep_rew_norm(ax2, res)
        else:
            plotter_ep_time(ax1, durations)
            plotter_ep_rew(ax2, res)

        if handleArguments().save_data:
            if handleArguments().load_model:
                scores = np.asarray([res])
                np.savetxt('CARTPOLE/cart_save_plot_data/a2c_sync_cart_comb_test.csv', scores, delimiter=',')
            else:
                scores = np.asarray([res])
                np.savetxt('CARTPOLE/cart_save_plot_data/a2c_sync_cart_comb.csv', scores, delimiter=',')

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 8,
            }
    plt.text(0, 450, f"Average Duration: {timedelta_sum}", fontdict=font)
    plt.title("Synchronous A2C-Cartpole (shared NN)", fontsize=16)
    plt.show()

    sys.exit()

