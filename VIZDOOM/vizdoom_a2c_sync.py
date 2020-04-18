import itertools
import numpy as np
from skimage.transform import resize
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from viz_utils import v_wrap, set_init, plotter_ep_rew, plotter_ep_rew_norm, handleArguments, push_and_pull, optimize, record, plotter_ep_time_norm, plotter_ep_time, confidence_intervall
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 2000
frame_repeat = 12
resolution = (30, 45)
config_file_path = "VIZDOOM/deadly_corridor.cfg"
worker_num = int(mp.cpu_count())     # Number of reincarnations for Agent


def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    return game

def preprocess(img):
    return torch.from_numpy(resize(img, resolution).astype(np.float32))


def game_state(game):
    return preprocess(game.get_state().screen_buffer)

game = initialize_vizdoom(config_file_path)
statesize = (game_state(game).shape[0])
state = game_state(game)
n = game.get_available_buttons_size()
actions = [list(a) for a in itertools.product([0, 1], repeat=n)]

print("Current State:", state, "\n")
print("Statesize:", statesize, "\n")

print("Action Size: ", n)
print("All possible Actions:", actions, "\n", "Total: ", len(actions))
print("Number of used CPUs: ", worker_num)

class Net(nn.Module):
    def __init__(self, a_dim):
        super(Net, self).__init__()
        self.s_dim = 45
        self.a_dim = a_dim
        self.pi1 = nn.Linear(self.s_dim, 120)
        self.pi2 = nn.Linear(120, 360)
        self.pi3 = nn.Linear(360, a_dim)
        self.v1 = nn.Linear(self.s_dim, 120)
        self.v2 = nn.Linear(120, 360)
        self.v3 = nn.Linear(360, 1)

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

        new_values = torch.zeros([len(v_t), 1], dtype=torch.float32)
        # Reshape Tensor of values
        for i in range(len(v_t)):
            for j in range(30):
                values[i][0] += values[i+j][0]
            new_values[i][0] =  values[i][0]

        td = v_t - new_values
        c_loss = td.pow(2)
        new_logits = torch.zeros([len(v_t), 128], dtype=torch.float32)
        # Reshape Tensor of logits
        for i in range(len(logits[0])):
            countrow = 0
            for j in range(len(logits)):
                logits[countrow][i] += logits[j][i]
                if j % 30 == 0:
                    new_logits[countrow][i] =  logits[countrow][i]
                    countrow += 1
        probs = F.softmax(new_logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss



class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, inner_ep, global_ep_r, global_time_done, res_queue, time_queue, action_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.g_time = global_ep, global_ep_r, global_time_done
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(len(actions))           # local network
        self.game = initialize_vizdoom(config_file_path)
        self.res_queue, self.time_queue, self.action_queue = res_queue, time_queue, action_queue
        self.inner_ep = inner_ep
        #self.sleep_max = sleep_max

    def run(self):
        total_step = 1
        stop_processes = False
        scores = []
        sleep = 15
        superdone = 0
        cpu_count = []
        while self.inner_ep.value < (mp.cpu_count()*3) and stop_processes is False:
            self.game.new_episode()
            state = game_state(self.game)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.

            #print("Outside:", sleep)
            print("initialized:", self.name)
            while True:
                start = time.time()
                done = False
                a = self.lnet.choose_action(state)

                r = self.game.make_action(actions[a], frame_repeat)

                if self.game.is_episode_finished():
                    done = True
                else:
                    s_ = game_state(self.game)

                ep_r += r
                buffer_a.append(a)
                buffer_s.append(state)
                buffer_r.append(r)

                if done or total_step % UPDATE_GLOBAL_ITER == 0:  # update network
                    # sync
                    optimize(opt, self.lnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)

                    #game.get_total_reward()
                    #buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        #print("done with:", self.name)
                        self.inner_ep.value += 1
                        print ("Inner Ep:", self.inner_ep.value)
                        end = time.time()
                        time_done = end - start
                        #print ("Duration:", time_done)
                        #print("Whats the time?", sleep)
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.time_queue, self.g_time, time_done, a,
                               self.action_queue, self.name)
                        #print("hore...")
                        #push_and_pull(opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA, True, self.g_ep)
                        #time.sleep(20)
                        #print(self.sync.get())
                            #print("Pull global model")
                            #self.lnet.load_state_dict(self.gnet.state_dict())
                        #print("here...")

                        for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
                             gp._grad = lp.grad


                        #if self.sync.get() == True:
                        #    print("Agent", self.name, "gets in sync")
                        #    self.agent_count.append(1)
                        #    while len(self.agent_count) != mp.cpu_count():
                        #        print(len(self.agent_count), self.agent_count)
                        #        time.sleep(15)
#
                        #    for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
                        #        gp._grad = lp.grad
#
                        #    print("loading state dict")
                        #    self.lnet.load_state_dict(self.gnet.state_dict())
                        #    if len(self.agent_count) >= 12:
                        #        self.agent_count = []
                        #    break


                        #if superdone % 10 == 0:
#
                        #    sleep = 120 - 10 * len(self.sync)
                        #    print(sleep)
                        #    print("Pull global model")
                        #    for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
                        #        gp._grad = lp.grad
                        #        opt.step()
                        #    time.sleep(sleep)
                        #    self.lnet.load_state_dict(self.gnet.state_dict())
                        #    if len(self.sync) == mp.cpu_count():
                        #        self.sync = []
                        #    break
                        scores.append(int(self.g_ep_r.value))

                        if handleArguments().load_model and handleArguments().normalized_plot:
                            if np.mean(scores[-min(100, len(scores)):]) >= 50 and self.g_ep.value >= 100:
                                stop_processes = True
                        elif handleArguments().normalized_plot:
                            if np.mean(scores[-min(mp.cpu_count(), len(scores)):]) >= 50 and self.g_ep.value >= mp.cpu_count():
                                stop_processes = True
                        else:
                            stop_processes = False
                        break

                state = s_
                total_step += 1

        self.time_queue.put(None)
        self.res_queue.put(None)
        self.action_queue.put(None)


if __name__ == '__main__':

    print ("Starting A2C-Sync Agent for Vizdoom-DeadlyCorridor")
    time.sleep(3)

    timedelta_sum = datetime.now()
    timedelta_sum -= timedelta_sum
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if handleArguments().normalized_plot and not handleArguments().save_data:
        runs = 3
    else:
        runs = 1

    for i in range(runs):
        starttime = datetime.now()

        # load global network
        if handleArguments().load_model:
            model = Net(len(actions))
            model = torch.load("./VIZDOOM/doom_save_model/a2c_sync_doom.pt")
            model.eval()
        else:
            model = Net(len(actions))

        # global optimizer
        opt = SharedAdam(model.parameters(), lr=0.001, betas=(0.92, 0.999))

        # record episode-reward and duration-episode to plot
        res = []
        durations = []
        action = []
        global_ep, global_ep_r, global_time_done = mp.Value('i', 0), mp.Value('d', 0.), mp.Value('d', 0.)
        res_queue, time_queue, action_queue = mp.Queue(), mp.Queue(), mp.Queue()
        loop = 0

        while global_ep.value < MAX_EP:
            loop += 1
            print ("loop: ", loop)
            # parallel training
            inner_ep = mp.Value('i', 0)
            manager = mp.Manager()
            sync, agent_count = mp.Queue(), manager.list()
            if handleArguments().load_model:
                workers = [Worker(model, opt, global_ep, global_ep_r, global_time_done,res_queue, time_queue, action_queue, i, sync) for i in range(1)]
                [w.start() for w in workers]
            else:
                workers = [Worker(model, opt, global_ep, inner_ep, global_ep_r, global_time_done, res_queue, time_queue, action_queue, i) for i in
                           range(worker_num)]
                [w.start() for w in workers]


            while True:
                #sync.put(False)
                #if global_ep.value % (mp.cpu_count()*3) == 0 and global_ep.value != 0:
                #    for w in workers:
                #        sync.put(True)
                r = res_queue.get()
                t = time_queue.get()
                a = action_queue.get()
                if r is not None:
                    res.append(r)
                if t is not None:
                    durations.append(t)
                if a is not None:
                    action.append(a)
                else:
                    break


            for w in workers:
                w.join()
                w.terminate()

            if np.mean(res[-min(10, len(res)):]) >= 0 and not handleArguments().load_model and global_ep.value >= 10:
                print("Save model")
                torch.save(model, "./VIZDOOM/doom_save_model/a2c_sync_doom.pt")
            elif handleArguments().load_model:
                print("Testing! No need to save model.")
            else:
                print("Failed to train agent. Model was not saved")

            if handleArguments().save_data:
                if handleArguments().load_model:
                    scores = np.asarray([res])
                    np.savetxt(f"VIZDOOM/doom_save_plot_data/a2c_sync_doom_test{loop}.csv", scores, delimiter=',')
                else:
                    scores = np.asarray([res])
                    np.savetxt(f"VIZDOOM/doom_save_plot_data/a2c_sync_doom{loop}.csv", scores, delimiter=',')

        endtime = datetime.now()
        timedelta = endtime - starttime
        print("Number of Episodes: ", global_ep.value, " | Finished within: ", timedelta)
        timedelta_sum += timedelta / 3

        # Get results for confidence intervall
        # if handleArguments().load_model:
        #   confidence_intervall(action, True)
        # else:
        #   confidence_intervall(action)

        # Plot results

        if handleArguments().normalized_plot:
            plotter_ep_time_norm(ax1, durations)
            plotter_ep_rew_norm(ax2, res)
        else:
            plotter_ep_time(ax1, durations)
            plotter_ep_rew(ax2, res)


    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 8
            }
    if handleArguments().normalized_plot:
        plt.text(0, 50, f"Average Training Duration: {timedelta_sum}", fontdict=font)
    else:
        plt.text(0, 400, f"Average Training Duration: {timedelta_sum}", fontdict=font)
    plt.title("Synchronous A2C-Vizdoom", fontsize=16)
    plt.show()

    game.close()
    sys.exit()

