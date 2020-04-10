
import itertools
import numpy as np
from skimage.transform import resize
import time
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from viz_utils import set_init, plotter_ep_rew, handleArguments, optimize, plotter_ep_time, confidence_intervall
from shared_adam import SharedAdam
import os
os.environ["OMP_NUM_THREADS"] = "1"


GAMMA = 0.9
MAX_EP = 30
frame_repeat = 12
resolution = (30, 45)
config_file_path = "deadly_corridor.cfg"


def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    return game

def preprocess(img):
    return torch.from_numpy(resize(img, resolution).astype(np.float32))


def game_state(game):
    return preprocess(game.get_state().screen_buffer)


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

        #self.optimizer = torch.optim.SGD(self.parameters(), FLAGS.learning_rate)

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
                    new_logits[countrow][i] = logits[countrow][i]
                    countrow += 1
        probs = F.softmax(new_logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss



if __name__ == '__main__':

    game = initialize_vizdoom(config_file_path)
    statesize = (game_state(game).shape[0])
    state=game_state(game)
    print("Current State:" , state, "\n")
    print("Statesize:" , statesize, "\n")
    n = game.get_available_buttons_size()
    print ("Action Size: ", n)
    actions = [list(a) for a in itertools.product([0, 1], repeat=n)]
    print("All possible Actions:", actions, "\n", "Total: ", len(actions))


    model = Net(len(actions))
    opt = SharedAdam(model.parameters(), lr=0.005, betas=(0.92, 0.999))  # global optimizer

    global_ep, global_ep_r = 1, 0.
    total_step = 1

    while global_ep < MAX_EP:
        game.new_episode()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0.
        while True:

            done = False
            a = model.choose_action(state)
            actions.append(a)

            r = game.make_action(actions[a], frame_repeat)

            if game.is_episode_finished():
                done = True
            else:
                s_ = game_state(game)

            ep_r += r
            buffer_a.append(a)
            buffer_s.append(state)
            buffer_r.append(r)

            if done or ep_r == 600:  # update network
                # sync
                optimize(opt, model, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                game.get_total_reward()
                buffer_s, buffer_a, buffer_r = [], [], []

                global_ep += 1

                if global_ep_r == 0.:
                    global_ep_r = ep_r
                else:
                    global_ep_r = global_ep_r * 0.99 + ep_r * 0.01

                print("w00 Ep:", global_ep, "| Ep_r: %.0f" % global_ep_r)
                break

            s = s_
            total_step += 1

    game.close()


