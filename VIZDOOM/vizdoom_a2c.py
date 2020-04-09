from absl import app, flags
import itertools
from random import sample, randint, random
import numpy as np
from skimage.transform import resize
from time import time, sleep
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from viz_utils import v_wrap, set_init, plotter_ep_rew, handleArguments, optimize, plotter_ep_time, confidence_intervall
from shared_adam import SharedAdam
import gym

env = gym.make('CartPole-v0').unwrapped
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


GAMMA = 0.9
MAX_EP = 30
FLAGS = flags.FLAGS
frame_repeat = 12
resolution = (30, 45)
default_config_file_path = "/home/juice100/pytorch-A3C/VIZDOOM/deadly_corridor.cfg"


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


class QNet(nn.Module):
    def __init__(self, a_dim):
        super(QNet, self).__init__()
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
                    new_logits[countrow][i] =  logits[countrow][i]
                    countrow += 1
        probs = F.softmax(new_logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


    def train_step(self, s1, target_q):
        output = self(s1)
        loss = self.criterion(output, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def learn_from_memory(self):
        if self.memory.size < FLAGS.batch_size: return
        s1, a, s2, isterminal, r = self.memory.get_sample(FLAGS.batch_size)
        q = self(s2).detach()
        q2, _ = torch.max(q, dim=1)
        target_q = self(s1).detach()
        idxs = (torch.arange(target_q.shape[0]), a)
        target_q[idxs] = r + FLAGS.discount * (1-isterminal) * q2
        self.train_step(s1, target_q)


def main(_):

    game = initialize_vizdoom(FLAGS.config)
    statesize = (game_state(game).shape[0])
    state=game_state(game)
    print("Current State:" , state, "\n")
    print("Statesize:" , statesize, "\n")
    n = game.get_available_buttons_size()
    print ("Action Size: ", n)
    actions = [list(a) for a in itertools.product([0, 1], repeat=n)]
    print("All possible Actions:", actions, "\n", "Total: ", len(actions))


    model = QNet(len(actions))
    opt = SharedAdam(model.parameters(), lr=0.005, betas=(0.92, 0.999))  # global optimizer

    global_ep, global_ep_r = 1, 0.
    name = 'w00'
    total_step = 1

    while global_ep < MAX_EP:
        game.new_episode()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0.
        while True:

            done = False
            a = model.choose_action(state)
            actions.append(a)
            #s_, r, done, _ = env.step(a)

            r = game.make_action(actions[a], frame_repeat)
            #print("Reward:", r)

            if game.is_episode_finished():
                done = True
            else:
                s_ = game_state(game)

            #if done: r = -1
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

if __name__ == '__main__':
    flags.DEFINE_integer('batch_size', 64, 'Batch size')
    flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate')
    flags.DEFINE_float('discount', 0.99, 'Discount factor')
    flags.DEFINE_integer('replay_memory', 10000, 'Replay memory capacity')
    flags.DEFINE_integer('epochs', 20, 'Number of epochs')
    flags.DEFINE_integer('iters', 2000, 'Iterations per epoch')
    flags.DEFINE_integer('watch_episodes', 10, 'Trained episodes to watch')
    flags.DEFINE_integer('test_episodes', 100, 'Episodes to test with')
    flags.DEFINE_string('config', default_config_file_path,
                        'Path to the config file')
    flags.DEFINE_boolean('skip_training', False, 'Set to skip training')
    flags.DEFINE_boolean('load_model', False, 'Load the model from disk')
    flags.DEFINE_string('save_path', 'model-doom.pth',
                        'Path to save/load the model')
    app.run(main)


