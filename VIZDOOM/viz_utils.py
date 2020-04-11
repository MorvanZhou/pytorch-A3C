"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]


    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))


    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def optimize(opt, lnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []

    for r in br[::-1]:    # reverse buffer r
        # Advantage function?
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))
    # calculate local gradients
    opt.zero_grad()
    loss.backward()
    opt.step()


def record(global_ep, global_ep_r, ep_r, res_queue, time_queue, time_done, a, action_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    time_queue.put(time_done)
    action_queue.put(a)
    print(name, "Ep:", global_ep.value, "| Ep_r: %.0f" % global_ep_r.value, "| Duration:", round(time_done, 5))


def plotter_ep_rew(ax2, scores):
    ax2.plot(scores)
    #ax2.axhline(y=200.00, color='r')
    ax2.set_ylim(-150,300)
    ax2.set_ylabel('Reward per Episode')
    ax2.set_xlabel('Episode')


def plotter_ep_time(ax1, duration_episode):
    ax1.plot(duration_episode)
    ax1.set_ylim(0,200)
    ax1.set_ylabel('Duration of Episode')


# TODO: check for reasonable count length and adjust
def confidence_intervall(actions, load_model = False):
    count = 1
    probab_count = 0
    probabilities = []
    for a in actions:
        if a == 1:
            probab_count += 1
        if count % 100 == 0 and count != 1:
            probab_count = probab_count / 100
            probabilities.append(probab_count)
            probab_count = 0
        count += 1

    # Check for probabilities of actions and create confidence intervall for two standard deviations
    print("Probabilities: ", probabilities)

    if load_model == True:
        stan_dev1 = np.sqrt(probabilities[0] * (1 - probabilities[0]) / 100) * 2
        print("First Confidence Intervall for 95% confidence: the action 'right' is chosen between",
              round(probabilities[0] - stan_dev1, 3), " and", round(probabilities[0] + stan_dev1, 3))

    else:
        stan_dev1 = np.sqrt(probabilities[1] * (1-probabilities[1]) / 100)*2
        print ("First Confidence Intervall for 95% confidence: the action 'right' is chosen between", round(probabilities[1]-stan_dev1, 3), " and", round(probabilities[1] + stan_dev1, 3))

        stan_dev2 = np.sqrt(probabilities[2] * (1 - probabilities[2]) /100) * 2
        print("Second Confidence Intervall for 95% confidence: the action 'right' is chosen between", round(probabilities[2] - stan_dev2, 3), " and",
              round(probabilities[2] + stan_dev2, 3))

        stan_dev3 = np.sqrt(probabilities[3] * (1 - probabilities[3]) /100) * 2
        print("Second Confidence Intervall for 95% confidence: the action 'right' is chosen between", round(probabilities[3] - stan_dev3, 3), " and",
              round(probabilities[3] + stan_dev3, 3))


def handleArguments():
    """Handles CLI arguments and saves them globally"""
    parser = argparse.ArgumentParser(
        description="Switch between modes in A2C or loading models from previous games")
    parser.add_argument("--demo_mode", "-d", help="Renders the gym environment", action="store_true")
    parser.add_argument("--load_model", "-l", help="Loads the model of previously gained training data", action="store_true")
    global args
    args = parser.parse_args()
    return args


