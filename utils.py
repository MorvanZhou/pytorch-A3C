from torch import nn
import torch
import numpy as np
from torch.autograd import Variable


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return Variable(torch.from_numpy(np_array))


def set_init(layers):
    for layer in layers:
        nn.init.normal(layer.weight, mean=0., std=0.1)
        nn.init.constant(layer.bias, 0.1)