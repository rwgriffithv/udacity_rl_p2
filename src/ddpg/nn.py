# neural network utility functions

import torch
import torch.nn as tnn
import torch.cuda as tcuda
import numpy as np


def build_actor_network(input_size, output_size):
    device = torch.device("cuda" if tcuda.is_available() else "cpu")
    net = tnn.Sequential(
        tnn.Linear(input_size, 128),
        tnn.ReLU(),
        tnn.Linear(128, 128),
        tnn.ReLU(),
        tnn.Linear(128, output_size)
    )
    return net.float().to(device)

def build_critic_network(input_size, output_size):
    device = torch.device("cuda" if tcuda.is_available() else "cpu")
    net = tnn.Sequential(
        tnn.Linear(input_size, 256),
        tnn.ReLU(),
        tnn.Linear(256, 256),
        tnn.ReLU(),
        tnn.Linear(256, output_size)
    )
    return net.float().to(device)

def polyak_update(net, target_net, polyak_factor):
    for param, t_param in zip(net.parameters(), target_net.parameters()):
        t_param.data.copy_(polyak_factor * t_param.data + (1 - polyak_factor) * param.data)