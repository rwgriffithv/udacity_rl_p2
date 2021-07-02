# neural network utility functions

import torch
import torch.nn as tnn
import torch.cuda as tcuda
import numpy as np


def build_network(input_size, output_size):
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

def categorical_projection(v_min, v_max, num_atoms, atom_probs, atom_values):
    if atom_probs.size != atom_values.size or atom_probs.size % num_atoms * atom_values.size % num_atoms != 0:
        return None
    atom_rows_shape = (atom_probs.size // num_atoms, num_atoms)
    probs = atom_probs.reshape(atom_rows_shape)
    vals = np.maximum(v_min, np.minimum(v_max, atom_values.reshape(atom_rows_shape)))
    # project each distribution (each row is a distribution of atoms)
    delta_v = (v_max - v_min) / (num_atoms - 1)
    proj_probs = np.zeros(atom_rows_shape)
    for i in range(num_atoms):
        zi = v_min + i * delta_v
        diff = np.abs(vals - zi)
        mask = diff <= delta_v
        proj_probs[:,i] = np.sum(probs[mask] * (delta_v - diff[mask]) / delta_v, axis=-1)
    # softmax probs
    for i, p in enumerate(proj_probs):
        p = np.exp(p)
        proj_probs[i] = p / np.sum(p)
    return proj_probs