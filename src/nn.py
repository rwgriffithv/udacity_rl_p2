# neural network utility functions

import torch
import torch.nn as tnn
import torch.cuda as tcuda
import numpy as np


def build_network(input_size, output_size):
    device = torch.device("cuda" if tcuda.is_available() else "cpu")
    net = tnn.Sequential(
        tnn.Linear(input_size, 128),
        tnn.ReLU(),
        tnn.Linear(128, 128),
        tnn.ReLU(),
        tnn.Linear(128, output_size)
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
    vals = atom_values.reshape(atom_rows_shape)
    proj_probs = np.zeros(atom_rows_shape)
    # project each distribution (each row is a distribution of atoms)
    for p, z, proj_p in zip(probs, vals, proj_probs):
        for i in range(num_atoms):
            zi = z[i]
            pzi = None if i == 0 else z[i-1]
            nzi = None if i == num_atoms - 1 else z[i+1]
            proj_zi = np.zeros(z.shape)
            for j in range(num_atoms):
                zj = z[j]
                if (i == 0 and zj <= v_min) or (i == num_atoms - 1 and zj >= v_max):
                    proj_zi[j] = 1
                elif pzi and pzi <= zj and zj <= zi:
                    proj_zi[j] = (zj - pzi) / (zi - pzi)
                elif nzi and zi <= zj and zj <= nzi:
                    proj_zi[j] = (nzi - zj) / (nzi - zi)
            proj_p[i] = np.sum(proj_zi * p)
    return proj_probs