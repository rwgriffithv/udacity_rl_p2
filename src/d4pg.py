# Distributed Distributional Deep Deterministic Policy Gradient (D4PG) (G. Barth-Maron et al.)

from random import sample
from numpy.lib.type_check import _nan_to_num_dispatcher
import torch
import torch.nn as tnn
import torch.optim as topt
import torch.cuda as tcuda
import numpy as np

# local imports
from .nn import polyak_update
from .nn import categorical_projection


class D4PG:
    def __init__(self, policynet, distqnet, target_policynet, target_distqnet, replay_buf, v_min, v_max, num_atoms, policynet_lr=0.0003, distqnet_lr=0.0001, discount_factor=0.99, polyak_factor=0.99):
        self.policynet = policynet
        self.distqnet = distqnet # distributional Q-network (/critic/value-function)
        self.target_policynet = target_policynet
        self.target_distqnet = target_distqnet
        self.replay_buf = replay_buf
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.raw_discount_factor = discount_factor # used in numpy computations
        # set target_qnet to qnet's weights
        polyak_update(self.distqnet, self.target_distqnet, 0)
        # initialize optimizers
        self.policy_optimizer = topt.Adam(policynet.parameters(), policynet_lr)
        self.distq_optimizer = topt.Adam(distqnet.parameters(), distqnet_lr)
        # get devices
        self.dev_gpu = torch.device("cuda" if tcuda.is_available() else "cpu")
        self.dev_cpu = torch.device("cpu")
        # create constant tensors
        self.discount_factor = torch.tensor(discount_factor, dtype=torch.float32, device=self.dev_gpu)
        self.polyak_factor = torch.tensor(polyak_factor, dtype=torch.float32, device=self.dev_gpu)
        # distq output atoms of values
        value_delta = (v_max - v_min) / num_atoms
        self.distq_values = torch.from_numpy(np.array([v_min + i * value_delta for i in range(num_atoms)])).float().to(self.dev_gpu)
        # for random action noise
        self.np_rng = np.random.default_rng()

    def optimize(self, num_steps, num_samples, sample_size):
        for _ in range(num_steps): # number of gradient steps
            # get sample batch, convert numpy arrays to tensors and send to GPU
            batch_tuple = self.replay_buf.sample(num_samples, sample_size)
            states, actions, rewards, terminals, priorities = [torch.from_numpy(b).float().to(self.dev_gpu) for b in batch_tuple]
            # torch.clamp(rewards, min=-1.0, max=1.0)
            
            # calculate distributional q-network loss
            self.distqnet.train(True) # ensure qnet is training so back propogation can occur
            reward_discount = np.array([self.raw_discount_factor**i for i in range(sample_size)])
            np_rewards = batch_tuple[2]
            discounted_rewards = torch.from_numpy([np.sum(np_rewards[i * sample_size : (i + 1) * sample_size] * reward_discount) for i in range(num_samples)]).float().to(self.dev_gpu)
            end_terminals = terminals[[(i + 1) * sample_size - 1 for i in range(num_samples)]]
            # calculate target values for each sample of transitions using target values from Q distribution summed with discounted rewards
            end_states = states[[i * (sample_size + 1) + sample_size for i in range(num_samples)]] # extract final end states from each sample of consecutive transitions
            targ_dist_probs = torch.softmax(self.target_distqnet(torch.cat((end_states, self.target_policynet(end_states)), -1)))
            targ_dist_values = discounted_rewards + self.discount_factor.pow(sample_size) * end_terminals * (targ_dist_probs * self.distq_values)
            # apply categorical projection h_zi(z) from Appendix B to target distrbution values and probabilities to get projected target distribution probabilities
            np_targ_dist_probs = targ_dist_probs.to(self.dev_cpu).numpy()
            np_targ_dist_values = targ_dist_values.to(self.dev_cpu).numpy()
            np_proj_targ_dist_probs = categorical_projection(self.v_min, self.v_max, self.num_atoms, np_targ_dist_probs, np_targ_dist_values)    
            proj_targ_dist_probs = torch.from_numpy(np_proj_targ_dist_probs).float().to(self.dev_gpu)
            # the loss is calculated as cross-entropy, according to Appendix A
            start_states = states[[i * (sample_size + 1) for i in range(num_samples)]]  # extract first state frome each sample of consecutive transitions
            start_actions = actions[[i * sample_size for i in range(num_samples)]] # extract first action
            # TODO: may have to take softmax of proj_targ_dist_probs, may want to use BCELoss
            # distq_loss = tnn.BCELoss()(proj_targ_dist_probs.detach(), torch.softmax(self.distqnet(torch.cat((start_states, start_actions), -1))), reduction="sum")
            distq_loss = torch.sum(proj_targ_dist_probs.detach() * torch.softmax(self.distqnet(torch.cat((start_states, start_actions), -1))), axis=-1)
            # apply sample priorities to loss and reduce
            distq_loss = torch.sum(distq_loss * self.replay_buf.num_tran / priorities) / num_samples
            
            # calculate policy-network loss, simple average expected value from policy-output-actions at each starting state
            self.policynet.train(True)
            policy_loss = -torch.sum(self.distqnet(torch.cat((start_states, self.policynet(start_states)), -1) * self.distq_values)) / num_samples

            # update distributional q-network
            self.distq_optimizer.zero_grad() # zero/clear previous gradients
            distq_loss.backward()
            self.distq_optimizer.step()
            
            # update policy-network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # polyak update target networks
            polyak_update(self.qnet, self.target_qnet, self.polyak_factor)
            polyak_update

    def get_action(self, state, epsilon=0.0):
        policy_in = torch.from_numpy(np.array(state)).float().to(self.dev_gpu) # convert state, state can be numpy array or list
        self.policynet.train(False)
        with torch.no_grad():
            action = torch.tanh(self.policynet(policy_in)).to(self.dev_cpu).numpy()
        # epsilon noise
        return action + epsilon * self.np_rng.random()