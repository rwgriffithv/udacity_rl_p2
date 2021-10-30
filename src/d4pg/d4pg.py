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
    def __init__(self, policynet, distqnet, target_policynet, target_distqnet, replay_buf, sample_size, v_min, v_max, num_atoms, policynet_lr=0.0003, distqnet_lr=0.0001, discount_factor=0.99, polyak_factor=0.99):
        self.policynet = policynet
        self.distqnet = distqnet # distributional Q-network (/critic/value-function)
        self.target_policynet = target_policynet
        self.target_distqnet = target_distqnet
        self.replay_buf = replay_buf
        self.sample_size = sample_size # number of consecutive transitions per batch sample element
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.raw_discount_factor = discount_factor # used in numpy computations
        # initialize target network weights
        polyak_update(self.policynet, self.target_policynet, 0)
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
        value_delta = (v_max - v_min) / (num_atoms - 1)
        self.distq_values = torch.from_numpy(np.array([v_min + i * value_delta for i in range(num_atoms)])).float().to(self.dev_gpu)
        # for random action noise
        self.np_rng = np.random.default_rng()

    def optimize(self, num_steps, num_samples):
        for _ in range(num_steps): # number of gradient steps
            # get sample batch, convert numpy arrays to tensors and send to GPU
            batch_tuple = self.replay_buf.sample(num_samples)
            if batch_tuple is None:
                # print("Skipping optimization due to shallow replay buffer...")
                return
            np_states, np_actions, np_rewards, np_terminals, np_priorities = batch_tuple
            # TODO: may want to clamp rewards to -1, 1
            # torch.clamp(rewards, min=-1.0, max=1.0)
            
            # calculate distributional q-network loss
            self.distqnet.train(True) # ensure qnet is training so back propogation can occur
            reward_discount = np.array([self.raw_discount_factor**i for i in range(self.sample_size)])
            discounted_rewards = torch.from_numpy(np.array([np.sum(np_rewards[i * self.sample_size : (i + 1) * self.sample_size] * reward_discount) for i in range(num_samples)])).float().to(self.dev_gpu).unsqueeze(1)
            np_end_terminal_check = 1 - np_terminals[[(i + 1) * self.sample_size - 1 for i in range(num_samples)]]
            end_terminal_check = torch.from_numpy(np_end_terminal_check).float().to(self.dev_gpu)
            # calculate target values for each sample of transitions using target values from Q distribution summed with discounted rewards
            np_end_states = np_states[[i * (self.sample_size + 1) + self.sample_size for i in range(num_samples)]] # extract final end states from each sample of consecutive transitions
            end_states = torch.from_numpy(np_end_states).float().to(self.dev_gpu)
            targ_dist_probs = tnn.functional.softmax(self.target_distqnet(torch.cat((end_states, self.target_policynet(end_states)), -1)), -1)
            # print("targ_dist_probs nan: ", torch.isnan(targ_dist_probs).any())
            sample_size = torch.tensor(self.sample_size).float().to(self.dev_gpu)
            targ_dist_values = discounted_rewards + self.discount_factor.pow(sample_size) * end_terminal_check * self.distq_values # to get expected value, multiply by targ_dist_probs and reduce sum
            # print("targ_dist_values nan: ", torch.isnan(targ_dist_values).any())
            # apply categorical projection h_zi(z) from Appendix B to target distrbution values and probabilities to get projected target distribution probabilities
            # TODO: may want to clamp targ_dist_values to -1, 1
            np_targ_dist_probs = targ_dist_probs.detach().to(self.dev_cpu).numpy()
            np_targ_dist_values = targ_dist_values.detach().to(self.dev_cpu).numpy()
            np_proj_targ_dist_probs = categorical_projection(self.v_min, self.v_max, self.num_atoms, np_targ_dist_probs, np_targ_dist_values)
            # print("np_proj_targ_dist_probs: ", np_proj_targ_dist_probs)
            # print("np_proj_targ_dist_probs sums: ", np.sum(np_proj_targ_dist_probs, axis=-1))
            proj_targ_dist_probs = torch.from_numpy(np_proj_targ_dist_probs).float().to(self.dev_gpu)
            # print("proj_targ_dist_probs nan: ", torch.isnan(proj_targ_dist_probs).any())
            # the loss is calculated as cross-entropy, according to Appendix A
            np_start_states = np_states[[i * (self.sample_size + 1) for i in range(num_samples)]]  # extract first state frome each sample of consecutive transitions
            np_start_actions = np_actions[[i * self.sample_size for i in range(num_samples)]] # extract first action
            start_states = torch.from_numpy(np_start_states).float().to(self.dev_gpu)
            start_actions = torch.from_numpy(np_start_actions).float().to(self.dev_gpu)
            distq_loss = tnn.BCELoss(reduction='none')(tnn.functional.softmax(self.distqnet(torch.cat((start_states, start_actions), -1)), -1), proj_targ_dist_probs.detach())
            # priorities = torch.from_numpy(np_priorities).float().to(self.dev_gpu)
            distq_loss = torch.sum(distq_loss, -1)
            distq_loss = torch.mean(distq_loss) # * priorities)
            # distq_loss = torch.sum(proj_targ_dist_probs.detach() * tnn.functional.softmax(self.distqnet(torch.cat((start_states, start_actions), -1)), -1), axis=-1)
            # print("distq_loss_1 nan: ", torch.isnan(distq_loss).any())
            # apply sample priorities to loss and reduce
            # distq_loss = -torch.sum(distq_loss / (self.replay_buf.num_tran * priorities)) / num_samples
            # print("distq_loss_2 nan: ", torch.isnan(distq_loss).any())

            # update distributional q-network
            self.distq_optimizer.zero_grad() # zero/clear previous gradients
            distq_loss.backward()
            self.distq_optimizer.step()
            
            # calculate policy-network loss, simple average expected value from policy-output-actions at each starting state
            self.policynet.train(True)
            policy_out = self.policynet(start_states)
            # print("policy_out nan: ", torch.isnan(policy_out).any())
            distq_out = self.distqnet(torch.cat((start_states, policy_out), -1))
            policy_loss = -torch.mean(torch.sum(distq_out * self.distq_values, -1)) # maximize expected q-value
            # print("policy_loss nan: ", torch.isnan(policy_loss).any())
            
            # update policy-network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
        # polyak update target networks
        polyak_update(self.distqnet, self.target_distqnet, self.polyak_factor)
        polyak_update(self.policynet, self.target_policynet, self.polyak_factor)

    def get_actions(self, states, epsilon=0.0):
        policy_in = torch.from_numpy(np.array(states)).float().to(self.dev_gpu) # convert state, state can be numpy array or list
        self.policynet.train(False)
        with torch.no_grad():
            actions = torch.tanh(self.policynet(policy_in)).to(self.dev_cpu).numpy()
        # epsilon noise and clipping
        noise = epsilon * self.np_rng.random(actions.shape[1])
        return np.clip(actions + noise, -1, 1)