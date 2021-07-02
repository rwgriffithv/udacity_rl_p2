# transition replay classes
import numpy as np
import random as rand


class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size):
        self.capacity = capacity
        self.size = 0
        self.states = np.zeros((capacity, state_size), np.dtype(float))
        # for discrete actions dtype is always int; using float for future continuous action use
        self.actions = np.zeros((capacity, action_size), np.dtype(float))
        self.rewards = np.zeros((capacity, 1), np.dtype(float))
        self.terminals = np.zeros((capacity, 1), np.dtype(int))
        self.next_states = np.zeros((capacity, state_size), np.dtype(float))

    def insert(self, states, actions, rewards, terminals, next_states):
        # append transitions if possible
        num_append = min(self.capacity - self.size, len(states))
        if num_append != 0:
            self.states[self.size : self.size + num_append] = states[:num_append]
            self.actions[self.size : self.size + num_append] = actions[:num_append]
            self.rewards[self.size : self.size + num_append] = rewards[:num_append]
            self.terminals[self.size : self.size + num_append] = terminals[:num_append]
            self.next_states[self.size : self.size + num_append] = next_states[:num_append]
            self.size += num_append
        # insert remaining transitions (randomly replacing prev trans)
        num_insert = len(states) - num_append
        if num_insert != 0:
            idxs = np.array(rand.sample(range(self.capacity + num_insert), num_insert))
            mask = idxs < self.capacity
            own_idxs = idxs * mask
            ins_idxs = np.array(range(num_append, num_append + num_insert)) * mask
            self.states[own_idxs] = states[ins_idxs]
            self.actions[own_idxs] = actions[ins_idxs]
            self.rewards[own_idxs] = rewards[ins_idxs]
            self.terminals[own_idxs] = terminals[ins_idxs]
            self.next_states[own_idxs] = next_states[ins_idxs]

    def sample(self, batch_size):
        idxs = rand.choices(range(self.size), k=batch_size)
        b_states = self.states[idxs]
        b_actions = self.actions[idxs]
        b_rewards = self.rewards[idxs]
        b_terminals = self.terminals[idxs]
        b_next_states = self.next_states[idxs]
        return b_states, b_actions, b_rewards, b_terminals, b_next_states