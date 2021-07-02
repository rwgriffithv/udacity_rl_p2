import numpy as np

# a replay buffer containing consecutive transitions, grouped by trajectory
# assumes constant-size trajectories, states, and actions
# each trajectory has one extra state, corresponding to the "next state" of the final transition
# transitions are sampled as consectuive groups (within a trajectory) without priority
# future implementations may implement prioritized replay as needed
class ConsecutiveReplayBuffer:
    def __init__(self, capacity, traj_size, sample_size, state_size, action_size):
        self.capacity = capacity # max number of trajectories to contain
        self.traj_size = traj_size # number of transitions in a trajectory
        self.sample_size = sample_size # number of consecutive transitions taken in each sample
        self.num_traj = 0 # number of populated trajectories
        self.num_tran = 0 # number of populated transitions
        self.sizes = np.zeros(capacity, np.dtype(int)) # number of transitions currently in each trajectory
        self.weights = np.zeros((capacity, traj_size), np.dtype(float)) # probability of each transition, sums to 1
        self.weight_discount = np.zeros((capacity, traj_size), np.dtype(float)) # persistent storage for efficiency
        self.traj_weights = np.zeros(capacity, np.dtype(float)) # sum of weights for completed trajectory, 0 otherwise
        # transition storage
        self.states = np.zeros((capacity, traj_size + 1, state_size), np.dtype(float))
        # for discrete actions dtype is always int; using float for future continuous action use
        self.actions = np.zeros((capacity, traj_size, action_size), np.dtype(float))
        self.rewards = np.zeros((capacity, traj_size, 1), np.dtype(float))
        self.terminals = np.zeros((capacity, traj_size, 1), np.dtype(int))
        # for tracking current trajectories being appended/inserted
        self.traj_idxs = None
        # for random sampling
        self.np_rng = np.random.default_rng()

    # set random indices to insert new trajectories into, should only be invoked internally
    def _set_insert_trajectory_indices(self, num_trajectories):
        if self.traj_idxs is not None:
            return
        num_append = min(num_trajectories, self.capacity - self.num_traj)
        self.traj_idxs = [self.num_traj + i for i in range(num_append)]
        num_insert = num_trajectories - num_append
        inv_traj_weights = 1 - self.traj_weights
        self.traj_idxs += self.np_rng.choice(range(num_insert), size=num_insert, replace=False, p=inv_traj_weights[:num_insert]).tolist()
        # size and weight updates
        self.sizes[self.traj_idxs] = 0
        self.num_traj += num_append
        if num_insert != 0:
            self.num_tran -= num_insert * self.traj_size
            self.weight_discount.fill(1 / (1 - np.sum(self.weights[self.traj_idxs])))
            self.weights *= self.weight_discount
            self.weights[self.traj_idxs].fill(0)
            self.traj_weights[self.traj_idxs] = 0

    # one transition per trajectory index, prob = probability to randomly sample one of the inserted transitions
    def insert_transitions(self, prob, states, actions, rewards, terminals):
        self._set_insert_trajectory_indices(len(rewards))
        tran_idxs = [self.sizes[i] for i in self.traj_idxs]
        singleton_idxs = [1] * len(self.traj_idxs)
        self.states[[self.traj_idxs, tran_idxs]] = states
        self.actions[[self.traj_idxs, tran_idxs]] = actions
        self.rewards[[self.traj_idxs, tran_idxs, singleton_idxs]] = rewards
        self.terminals[[self.traj_idxs, tran_idxs, singleton_idxs]] = terminals
        # size and weight updates
        # mask weights out for transitions that end a trajectory and cannot be selected as starting a sample
        tran_mask = np.array([1 if i < (self.traj_size - self.sample_size + 1) else 0 for i in tran_idxs])
        self.sizes[self.traj_idxs] += 1
        self.num_tran += len(self.traj_idxs)
        prob = max(0.0, min(1.0, prob))
        masked_probs = tran_mask * prob
        self.weight_discount.fill(1 - prob)
        self.weights *= self.weight_discount
        self.weights[[self.traj_idxs, tran_idxs]] = (1 / max(1, np.sum(tran_mask))) * masked_probs
        # leave trajectory probability at 0 until it has enough transitions to select a sample from
        traj_mask = np.array([1 if i >= self.sample_size else 0 for i in tran_idxs])
        self.traj_weights[self.traj_idxs] = traj_mask * np.sum(self.weights[self.traj_idxs], axis=-1)

    # should only be invoked after inserting self.traj_size transitions into these trajectories
    def append_end_states(self, states):
        # does not count as a full transition, no size increments or weight updates
        self.states[[self.traj_idxs, [self.traj_size] * len(self.traj_idxs)]] = states
        # clear current trajectory indices
        self.traj_idxs = None

    # get random sample of transitions with replacement
    # each sample is a set of consecutive transitions
    # sample size determines the number of consecutive transitions
    def sample(self, num_samples):
        # self.traj_weights[i] will be 0 for all trajectories without enough transitions to populate a full sample of self.sample_size
        traj_idxs = self.np_rng.choice(range(self.num_traj), size=num_samples, replace=True, p=self.traj_weights[:self.num_traj])
        tran_start_idxs = [self.np_rng.choice(range(self.sizes[i] - self.sample_size + 1)) for i in traj_idxs]
        b_weights = self.weights[[traj_idxs, tran_start_idxs]]
        # each sample from a trajectory will have self.sample_size transitions
        traj_idxs = [i for i in traj_idxs for _ in range(self.sample_size)]
        # each set of states is of size self.sample_size + 1 to include final next state
        tran_state_idxs = [i + j for i in tran_start_idxs for j in range(self.sample_size + 1)]
        b_states = self.states[traj_idxs, tran_state_idxs]
        # each set of actions, rewards, and terminals are only of size self.sample_size
        tran_idxs = [i + j for i in tran_start_idxs for j in range(self.sample_size)]
        b_actions = self.actions[traj_idxs, tran_idxs]
        b_rewards = self.rewards[traj_idxs, tran_idxs]
        b_terminals = self.terminals[traj_idxs, tran_idxs]
        return b_states, b_actions, b_rewards, b_terminals, b_weights