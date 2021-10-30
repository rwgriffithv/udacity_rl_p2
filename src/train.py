# training a single angent for a given UnityEnviroment executable using Deep Q-Learning

import sys
import numpy as np
from unityagents import UnityEnvironment, brain
from torch import save

# local imports
from .nn import build_actor_network, build_critic_network
from .replay import ConsecutiveReplayBuffer
from .d4pg import D4PG


def train(executable_path):
    # environment solution constants
    MAX_NUM_EPISODES = 1000 # must solve environment before this
    REQ_AVG_SCORE = 30
    # training constants
    REPBUF_TRAJ_CAPCITY = 200 # actual number of samples within buffer is ~REPBUF_TRAN_PER_TRAJ times larger
    REPBUF_TRAN_PER_TRAJ = 501 # must match ceil(environment's true length / K) (K defined below)
    SAMPLE_TRAJ_LENGTH = 5 # number of consecutive transitions that are taken as a sample from the replay buffer
    NUM_ATOMS = 24 # number of discrete distribution points for distributional Q-network to learn
    V_MIN = 1e-7 # minimum value for distrbutional Q-network, non-zero to avoid zero policy gradient coefficients
    V_MAX = 0.1 # maximum value for distrbutional Q-network
    POLICY_LR = 0.001 # small due to frequency of gradient steps
    DISTQ_LR = 0.001 # small due to frequency of gradient steps
    DISCOUNT_FACTOR = 0.5 # should inversely correlate with SAMPLE_TRAJ_LENGTH
    POLYAK_FACTOR = 0.99 # large due to frequency of gradient steps
    NUM_GRAD_STEPS_PER_UPDATE = 2
    BATCH_SIZE = 256
    K = 2 # number of simulation steps per RL algorithm step (taken from DeepQ)
    EPSILON_MIN = 0.01
    EPSILON_MAX = 1.0
    EPSILON_DECAY = 0.95
    PRIORITY_MIN = 0.0001 / BATCH_SIZE
    PRIORITY_MAX = 0.01 / BATCH_SIZE
    PRIOIRTY_DECAY = 0.999
    PRIORITY_REWARD_FACTOR = 10
    
    # instantiate environment
    env = UnityEnvironment(file_name=executable_path)
    # get default brain name
    brain_name = env.brain_names[0]
    
    # get environment state and action size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = env_info.vector_observations.shape[1]
    action_size = env.brains[brain_name].vector_action_space_size
    num_agents = len(env_info.agents)
    
    # build sequential artificial neural networks for the policy, distributional Q-network, and their targets
    policy = build_actor_network(state_size, action_size)
    target_policy = build_actor_network(state_size, action_size)
    distq = build_critic_network(state_size + action_size, NUM_ATOMS)
    target_distq = build_critic_network(state_size + action_size, NUM_ATOMS)

    # replay buffer that allows for prioritized sampling and the sampling of consecutive transitions
    replay_buf = ConsecutiveReplayBuffer(REPBUF_TRAJ_CAPCITY, REPBUF_TRAN_PER_TRAJ, SAMPLE_TRAJ_LENGTH, state_size, action_size)

    # training using Distributed Distributional Deep Deterministic Policy Gradient (D4PG)
    d4pg = D4PG(policy, distq, target_policy, target_distq, replay_buf, SAMPLE_TRAJ_LENGTH, V_MIN, V_MAX, NUM_ATOMS, POLICY_LR, DISTQ_LR, DISCOUNT_FACTOR, POLYAK_FACTOR)
    scores_history = [] # list of arrays of sums of rewards for each agent throughout an episode
    scores_averages = [] # list of average of rewards across all agents through an episode, used to determine if the agent has solved the environment
    epsilon = EPSILON_MAX
    priority = PRIORITY_MAX
    max_avg_score = int(-1e6)
    print("\n\ntraining (K=%d, PLR=%f, QLR=%f, BS=%d, ED=%f) ...." % (K, POLICY_LR, DISTQ_LR, BATCH_SIZE, EPSILON_DECAY))
    while len(scores_averages) < MAX_NUM_EPISODES:
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        terminals = np.zeros(num_agents) # environment does not have a true end point where the agent will receive no more rewards
        while True:
            done = False
            states = env_info.vector_observations
            actions = d4pg.get_actions(states, epsilon)
            rewards = np.zeros(num_agents)
            for _ in range(K):
                env_info = env.step(actions)[brain_name]
                rewards += env_info.rewards
                done = np.any(env_info.local_done)
                if done: # check if episode is done
                    break
            prf_mask = rewards > 0
            priorities = np.full(num_agents, priority / num_agents)
            priorities[prf_mask] *= PRIORITY_REWARD_FACTOR
            replay_buf.insert_transitions(priorities, states, actions, rewards, terminals)
            d4pg.optimize(NUM_GRAD_STEPS_PER_UPDATE, BATCH_SIZE)
            scores += rewards # accumulate score
            if done:
                end_states = env_info.vector_observations
                replay_buf.append_end_states(end_states)
                break
        # check for environment being solved
        scores_history.append(scores)
        scores_averages.append(np.mean(scores))
        num_prev_scores = min(100, len(scores_averages))
        avg_score = sum(scores_averages[-num_prev_scores:]) / num_prev_scores
        print("\raverage score for episodes [%d, %d):\t%f" % (len(scores_averages) - num_prev_scores, len(scores_averages), avg_score), end="")
        if avg_score > REQ_AVG_SCORE:
            break
        max_avg_score = max(max_avg_score, avg_score)
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
        priority = max(priority * PRIOIRTY_DECAY, PRIORITY_MIN)

    env.close()
    # save models and plot final rewards curve
    print("\n\nenvironment solved, saving model to qnet.pt and scores to scores.csv")
    with open("scores.csv", "w") as f:
        f.write(str(scores_history.flatten())[1:-1])
    with open("scores_avg.csv" "w") as f:
        f.write(str(scores_averages)[1:-1])
    save(policy.state_dict(), "policy.pt")
    save(distq.state_dict(), "distq.pt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nERROR:\tinvalid arguments\nUSAGE:\ttrain.py <unity_environment_executable>\n")
    else:
        train(sys.argv[1])