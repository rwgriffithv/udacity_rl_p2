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
    REQ_AVG_SCORE = 30 # mean agent espisode score requirement for environment to be solved
    NUM_PREV_SCORES_REQ = 100 # number of previous mean agent episode scores to average against REG_AVG_SCORE
    NUM_PREV_SCORES_TRAIN = 10 # number of previous mean agent episode scores to check to see if training should occur, None if check should not be made to stop training early
    NUM_EPISODES_UNTIL_TRAIN = 10 # number of startup episodes used to prefill the replay buffer before training, not included in final output
    # training constants
    REPBUF_TRAJ_CAPCITY = 1000 # actual number of samples within buffer is ~REPBUF_TRAN_PER_TRAJ times larger
    REPBUF_TRAN_PER_TRAJ = 1001 # must match ceil(environment's true length / K) (K defined below)
    SAMPLE_TRAJ_LENGTH = 10 # number of consecutive transitions that are taken as a sample from the replay buffer
    NUM_ATOMS = 12 # number of discrete distribution points for distributional Q-network to learn
    V_MIN = 1e-5 # minimum value for distrbutional Q-network, non-zero to avoid zero policy gradient coefficients
    V_MAX = 0.1 # maximum value for distrbutional Q-network
    POLICY_LR = 0.0001 # small due to frequency of gradient steps
    DISTQ_LR = 0.0002 # small due to frequency of gradient steps
    DISCOUNT_FACTOR = 0.5 # should inversely correlate with SAMPLE_TRAJ_LENGTH
    POLYAK_FACTOR = 0.95 # large due to frequency of gradient steps
    MAX_GRAD_NORM = 1.0 # maximum gradient norm
    REGULARIZATION_FACTOR = 1e-5 # regularization factor applied to actor and critic weights
    APPLY_PRIORITY_SCALING = False # whether priority of samples used is factored into critic loss
    NUM_GRAD_STEPS_PER_UPDATE = 1 # number of gradient steps taken per model update through "optimize" function call
    BATCH_SIZE = 128 # number of samples used per model update
    K = 1 # number of simulation steps per RL algorithm step (taken from DeepQ, not very effective in this environment)
    EPSILON_MIN = 0.05 # minimum noise
    EPSILON_MAX = 1.0 # maximum noise
    EPSILON_DECAY = 0.975 # noise decay
    PRIORITY_MIN = 0.0001 / BATCH_SIZE # minimum probability of a recently inserted transition to be sampled from replay buffer
    PRIORITY_MAX = 0.01 / BATCH_SIZE # maximum probability of a recently inserted transition to be sampled from replay buffer
    PRIOIRTY_DECAY = 0.999 # decay of sample probability
    PRIORITY_REWARD_FACTOR = 10 # factor sample probability is multiplied by if sample has favorable rewards (helps with initially sparse rewards environment)
    
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
    d4pg = D4PG(policy, distq, target_policy, target_distq, replay_buf, SAMPLE_TRAJ_LENGTH, V_MIN, V_MAX, NUM_ATOMS, \
        POLICY_LR, DISTQ_LR, DISCOUNT_FACTOR, POLYAK_FACTOR, MAX_GRAD_NORM, REGULARIZATION_FACTOR, APPLY_PRIORITY_SCALING)
    scores_history = [[] for _ in range(num_agents)] # list of num_agents lists of episode rewards sums (one list per agent)
    scores_averages = [] # list of average of rewards across all agents through an episode, used to determine if the agent has solved the environment
    epsilon = EPSILON_MAX
    priority = PRIORITY_MAX
    print("\n\ntraining (K=%d, PLR=%f, QLR=%f, BS=%d, ED=%f) ...." % (K, POLICY_LR, DISTQ_LR, BATCH_SIZE, EPSILON_DECAY))
    while len(scores_averages) < MAX_NUM_EPISODES + NUM_EPISODES_UNTIL_TRAIN:
        num_prev_scores_train = 0 if NUM_PREV_SCORES_TRAIN is None else min(NUM_PREV_SCORES_TRAIN, len(scores_averages))
        training = (len(scores_averages) > NUM_EPISODES_UNTIL_TRAIN) and ((sum(scores_averages[-num_prev_scores_train:]) / max(num_prev_scores_train, 1)) < REQ_AVG_SCORE)
        distq_losses, policy_losses = np.zeros(NUM_GRAD_STEPS_PER_UPDATE), np.zeros(NUM_GRAD_STEPS_PER_UPDATE)
        
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
            if training:
                d_l, p_l = d4pg.optimize(NUM_GRAD_STEPS_PER_UPDATE, BATCH_SIZE)
                distq_losses += d_l
                policy_losses += p_l
            scores += rewards # accumulate score
            if done:
                end_states = env_info.vector_observations
                replay_buf.append_end_states(end_states)
                break
        # check for environment being solved
        for s_h, s in zip(scores_history, scores):
            s_h.append(s)
        scores_averages.append(np.mean(scores))
        num_prev_scores = min(NUM_PREV_SCORES_REQ, len(scores_averages))
        avg_score = sum(scores_averages[-num_prev_scores:]) / num_prev_scores
        print("\nepisode %d:" % len(scores_averages) - 1)
        print("noise:                          %f" % (len(scores_averages) - 1, epsilon))
        if training:
            print("mean loss (distq, policy):  (%f, %f)" % (np.mean(distq_losses), np.mean(policy_losses)))
        print("average score:                  %f" % scores_averages[-1])
        print("average score for eps [%d, %d): %f" % avg_score)
        if avg_score > REQ_AVG_SCORE:
            break
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
        if training:
            priority = max(priority * PRIOIRTY_DECAY, PRIORITY_MIN)

    env.close()
    # save models and plot final rewards curve
    print("\n\nenvironment solved, saving model to qnet.pt and scores to scores.csv")
    with open("scores.csv", "w") as f:
        for s_h in scores_history:
            f.write(str(s_h[NUM_EPISODES_UNTIL_TRAIN:])[1:-1] + "\n")
    with open("scores_avg.csv", "w") as f:
        f.write(str(scores_averages[NUM_EPISODES_UNTIL_TRAIN:])[1:-1])
    save(policy.state_dict(), "policy.pt")
    save(distq.state_dict(), "distq.pt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nERROR:\tinvalid arguments\nUSAGE:\ttrain.py <unity_environment_executable>\n")
    else:
        train(sys.argv[1])