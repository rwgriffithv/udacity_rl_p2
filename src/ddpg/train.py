# training a single angent for a given UnityEnviroment executable using Deep Q-Learning

import sys
import numpy as np
from unityagents import UnityEnvironment, brain
from torch import save

# local imports
from .nn import build_actor_network, build_critic_network
from .replay import ReplayBuffer
from .ddpg import DDPG


def train(executable_path):
    # environment solution constants
    MAX_NUM_EPISODES = 200 # must solve environment before this
    REQ_AVG_SCORE = 30
    # training constants
    REPBUF_CAPCITY = int(1e5)
    LEARNING_RATE = 0.0003 # small due to frequency of gradient steps
    DISCOUNT_FACTOR = 0.99
    POLYAK_FACTOR = 0.975 # large due to frequency of gradient steps
    NUM_GRAD_STEPS_PER_UPDATE = 2 # 10
    BATCH_SIZE = 128
    K = 1 # number of simulation steps per RL algorithm step (taken from DeepQ)
    EPSILON_MIN = 0.01
    EPSILON_MAX = 1.0
    EPSILON_DECAY = 0.98
    
    # instantiate environment
    env = UnityEnvironment(file_name=executable_path)
    # get default brain name
    brain_name = env.brain_names[0]
    
    # get environment state and action size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = env_info.vector_observations.shape[1]
    action_size = env.brains[brain_name].vector_action_space_size
    num_agents = len(env_info.agents)
    
    # build sequential artificial neural networks for the actor, critic, and their target networks
    actors = [build_actor_network(state_size, action_size) for _ in range(num_agents)]
    target_actor = build_actor_network(state_size, action_size)
    critics = [build_critic_network(state_size + action_size, 1) for _ in range(num_agents)]
    target_critic = build_critic_network(state_size + action_size, 1)

    # replay buffer that allows for time-prioritized sampling with 
    replay_buf = ReplayBuffer(REPBUF_CAPCITY, state_size, action_size)

    # training using DDPG
    ddpg = DDPG(actors, critics, target_actor, target_critic, replay_buf, LEARNING_RATE, DISCOUNT_FACTOR, POLYAK_FACTOR)
    scores_history = [] # list of arrays of sums of rewards for each agent throughout an episode
    scores_averages = [] # list of average of rewards across all agents through an episode, used to determine if the agent has solved the environment
    epsilon = EPSILON_MAX
    max_avg_score = int(-1e6)
    print("\n\ntraining (K=%d, LR=%f, BS=%d, ED=%f) ...." % (K, LEARNING_RATE, BATCH_SIZE, EPSILON_DECAY))
    while len(scores_averages) < MAX_NUM_EPISODES:
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        terminals = np.zeros(num_agents) # environment does not have a true end point where the agent will receive no more rewards
        i = 0
        while True:
            done = False
            actions = ddpg.get_actions(states, epsilon)
            rewards = np.zeros(num_agents)
            for _ in range(K):
                env_info = env.step(actions)[brain_name]
                rewards += env_info.rewards
                next_states = env_info.vector_observations
                done = np.any(env_info.local_done)
                if done: # check if episode is done
                    break
            replay_buf.insert(states, actions, rewards[:,np.newaxis], terminals[:,np.newaxis], next_states)
            if i % 20 == 0:
                ddpg.optimize(NUM_GRAD_STEPS_PER_UPDATE, BATCH_SIZE)
            states = next_states
            i += 1
            # print("completed episode step: %d" % i)
            scores += rewards # accumulate score
            if done:
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

    env.close()
    # save models and plot final rewards curve
    print("\n\nenvironment solved, saving model to qnet.pt and scores to scores.csv")
    with open("scores.csv", "w") as f:
        f.write(str(scores_history.flatten())[1:-1])
    with open("scores_avg.csv" "w") as f:
        f.write(str(scores_averages)[1:-1])
    for i, (a, c) in enumerate(zip(actors, critics)):
        save(a.state_dict(), "actor_%d.pt" % i)
        save(c.state_dict(), "critic_%d.pt" % i)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nERROR:\tinvalid arguments\nUSAGE:\ttrain.py <unity_environment_executable>\n")
    else:
        train(sys.argv[1])