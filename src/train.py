# training a single angent for a given UnityEnviroment executable using Deep Q-Learning

import sys
import numpy as np
from unityagents import UnityEnvironment, brain
from torch import save

# local imports
from .nn import build_network
from .replay import Transition, ReplayBuffer
from .d4pg import D4PG


def train(executable_path):
    # environment solution constants
    MAX_NUM_EPISODES = 200 # must solve environment before this
    NUM_AGENTS = 20
    REQ_AVG_SCORE = 30
    # training constants
    REPBUF_CAPCITY = int(1e5)
    SAMPLE_TRAJ_LENGTH = 5 # number of consecutive transitions that are stored together in the replay buffer
    LEARNING_RATE = 0.0003 # small due to frequency of gradient steps
    DISCOUNT_FACTOR = 1
    POLYAK_FACTOR = 0.995 # large due to frequency of gradient steps
    NUM_GRAD_STEPS_PER_UPDATE = 1
    BATCH_SIZE = 128
    K = 2 # number of simulation steps per RL algorithm step
    EPSILON_MIN = 0.01
    EPSILON_MAX = 1.0
    EPSILON_DECAY = 0.99
    # epsilon refreshing to encourage exploration after standard epsilon annealing
    EPSILON_REFRESH = 0.05 # for refreshing the value of epsilon
    AVG_SCORE_DECREASE_TO_REFRESH = 0.5 # average score decrease that prompts an epsilon refresh
    
    # instantiate environment
    env = UnityEnvironment(file_name=executable_path)
    # get default brain name
    brain_name = env.brain_names[0]
    
    # get environment state and action size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = env.brains[brain_name].vector_action_space_size
    
    # build sequential artificial neural networks for the q-function and target q-function
    qnet = build_network(state_size, action_size)
    target_qnet = build_network(state_size, action_size) # target q network

    # states stored in replay buffer are environment states concatenated with previous action id
    replay_buf = ReplayBuffer(REPBUF_CAPCITY, state_size)

    # training using Distributed Distributional Deep Deterministic Policy Gradient (D4PG)
    d4pg = D4PG(qnet, target_qnet, replay_buf, LEARNING_RATE, DISCOUNT_FACTOR, POLYAK_FACTOR)
    scores = [] # sum of rewards throughout an episode, used to determine if the agent has solved the environment
    epsilon = EPSILON_MAX
    stagnation_count = 0
    max_avg_score = int(-1e6)
    print("\n\ntraining (K=%d, LR=%f, BS=%d) ...." % (K, LEARNING_RATE, BATCH_SIZE))
    while len(scores) < MAX_NUM_EPISODES:
        score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        while True:
            action = deepq.get_action(state, epsilon)
            reward = 0
            for _ in range(K):
                env_info = env.step(action)[brain_name]
                reward += env_info.rewards[0]
                terminal = 1 if env_info.local_done[0] else 0
                if terminal: # check if episode is done
                    break
            next_state = env_info.vector_observations[0]
            replay_buf.insert([Transition(state, action, reward, 0, next_state)])
            deepq.optimize(NUM_GRAD_STEPS_PER_UPDATE, BATCH_SIZE)
            state = next_state  # roll over state
            score += reward # accumulate score
            if terminal:
                break
        # check for environment being solved
        scores.append(score)
        num_prev_scores = min(100, len(scores))
        avg_score = sum(scores[-num_prev_scores:]) / num_prev_scores
        print("\raverage score for episodes [%d, %d):\t%f" % (len(scores) - num_prev_scores, len(scores), avg_score), end="")
        if avg_score > REQ_AVG_SCORE:
            break
        # update epsilon according to stagnation or average score decline
        max_avg_score = max(max_avg_score, avg_score)
        score_diff = max_avg_score - avg_score
        stagnation_count = 0 if epsilon != EPSILON_MIN or score_diff == 0 else stagnation_count + 1
        if epsilon == EPSILON_MIN and (stagnation_count == 100 or score_diff >= AVG_SCORE_DECREASE_TO_REFRESH):
            print("\nrefreshing epsilon to %f,\tmax average score: %f\n" % (EPSILON_REFRESH, max_avg_score))
            epsilon = EPSILON_REFRESH
            stagnation_count = 0
            max_avg_score = -1e6 # goal is now to surpass a newly found max average (and hopefully surpass previous max as well)
        else:
            epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

    env.close()
    # save models and plot final rewards curve
    print("\n\nenvironment solved, saving model to qnet.pt and scores to scores.csv")
    with open("scores.csv", "w") as f:
        f.write(str(scores)[1:-1])
    save(qnet.state_dict(), "qnet.pt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nERROR:\tinvalid arguments\nUSAGE:\ttrain.py <unity_environment_executable>\n")
    else:
        train(sys.argv[1])