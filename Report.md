# Udacity Deep Reinforcement Learning<br>Project 2 Version 2 Report
Robert Griffith  
31 October 2021


## Project Description

The goal of version 2 of this project was to use deep reinforcement learning  
techniques for continuous action and state spaces to learn a policy that  
provides an agent with continuous control over a double-jointed arm attempting  
to move to target locations in a provided UnityEnvironment. This Reacher  
environment is provided by Udacity and follows the architecture of the  
open-source Unity plugin **Unity Machine Learning Agents (ML-Agents)**.<br>
Version 2 specifically involves the use of deep reinforcement learning  
techniques that make use of multiple (non-interacting, parallel) copies of the  
same agent operating in parallel simulations to distribute the task of  
gathering experience.

In this Reacher environment, an agent is provided a reward of +0.1 for each  
timestep that the agent's "hand" is in the goal location. The state space that  
the agent perceives to perform actions in the environment has 33 dimensions and  
contains the position, rotoation, velocity, and angular velocities of the agent  
arm. Each action has 4 dimensions corresponding to torque applicable to two  
joints of the agent arm.

This task is episodic, and the environment is considered "solved" upon the mean  
earned agent score (taken over all 20 agents) averaged over 100 consecutive   
episodes being greater than or equal to +30. An individual agent "score" for an  
episode is the cumulative sum of that agent's reward from each time step in the  
episode.

In order to complete this project, not only does a deep reinforcement learning  
algorithm need to be implemented, but a myriad of hyperparameters, neural  
network model structures, and transition augmentations tailored to the Reacher  
environment need to be explored and implemented as well.

## Learning Algorithm - [Distributed Distributional<br>Deterministic Policy Gradients (Barth-Maron et al., 2018)](https://arxiv.org/pdf/1804.08617.pdf)

I chose version 2 of this project so I could specifically apply an algorithm  
other than DDPG and challenge myself more when in the pursuit of "solving" the  
afformentioned environment (unlike in project 1 where I was content making minor  
improvements to [Deep Q-Learning (Mnih et al., 2015)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). For the purpose of brevity,  
the following explanations assume familiarity with Deep Q-Learning and the core  
deep reinforcement learning concepts described within its respective paper.

In pursuit of a more complex algorithm, I chose to implement D4PG, as described  
in the above-cited paper. I implemented the gradient step portions of the  
algorithm for both the actor and critic networks, as well as the application of  
random noise to the policy output within the `D4PG` class in `src/d4pq.py`. The  
experience replay buffer used to store and randomly sample weighted transitions  
(of with an arbitrary number of consecutive transitions after each sample) is  
implemented within the `ConsecutiveReplayBuffer` class in `src/replay.py`.  

To briefly summarize the D4PG algorithm, the first framework to describe is the  
relationship between "actor" and "critic" networks. An "actor" network *&pi;<sub>&theta;</sub>*,<br>
parameterized by traiable weights *&theta;*, produces continuous actions *a*  
from state input *s* such that *&pi;<sub>&theta;</sub>*(*s*) = *a*. This is naturally used during execution  
to produced the desired actions for a given agent. A "critic" network *Q<sub>&omega;</sub>*,<br>
parameterized by trainable weights *&omega;*, is usually learning an action-value  
function which takes some state *s* and executed action *a* as input and produces  
a value *V* such that *Q<sub>&omega;</sub>*(*s*, *a*) = *V*. The loss minimized to compute the gradients  
applied to *&omega;* is usually related to the difference between *V* and some recorded  
reward *r* (or set of discounted rewards *R* and also target action-value function  
*Q<sub>&omega;__'__</sub>* parameterized by trainable weights *&omega;__'__*) observed from the environment  
upon executing action *a* from state *s*. The gradients applied to *&theta;* are taken from  
minimizing the loss of -*Q<sub>&omega;</sub>*(*s*, *&pi;<sub>&theta;</sub>*(*s*)) such that the policy network is "encouraged"  
to learn to map states to actions that yield higher value *V*.

D4PG implements this classical actor-critic framework but applies an interesting  
twist to the critic network. Instead of learing to map *s* and *a* to *V*, D4PG earns its  
second D for "Distributional" by mapping *s* and *a* to a random variable such that  
the distributional action-value function *Z* can be defined in relation to *V* and *Q*  
as *Q*(*s*, *a*) = *V* = __E__ *Z*(*s*, *a*) where **E** is the expected value operator and *Z*(*s*, *a*)<br>
returns a distribution from *V<sub>min</sub>* to *V<sub>max</sub>*. Thus in practice we train the critic network  
*Z<sub>&omega;*, parameterized by traiable weights *&omega;*, which will have *l* output logits describing  
the *l* discrete atoms of this discretized distribution. These final output logits have  
a softmax activation function applied to them to get the probabilities of each atom,  
where *p<sub>i</sub>* is the probability at atom *i* calculated from output logit *&omega;<sub>i</sub>*. The actual  
value *v<sub>i</sub>* that atom *i* represents is then *i* * (*V<sub>max</sub>* - *V<sub>min</sub>*) / (*l* - 1).  
__E__ *Z<sub>&omega;</sub>*(*s*, *a*) is then simply calculated by __&Sigma;__<sub>*i*</sub> *p<sub>i</sub>* *v<sub>i</sub>* and used in the policy loss the same  
*Q*(*s*, *a*) is used. The loss used to calculate the gradients applied to *&omega;* however is more  
involved and is the binary cross entropy of the output *p<sub>i</sub>* probabilities from *Z<sub>&omega;</sub>*(*s*, *a*)<br>
and the categorical projection of the sum of a set of discounted consecutive recorded rewards *R*<br>
and the output *p<sub>i</sub>* probabilities of a target critic network *Z<sub>&omega;__'__</sub>*(*s*__'__, *&pi;<sub>&theta;__'__</sub>*(*s*__'__)). The details of  
the categorical projection can be found in Appendix A of the paper, and my  
implementation found in `src/nn.py`.



"distributed" aspect of it. During training, some positive non-zero integer  
number of agents will be executing actions the fundamental idea is to  
estimate the action-value function by using the Bellman equation in iterative  
updates. The optimal action-value function is denoted by _Q*(s,a)_, where *s*  
is a state and *a* is an action. The learned action-value function estimator  
*Q<sub>&theta;</sub>(s,a)* is parameterized by *&theta;*. In Deep Q-Learning, *Q<sub>&theta;</sub>(s,a)* is a neural network that  
learns to approximate _Q*_(*s,a*) through gradient descent updates to network  
weights/parameters *&theta;*. This neural network action-vaue approximator is  
referred to as a Q-network (`qnet` in the code). The Bellman equation  
_Q*_(*s,a*) = **E**<sub>s'</sub>[*r* + *&gamma;* max<sub>*a'*</sub> _Q*_(*s',a'*)] is used to produce that target optimal value of  
_Q*_(*s,a*) for a given "transition" (*s,a,r,t,s'*) (state, action, reward, next state) that  
encodes the information of an agent's experience of a given time step. The value  
_Q*_(*s',a'*) is estimated by a target delayed Q-network *Q<sub>&theta;'</sub>*(*s',a'*) (`target_qnet` in the  
code) that shares the same structure as *Q<sub>&theta;</sub>* but uses parameters *&theta;'* that are  
equal to a previous iteration of parameters *&theta;*. Through many simulations,  
transitions are collected in a replay buffer and randomly sampled to perform  
updates to the Q-network by minimizing the difference between *Q<sub>&theta;</sub>*(*s,a*) and the  
Bellman-equation-computed _Q*_(*s,a*) (using *Q<sub>&theta;'</sub>*(*s',a'*)) for all sampled transitions. This  
minimization is performed by computing the gradient of a selected loss function  
with respect to Q-network parameters *&theta;*. A simple example of such a loss function is  
"Mean-Squared-Error": &nbsp; **E**<sub>*s,a,r,s'*</sub>(*Q<sub>&theta;</sub>*(*s,a*) - **E**<sub>s'</sub>[*r* + *&gamma;* max<sub>*a'*</sub> *Q<sub>&theta;'</sub>*(*s',a'*)])<sup>2</sup>. &nbsp;

My implementation of the replay buffer has slight modifications from the buffer  
in the paper, as a new transition is not guaranteed to be inserted if the  
buffer is full (it is logically inserted and then randomly selected to be removed  
so the buffer does not exceed its capacity). Similarly, when randomly sampling  
from the buffer, transitions are sampled with replacement. This allows early  
training iterations, performed upon a nearly empty replay buffer with transitions  
taken with many random actions, to have higher sample-efficiency and extract  
information from these more exploratory actions while they are still in the  
buffer and not unlikely to be sampled. The replay buffer capacity used is 1e6.

The gradient step implemented within `DeepQ` is exactly as defined within the  
paper; minimizing the exact same loss. The target action-value network  
(`target_qnet` in the code) is updated using polyak averaging however, where the  
network parameters *&theta;'* are updated as a weighted average of themselves and the  
parameters *&theta;* from the updating action-value network (`qnet` in the code). In the  
paper the target action-value network was instead updated by simply setting its  
parameters equal to those of a previous iteration of the updating action-value  
network. This change helps prevent the difference between the two action-value  
networks from approaching zero, as is possible if model updates are small and  
the target action-value network's parameters are updated frequently. The polyak  
factor *&tau;* used is 0.995, where *&theta;'* = *&tau;* * *&theta;'* + (1 - *&tau;*) * *&theta;*.

Two more small changes not specified in the Deep Q-Learning paper were made, one  
of which was to clamp transition rewards to fit within the range [-1, 1] during training  
so as to provide more regular reward inputs that simplify the optimal _Q*(s,a)_ that the  
training *Q<sub>&theta;</sub>* is training to approximate without changing which action the action-value  
function should learn to prefer for such transitions with clamped/clipped rewards. The  
other small change was to implement a form of *&epsilon;* "refreshing" that can occur once the  
minimum *&epsilon;* value is reached after annealing. *&epsilon;* is "refreshed" to the small value of  
0.05 (5% random actions) to encourage exploration and hopefully contribute to escaping  
the current local maxima. Once the minimum *&epsilon;* value, 0.01, is reached either originally  
or after refreshing, a maximum average score is maintained over the past 100 scores.  
If the agent is stuck in a local maxima and is unable to improve rewards for 100 episodes  
or if the agent's average score over the past 100 episodes is less than the current  
maximum average score by more than 0.5, then *&epsilon;* is refreshed in this way. This was a very  
useful feature when my hyperparameters were not tuned as well, however in my final  
training run I did not make use of this feature as none of the above conditions were ever  
met and the agent did not get stuck in a local maxima. The episode length of 100 used  
for these various hyperparameters concerning *&epsilon;* refreshing is chosen to match the length  
of the score window used to determine if the environment is solved to be analagous to  
the average score metric that we are primarily concerned with.

Hyperparameters used during training that have not been mentioned above are included  
here:
* *Q<sub>&theta;</sub>* learning rate: **0.0003** (small due to frequency of gradient steps)
* discount factor *&gamma;*: **1**
* (mini) batch size: **128** (chosen to be larger than in the Deep Q-Learning paper to help stability)
* *&epsilon;* max: **1.0**
* *&epsilon;* decay: **0.99** (*&epsilon;<sub>t+1</sub>* = 0.99 * *&epsilon;<sub>t</sub>*)
* *K* (number of simulation steps per algorithm step, see paper): **2**

The neural network architecture used for the Q-function *Q<sub>&theta;</sub>* (and *Q<sub>&theta;'</sub>*) is a sequential multilayer  
perceptron with layers only consisting of "linear" layers and ReLU activation layers.  
Each linear layer is comprised of parameters "weights" *w* and "biases" *b*, where  
given an input *i* the layer outputs _w*i + b_ where the weights and the input must  
match their final and first dimensions respectively, and the biases are added  
by an elementwise broadcast along the final dimensions of _w*i_. The layer order used  
is provided as follows:
* Linear layer with weights of shape (37, 128), biases of shape (128,)
* ReLU activation layer
* Linear layer with weights of shape (128, 128), biases of shape (128,)
* ReLU activation layer
* Linear layer with weights of shape (128, 4), biases of shape (4,)

## Plot of Rewards
![reward plots](./model/plot.png)


## Ideas for Future Work

I am particularly interested in continuous actions reinforcement learning algorithms, and  
I have implemented Soft Actor Critic and applied it to my current work as an AI software  
engineer. It would be interesting to try to apply the learned temperature variable in the  
Soft Actor Critic algorithm to a discrete action algorithm like Deep Q-Learning or Vanilla  
Policy Gradient and see how they perform in a simple environment like this. I essentially  
tried to mimic the functionality of the temperature variable with my somewhat hacky *&epsilon;*  
"refreshing" implementation here.

I was also planning on implementing [Double Deep Q-Learning (van Hasselt et al., 2015)](https://arxiv.org/pdf/1509.06461.pdf)  
and potentially [Prioritized Experience Replay (Schaul et al., 2016)](https://arxiv.org/pdf/1511.05952.pdf) if for some reason I was  
to be unsuccesful in using only Deep Q-Learning. This need (thankfully) did not arise, so it  
remains as potential future work to at least compare the performance of each of these  
techniques.