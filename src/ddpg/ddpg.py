# Deep Q-Learning With Experience Replay (V. Mnih et al.)

from numpy.lib.polynomial import poly
import torch
import torch.nn as tnn
import torch.optim as topt
import torch.cuda as tcuda
import numpy as np

# local imports
from .nn import polyak_update


class DDPG:
    def __init__(self, actors, critics, target_actor, target_critic, replay_buf, learning_rate=0.0003, discount_factor=0.99, polyak_factor=0.99):
        # multiple actors and critics, only one set of targets
        self.actors = actors
        self.critics = critics
        self.target_actor = target_actor
        self.target_critic = target_critic
        # initialize target network weights and all actors and critics
        for a, c in zip(actors[1:], critics[1:]):
            polyak_update(self.actors[0], a, 0)
            polyak_update(self.critics[0], c, 0)
        polyak_update(self.actors[0], self.target_actor, 0)
        polyak_update(self.critics[0], self.target_critic, 0)
        self.replay_buf = replay_buf
        # initialize optimizer
        self.actor_opts = [topt.Adam(a.parameters(), learning_rate) for a in self.actors]
        self.critic_opts = [topt.Adam(c.parameters(), learning_rate) for c in self.critics]
        # get devices
        self.dev_gpu = torch.device("cuda" if tcuda.is_available() else "cpu")
        self.dev_cpu = torch.device("cpu")
        # create constant tensors
        self.discount_factor = torch.tensor(discount_factor, dtype=torch.float32, device=self.dev_gpu)
        self.polyak_factor = torch.tensor(polyak_factor, dtype=torch.float32, device=self.dev_gpu)
        # for random action noise
        self.np_rng = np.random.default_rng()

    def optimize(self, num_steps=1, batch_size=1000):
        for _ in range(num_steps): # number of gradient steps
            for actor, critic, opt_a, opt_c in zip(self.actors, self.critics, self.actor_opts, self.critic_opts):
                # get sample batch, convert numpy arrays to tensors and send to GPU
                batch_tuple = self.replay_buf.sample(batch_size)
                states, actions, rewards, terminals, next_states = [torch.from_numpy(b).float().to(self.dev_gpu) for b in batch_tuple]
                # update critic
                critic.train(True)
                targ_actor_out = self.target_actor(next_states)
                targ_critic_out = self.target_critic(torch.cat((next_states, targ_actor_out), -1))
                terminal_check = torch.tensor(1).float().to(self.dev_gpu) - terminals
                targ_vals = rewards + self.discount_factor * terminal_check * targ_critic_out
                # calculate MSE loss and perform gradient step
                critic_loss = tnn.MSELoss()(critic(torch.cat((states, actions), -1)), targ_vals.detach())
                opt_c.zero_grad() # zero/clear previous gradients
                critic_loss.backward()
                tnn.utils.clip_grad_norm_(critic.parameters(), 1)
                opt_c.step()
                # update actor
                actor.train(True)
                actor_loss = -torch.mean(critic(torch.cat((states, actor(states)), -1)))
                opt_a.zero_grad()
                actor_loss.backward()
                tnn.utils.clip_grad_norm_(actor.parameters(), 1)
                opt_a.step()
                # polyak update targets
                polyak_update(actor, self.target_actor, self.polyak_factor)
                polyak_update(critic, self.target_critic, self.polyak_factor)

    def get_actions(self, states, epsilon=0.0):
        actions_list = []
        for actor, state in zip(self.actors, states):
            actor_in = torch.from_numpy(np.array(state)).float().to(self.dev_gpu) # convert state, state can be numpy array or list
            actor.train(False)
            with torch.no_grad():
                actions = torch.tanh(actor(actor_in)).to(self.dev_cpu).numpy()
            # epsilon noise and clipping
            noise = epsilon * self.np_rng.random(actions.shape[0])
            actions_list.append(np.clip(actions + noise, -1, 1))
        return np.array(actions_list)