import numpy as np
import random 
import copy 
import logging 
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F 
import torch.optim as optim 
from utils import pick_device


class Agent():
	""" Agent used to interact with and learns from the environment """

	def __init__(self, state_size, action_size, config):
		""" Initialize an agent object """

		self.state_size = state_size
		self.action_size = action_size
		self.config = config

		# logging for this class 
		self.logger = logging.getLogger(self.__class__.__name__)

		# gpu support 
		self.device = pick_device(config, self.logger)

		## Actor local and target networks
		self.actor_local = Actor(state_size, action_size, config).to(self.device)
		self.actor_target = Actor(state_size, action_size).to(self.device)
		self.actor_optimizer = getattr(optim, config["optimizer_actor"]["optimizer_type"])(
			self.actor_local.parameters(), 
			betas=tuple(config["optimizer_actor"]["betas"], 
				**config["optimizer_actor"]["optimizer_params"]))

		## Critic local and target networks 
		self.critic_local = Critic(state_size, action_size, config).to(self.device)
		self.critic_target = Actor(state_size, action_size, config).to(self.device)
		self.actor_optimizer = getattr(optim, config["optimizer_critic"]["optimizer_type"])(
			self.critic_local.parameters(), 
			betas=tuple(config["optimizer_critic"]["betas"], 
				**config["optimizer_critic"]optimizer_critic["optimizer_params"]))

		## Noise process 
		self.noise = OUNoise(action_size)

		## Replay memory
		self.memory = ReplayBuffe(
			config=config, 
			action_size=action_size,
			buffer_size=int(config["DDPG"]["buffer_size"]),
			batch_size=config["trainer"]["batch_size"]
			)


	def step(self, state, action, reward, next_state, done):
		""" Save experience in replay memory, 
		and use random sample from buffer to learn """

		# Save experience in replay memory 
		self.memory.add(state, action, reward, next_state, done)

		# learn every timestep as long as enough samples are available in memory
		if len(self.memory) > self.config["trainer"]["batch_size"]:
			experiences = self.memory.sample()
			self.learn(experiences, self.config["DDPG"]["gamma"])


	def act(self, state):
		""" Returns actions for given state as per current policy """

		# Convert state to tensor
		state = torch.from_numpy(state).float().to(self.device)

		## Evaluation mode
		self.actor_local.eval()
		with torch.no_grad():
			# Forward pass of local actor network 
			action_values = self.actor_local.forward(state)

		## Training mode
		self.actor_local.train()
		# Add noise to improve exploration
		action_values += self.noise.sample()
		# Clip action to stay in the range [-1, 1] for our task
		action_values = np.clip(action_values, -1, 1)

		return action_values


	def learn(self, experiences, gamma): 
		""" Update value parameters using given batch of experience tuples """

		states, actions, rewards, next_states, dones = experiences

		## Update actor (policy) network using the sampled policy gradient
		# Compute actor loss 
		actions_pred = self.actor_local.forward(states)
		actor_loss = -self.critic_local.forward(states, actions_pred).mean()
		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		## Update critic (value) network
		# Get predicted next-state actions and Q-values from target models
		actions_next = self.actor_target.forward(next_states)
		Q_targets_next = self.critic_target.forward(next_states, actions_next)
		# Compute Q-targets for current states
		Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
		# Get expected Q-values from local critic model
		Q_expected = self.critic_local.forward(states)
		# Compute loss
		critic_loss = F.mse_loss(Q_expected, Q_targets)
		# Minimize the loss
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


		## Update target networks with a soft update 
		self.soft_update(self.actor_local, self.self.actor_target, self.config["DDPG"]["tau"])
		self.soft_update(self.critic_local, self.critic_target, self.config["DDPG"]["tau"])


	def soft_update(self, local_model, target_model, tau):
		""" Soft update model parameters,
		improves the stability of learning """

		for target_pararam, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)



class OUNoise():
	""" Ornstein-Uhlenbeck process """

	def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
		""" Initialize parameters and noise process """
		self.mu = mu * np.ones(size)
		self.theta = theta 
		self.sigma = sigma 
		self.reset()

	def reset(self):
		""" Reset the interal state (= noise) to mean (mu). """
		self.state = copy.copy(self.mu)

	def sample(self):
		""" Update internal state and return it as a noise sample """
		x = self.state 
		dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
		self.state = x + dx 

		return self.state



	