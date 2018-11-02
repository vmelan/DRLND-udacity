import numpy as np
import logging 
from models import Actor, Critic
from ReplayBuffer import ReplayBuffer
from OUNoise import OUNoise
import torch
import torch.nn.functional as F 
import torch.optim as optim 
from utils import pick_device

import pdb

class DDPGAgents():
	""" Agent used to interact with and learns from the environment """

	def __init__(self, state_size, action_size, config):
		""" Initialize an agent object """

		self.state_size = state_size
		self.action_size = action_size
		self.config = config

		# retrieve number of agents
		self.num_agents = config["DDPG"]["num_agents"]

		# logging for this class 
		self.logger = logging.getLogger(self.__class__.__name__)

		# gpu support 
		self.device = pick_device(config, self.logger)

		## Actor local and target networks
		self.actor_local = Actor(state_size, action_size, config).to(self.device)
		self.actor_target = Actor(state_size, action_size, config).to(self.device)
		self.actor_optimizer = getattr(optim, config["optimizer_actor"]["optimizer_type"])(
			self.actor_local.parameters(), 
			betas=tuple(config["optimizer_actor"]["betas"]), 
				**config["optimizer_actor"]["optimizer_params"])

		## Critic local and target networks 
		self.critic_local = Critic(state_size, action_size, config).to(self.device)
		self.critic_target = Critic(state_size, action_size, config).to(self.device)
		self.critic_optimizer = getattr(optim, config["optimizer_critic"]["optimizer_type"])(
			self.critic_local.parameters(), 
			betas=tuple(config["optimizer_critic"]["betas"]), 
				**config["optimizer_critic"]["optimizer_params"])

		## Noise process 
		self.noise = OUNoise((self.num_agents, action_size))

		## Replay memory
		self.memory = ReplayBuffer(
			config=config, 
			action_size=action_size,
			buffer_size=int(config["DDPG"]["buffer_size"]),
			batch_size=config["trainer"]["batch_size"]
			)


	def step(self, state, action, reward, next_state, done):
		""" Save experience in replay memory, 
		and use random sample from buffer to learn """

		# Save experience in replay memory shared by all agents
		for agent in range(self.num_agents):
			self.memory.add(state[agent, :], 
				action[agent, :], 
				reward[agent], 
				next_state[agent, :], 
				done[agent]
				)

		# learn every timestep as long as enough samples are available in memory
		if len(self.memory) > self.config["trainer"]["batch_size"]:
			experiences = self.memory.sample()
			self.learn(experiences, self.config["DDPG"]["gamma"])


	def act(self, states, add_noise=False):
		""" Returns actions for given state as per current policy """

		# Convert state to tensor²
		states = torch.from_numpy(states).float().to(self.device)

		# prepare actions numpy array for all agents
		actions = np.zeros((self.num_agents, self.action_size))

		## Evaluation mode
		self.actor_local.eval()
		with torch.no_grad():
			# Forward pass of local actor network 
			for agent, state in enumerate(states):
				action_values = self.actor_local.forward(state).cpu().data.numpy()
				actions[agent, :] = action_values

		# pdb.set_trace()
		## Training mode
		self.actor_local.train()
		if add_noise:
			# Add noise to improve exploration to our actor policy
			# action_values += torch.from_numpy(self.noise.sample()).type(torch.FloatTensor).to(self.device)
			actions += self.noise.sample()
		# Clip action to stay in the range [-1, 1] for our task
		actions = np.clip(actions, -1, 1)

		return actions


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
		Q_expected = self.critic_local.forward(states, actions)
		# Compute loss
		critic_loss = F.mse_loss(Q_expected, Q_targets)
		# Minimize the loss
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


		## Update target networks with a soft update 
		self.soft_update(self.actor_local, self.actor_target, self.config["DDPG"]["tau"])
		self.soft_update(self.critic_local, self.critic_target, self.config["DDPG"]["tau"])


	def soft_update(self, local_model, target_model, tau):
		""" Soft update model parameters,
		improves the stability of learning """

		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)


	def reset(self):
		""" Reset noise """
		self.noise.reset()






