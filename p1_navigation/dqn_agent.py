import numpy as np 
import random 
from collections import namedtuple, deque 
import logging 

from model import QNetwork 

import torch 
import torch.nn.functional as F 
import torch.optim as optim 
from utils import pick_device 

class Agent():
	""" Agent used to interact with and learns from the environment """

	def __init__(self, state_size, action_size, config):
		""" Initialize an Agent object """

		self.state_size = state_size
		self.action_size = action_size 
		self.config = config 

		# logging for this class 
		self.logger = logging.getLogger(self.__class__.__name__)

		# gpu support 
		self.device = pick_device(config, self.logger)

		## Q-Networks 
		self.qnetwork_local = QNetwork(state_size, action_size, config).to(self.device)
		self.qnetwork_target = QNetwork(state_size, action_size, config).to(self.device)

		## Get optimizer for local network 
		self.optimizer = getattr(optim, config["optimizer"]["optimizer_type"])(
			self.qnetwork_local.parameters(), **config["optimizer"]["optimizer_params"])

		## Replay memory
		self.memory = ReplayBuffer(
			action_size, 
			config["DQN"]["buffer_size"], 
			config["trainer"]["batch_size"]
			)

		## Initialize time step (for update every `update_every` steps)
		self.t_step = 0


	def step(self, state, action, reward, next_state, done):
		
		# Save experience in replay memory 
		self.memory.add(state, action, reward, next_state, done)

		# Learn every `update_every` time steps 
		self.t_step = (self.t_step + 1) % config["DQN"]["update_every"]
		if (self.t_step == 0):
			# If enough samples are available in memory, get random subset and learn
			if len(self.memory) > self.config["trainer"]["batch_size"]:
				experiences = self.memory.sample()
				self.learn(experiences, self.config["DQN"]["gamma"])



	def act(self, state, eps):
		""" Returns actions for given state as per current policy """

		state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
		
		## Evaluation mode
		self.qnetwork_local.eval()
		with torch.no_grad():
			action_values = self.qnetwork_local.forward(state)
		
		## Training mode 
		self.qnetwork_local.train()
		# Epsilon-greedy action selection 
		if random.random() > eps:
			# Choose the best action (exploitation)
			return np.argmax(action_values.cpu().data.numpy())
		else:
			# Choose random action (exploration)
			return random.choice(np.arange(self.action_size))


	def learn(self, experiences, gamma):
		""" Update value parameters using given batch of experience tuples """
		
		states, actions, rewards, next_states, dones = experiences 

		## TD target
		# Get max predicted Q-values (for next states) from target model
		Q_targets_next = torch.argmax(self.qnetwork_target(next_states).detach()).unsqueeze(1)
		# Compute Q-targets for current states 
		Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

		## old value
		# Get expected Q-values from local model 
		Q_expected = torch.gather(self.qnetwork_local(states), dim=1, index=actions)

		# Compute loss 
		loss = F.mse_loss(Q_expected, Q_targets)
		# Minimize loss 
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# update target network with a soft update
		self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config["DQN"]["tau"])



	def soft_update(self, local_model, target_model, tau):
		""" 
		Soft update model parameters
		θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
		"""

		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data_copy_(tau*local_param.data + (1.0 - tau)*target_param.data)


class ReplayBuffer():
	""" Fixed-size buffer to store experience tuples """ 

	def __init__(self, action_size, buffer_size, batch_size):
		""" Initialize a ReplayBuffer object """ 

		self.action_size = action_size 
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size 
		self.experience = namedtuple("Experience", 
			field_names=["state", "action", "reward", "next_state", "done"])

		# logging for this class 
		self.logger = logging.getLogger(self.__class__.__name__)

		# gpu support 
		self.device = pick_device(config, self.logger)

		
		def add(self, state, action, reward, next_state, done):
			""" Add a new experience to memory """ 
			e = self.experience(state, action, reward, next_state, done) 
			self.memory.append(e)

		def sample(self):
			""" Randomly sample a batch of experiences from memory """ 
			experiences = random.sample(self.memory, k=self.batch_size)

			states = torch.from_numpy(
				np.vstack([e.state for e in experiences if e is not None])
				).float().to(self.device)
			actions = torch.from_numpy(
				np.vstack([e.action for e in experiences if e is not None])
				).long().to(self.device)
			rewards = torch.from_numpy(
				np.vstack([e.reward for e in experiences if e is not None])
				).float().to(self.device)
			next_states = torch.from_numpy(
				np.vstack([e.next_state for e in experiences if e is not None])
				).float().to(self.device)
			dones = torch.from_numpy(
				np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
				).float().to(self.device)

			return (states, actions, rewards, next_states, dones)


		def __len__(self):
			""" Return the current size of internal memory """
			return len(self.memory)
