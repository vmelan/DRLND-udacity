import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.function as F 


class Actor(nn.Module):
	""" Actor (Policy) model """

	def __init__(self, state_size, action_size, config):
		""" Initalize parameters and build model """

		super(Actor, self).__init__()
		fc1_units = config["architecture"]["fc1_units"]
		fc2_units = config["architecture"]["fc2_units"]

		self.fc1 = nn.Linear(in_features=state_size, out_features=fc1_units)
		self.fc2 = nn.Linear(in_features=fc1_units, out_features=fc2_units)

		# weights initialization 
		for m in self.modules():
			if isinstance(m, nn.Linear):
				# FC layers have weights initialized with Glorot 
				m.weight = nn.init.xavier_uniform(m.weight, gain=1)

	def forward(self, state):
		""" Build an actor (policy) network that maps states to actions """
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.tanh(self.fc3(x)) # outputs are in the range [-1, 1]

		return x


class Critic(nn.Module):
	""" Critic (Value) Model """

	def __init__(self, state_size, action_size, config):
		""" Initialize parameters and build model """
		super(Critic, self).__init__()

		fc1_units = config["architecture"]["fc1_units"]
		fc2_units = config["architecture"]["fc2_units"]

		self.fc1 = nn.Linear(in_features=state_size, out_features=fc1_units)
		self.fc2 = nn.Linear(in_features=fc1_units + action_size, 
			out_features=fc2_units)
		self.fc3 = nn.Linear(in_features=fc2_units, 1)

		# weights initialization 
		for m in self.modules():
			if isinstance(m, nn.Linear):
				# FC layers have weights initialized with Glorot 
				m.weight = nn.init.xavier_uniform(m.weight, gain=1)

	def forward(self, state, action):
		""" Build a critic (value) network that maps 
		(state, action) pairs -> Q-values """
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(torch.cat([x, action], dim=1))) # add action too for the mapping
		x = F.relu(self.fc3(x))

		return x

