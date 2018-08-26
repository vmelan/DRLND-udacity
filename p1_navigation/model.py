import torch 
import torch.nn as nn
import torch.nn.functional as F 


class QNetwork(nn.Module):
	""" Policy model that maps state to actions """

	def __init__(self, state_size, action_size, config):
		""" Initialize parameters and build model """
		super().__init__()


		# self.fc1 = nn.Linear(state_size, fc1_units)
		# self.fc2 = nn.Linear(fc1_units, fc2_units)
		# self.fc3 = nn.Linear(fc2_units, action_size)

		self.config = config 
		# Retrieve variable from config file
		hidden_layers_units = config["architecture"]["hidden_layers_units"]
		dropout_proba = config["architecture"]["dropout_proba"]

		# Add the first layer
		self.layers = nn.ModuleList([nn.Linear(state_size, hidden_layers_units[0])])

		# Add a variable number of more hidden layers 
		layer_sizes = zip(hidden_layers_units[:-1], hidden_layers_units[1:])
		self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

		# Add last layer 
		self.output = nn.Linear(hidden_layers_units[-1], action_size)

		# Dropout
		self.dropout = nn.Dropout(p=dropout_proba)

	def forward(self, x):
		""" Forward pass """
		# x = F.relu(self.fc1(state))
		# x = F.relu(self.fc2(x))
		# x = self.fc3(x)

		for layer in self.layers:
			x = F.relu(layer(x))
			if self.config["architecture"]["use_dropout"]:
				x = self.dropout(x)

		x = self.output(x)

		return x

if __name__ == '__main__':
	import json 
	with open("config.json", "r") as f:
		config = json.load(f)

	net = QNetwork(state_size=37, action_size=4, config=config)
	print("net:", net)
