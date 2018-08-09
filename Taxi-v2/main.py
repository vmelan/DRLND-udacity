from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt

import sys
from collections import defaultdict
import time

def plot_performance(num_episodes, avg_rewards, label, disp_plot=True):
	plt.plot(np.linspace(0, num_episodes, len(avg_rewards),endpoint=False), np.asarray(avg_rewards), label=label)
	plt.xlabel('Episode Number')
	plt.ylabel('Average Reward (Over Next %d Episodes)' % (100))
	plt.title(label + " " + "performance")
	if disp_plot: plt.show()

def plot_all_performances(num_episodes, all_avg_rewards, title):
	for (avg_reward, method) in zip(all_avg_rewards, ['Sarsa', 'Sarsamax (Q-Learning)', 'Expected Sarsa']):
		plot_performance(num_episodes, avg_reward, method, disp_plot=False)
	plt.title(title)
	plt.legend(loc='best')
	plt.show()

def main():
	env = gym.make('Taxi-v2')
	num_episodes = 300

	# Sarsa
	agent = Agent(method='Sarsa')
	sarsa_avg_rewards, sarsa_best_avg_reward = interact(env, agent, num_episodes=num_episodes)
	plot_performance(num_episodes, sarsa_avg_rewards, "Expected Sarsa", disp_plot=True)



	## Q-Learning
	agent = Agent(method='Q-Learning')
	sarsamax_avg_rewards, sarsamax_best_avg_reward = interact(env, agent, num_episodes=num_episodes)
	plot_performance(num_episodes, sarsamax_avg_rewards, "Sarsamax (Q-Learning)", disp_plot=True)

	## Expected Sarsa
	agent = Agent(method='Expected Sarsa')
	exp_sarsa_avg_rewards, exp_sarsa_best_avg_reward = interact(env, agent, num_episodes=num_episodes)
	plot_performance(num_episodes, exp_sarsa_avg_rewards, "Expected Sarsa", disp_plot=True)

	## All performances
	plot_all_performances(num_episodes, [sarsa_avg_rewards, sarsamax_avg_rewards, exp_sarsa_avg_rewards], 
		title="Comparison of Temporal Difference control methods")


if __name__ == '__main__':
	main()
