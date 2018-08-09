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

def render():
	env = gym.make('Taxi-v2')
	num_episodes = 10000

	Q = defaultdict(lambda: np.zeros(6))
	eps_decay = 0.99
	eps_min = 0.005

	alpha = 0.1
	gamma = 0.9

	state = env.reset()
	env.render()
	epsilon = 1.0
	for num_episode in range(num_episodes):
		print("num_episode: ", num_episode)
		epsilon = max(epsilon * 0.99, 0.05)
		for t in range(1000):
			policy_s = np.ones(6) * (epsilon / 6)
			policy_s[np.argmax(Q[state])] = 1 - epsilon + (epsilon / 6)
			# action = env.action_space.sample()
			action = np.random.choice(np.arange(6), p=policy_s)
			next_state, reward, done, _ = env.step(action)
			env.render()
			sys.stdout.flush()
			time.sleep(0.1)
			Q[state][action] = Q[state][action] + alpha * (reward + (gamma * np.max(Q[next_state])) - Q[state][action])
			state = next_state

			if done:
				# print('reward: ', reward)
				break

	env.close()


if __name__ == '__main__':
	# main()
	render()