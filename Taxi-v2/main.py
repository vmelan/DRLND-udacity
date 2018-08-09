from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt

def plot_performance(num_episodes, avg_rewards, title):
	plt.plot(np.linspace(0, num_episodes, len(avg_rewards),endpoint=False),np.asarray(avg_rewards))
	plt.xlabel('Episode Number')
	plt.ylabel('Average Reward (Over Next %d Episodes)' % 100)
	plt.title(title)
	plt.show()

def main():
	env = gym.make('Taxi-v2')
	agent = Agent()
	num_episodes = 20000
	avg_rewards, best_avg_reward = interact(env, agent, num_episodes=num_episodes)
	plot_performance(num_episodes, avg_rewards, "Sarsamax (Q-Learning) performance")

if __name__ == '__main__':
	main()