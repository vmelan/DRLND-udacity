import json
import logging
import torch
import numpy as np
from collections import deque
from dqn_agent import Agent
from utils import ensure_dir
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

plt.ion()


def dqn(agent,
        brain_name,
        config,
        n_episodes, 
        max_timesteps_per_ep, 
        eps_start, 
        eps_end, 
        eps_decay
        ):
    
    """
    Deep Q-Learning
    """
    logger = logging.getLogger('dqn') # logger 
    flag = False # When environment is technically solved
    # Save path
    save_path = config["trainer"]["save_dir"] + config["exp_name"] + "/" 
    ensure_dir(save_path)    
    scores = [] # list containing scores from each episodes
    scores_window = deque(maxlen=100)
    epsilon = eps_start # init epsilon
    
    for i_episode in range(1, n_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get the current state
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_timesteps_per_ep):
            # choose action based on epsilon-greedy policy
            action = agent.act(state, epsilon) 
            # send the action to the environment
            env_info = env.step(action)[brain_name] 
            # get the next state
            next_state = env_info.vector_observations[0]
            # get the reward
            reward = env_info.rewards[0]
            # see if episode has finished 
            done = env_info.local_done[0]
            # step 
            agent.step(state, action, reward, next_state, done)
            # cumulative rewards into score variable
            score += reward
            # get next_state and set it to state
            state = next_state
            
            if done: 
                break
                
        # Update epsilon 
        epsilon = max(eps_decay*epsilon, eps_end)
        
        # save most recent score
        scores.append(score)
        scores_window.append(score)
        
        logger.info('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)))

        if (i_episode % 100 == 0): 
            logger.info("\rEpisode {}\tAverage Score: {:.3f}".format(i_episode, \
                                                               np.mean(scores_window)))
        
        # Save occasionnally
        if (i_episode % config["trainer"]["save_freq"] == 0):

            torch.save(agent.qnetwork_local.state_dict(), save_path + 
                config["trainer"]["save_trained_name"] + "_" + str(i_episode) + ".pth")
        
        # Check if environment solved (if not already)
        if not flag:
            if (np.mean(scores_window) >= 13.0):
                logger.info('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(
                i_episode-100, np.mean(scores_window)))        
                # Save solved model
                torch.save(agent.qnetwork_local.state_dict(), save_path + 
                        config["trainer"]["save_trained_name"] + "_solved.pth")
                flag = True
    
    return scores

if __name__ == '__main__':
	# Configure logging for all loggers
	logging.basicConfig(level=logging.INFO, format='')

	# Load config file
	with open("config.json", "r") as f:
   		config = json.load(f)

    # Start the environment
	env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")

    # get the default brain
	brain_name = env.brain_names[0]
	brain = env.brains[brain_name]

	# Create agent
	agent = Agent(state_size=37, action_size=4, config=config)

	# Train the agent 
	scores = dqn(agent=agent, 
		brain_name=brain_name, 
		config=config, 
		n_episodes=config["trainer"]["num_episodes"],
		max_timesteps_per_ep=config["trainer"]["max_timesteps_per_ep"],
		eps_start=config["GLIE"]["eps_start"],
		eps_end=config["GLIE"]["eps_end"],
		eps_decay=config["GLIE"]["eps_decay"]
		)

	# Close the environment 
	env.close()