# Project : Navigation

## Description 
For this project, we train an agent to navigate and collect bananas in a large, 
square world.

## Problem statement 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided 
for collecting a blue banana. Thus, the goal of the agent is to collect 
as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions, and contains the agent's velocity, along
with ray-based perception of objects around the agent's forward
direction. Given this information, the agent has to learn how to best select 
actions. 
Four discrete actions are available, corresponding to: 
- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right
The task is episodic, and in order to solve the environment, the 
agent must get an average score of +13 over 100 consecutive episodes.

## Files
- `Navigation.ipynb`: Notebook used to control and train the agent 
- `dqn_agent.py`: Create an Agent class that interacts with and learns from the environment 
- `model.py`: Q-network class used to map state to action values 
- `config.json`: Configuration file to store variables and paths
- `utils.py`: Helper functions 

## Dependencies
To be able to run this code, you will need an environment with Python 3 and 
the dependencies are listed in the `requirements.txt` file so that you can install them
using the following command: 
```
pip install requirements.txt
``` 

Furthermore, you need to download the environment from one of the links below. You need only to select
the environment that matches your operating system:
- Linux : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- MAC OSX : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Running
Run the cells in the notebook `Navigation.ipynb` to train an agent that solves our required
task of collecting bananas.