import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        self.eps = 1.0
        self.eps_decay = 0.99
        self.eps_min = 0.005
    
        self.alpha = 0.1
        self.gamma = 0.9
        
    def get_policy(self, Q_s):
        """ Obtain the action probabilities corresponding to epsilon-greedy policies """
        self.eps = max(self.eps*self.eps_decay, self.eps_min) 
        policy_s = np.ones(self.nA) * (self.eps / self.nA)
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - self.eps + (self.eps / self.nA)
        
        return policy_s
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = self.get_policy(self.Q[state])
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        
        return action 
        
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        
        """
        ## Using update rule of Sarsamax (Q-Learning)

        if not done:
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action])
        if done:
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * 0 - self.Q[state][action])
            
        