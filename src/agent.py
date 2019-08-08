import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from batcher import batch_generator
from trajectories import trajectories_advantage
from learner import ProximalPolicyOptimisation


#This is the agent script. It essentially sets up the agent and with every step it takes learn according to rules defined by the agent
#itself. For both agents the step generates trajectories and learns from them.

class PPOAgent:
    """Implements the PPO agent"""
    def __init__(self, env, trajectory_class, learner, number_of_agent = 20, batch_number = 32):
        """Initialize parameters and build PPO agent.

        Params
        ======
            env (Environment) : Environtment class wrapping Unity.
            trajectory_class (trajectory class) : An instance of the trajectory class
            learner (learner class) : An instace of the learner class
            number_of_agents (int) : number of agents
            batch_number (int) : batch_size
        """
        self.total_steps = 0
        self.number_of_agent = number_of_agent
        self.all_rewards = np.zeros(number_of_agent)
        self.episode_rewards = []
        self.env = env
        self.trajectory_class = trajectory_class
        self.trajectory_class.set_env(self.env)
        self.learner = learner
        self.states = self.env.reset(True)  
        self.batch_number = batch_number
        
    def step(self):
        """The agent takes a step."""

        #Set up environment reset
        self.states = self.env.reset(True)
        states = self.states

        #generate trajectories
        processed_trajectory, states, all_rewards, episode_rewards = self.trajectory_class.collect(states)

        self.states = states
        self.all_rewards = all_rewards
        self.episode_rewards = episode_rewards
 
        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_trajectory))

        #normalise advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        #define the batcher
        batcher = batch_generator(states.size(0) // self.batch_number,np.arange(states.size(0)))

        #learn
        self.learner.set_batcher(batcher)

        self.learner.learn(states, actions, log_probs_old, returns, advantages)
        
class VPGAgent:
    """Implements the VPG agent"""
    def __init__(self, env, learner, number_of_agent = 20):
        """Initialize parameters and build VPA agent.

        Params
        ======
            env (Environment) : Environtment class wrapping Unity.
            learner (learner class) : An instace of the learner class
            number_of_agents (int) : number of agents
        """
        self.total_steps = 0
        self.number_of_agent = number_of_agent
        self.all_rewards = np.zeros(number_of_agent)
        self.episode_rewards = []
        self.env = env
        self.learner = learner
        self.states = self.env.reset(True)  
        self.learner.trajectory.set_env(self.env)
        
    def step(self):
        self.states = self.env.reset(True)
        states = self.states

        self.learner.learn(states)
