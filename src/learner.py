import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable


#This script contains the learner classes. learner is the base class, whilst ProximalPolicyOptimisation and VanillaPolicyGradientOptimisation
#inherit from it. The learn function associated to these functions is what happens as a result of the agent taking a step throughout the 
#environment.

class learner:
    """Base learner class"""
    def __init__(self, optimiser, number_of_epochs):
        """Initialize parameters and build ase learner.

        Params
        ======
            optimiser (torch.optim): Optimiser.
            number_of_epochs (int): Number of epochs to consider.
        """
        self.optmiser = optimiser
        self.batcher = None
        self.number_of_epochs = number_of_epochs

    def set_batcher(self, batcher):
        """Set batcher"""
        self.batcher = batcher

    def learn(self):
        """Defined by functions which inherit this"""
        pass

class ProximalPolicyOptimisation(learner):
    """Build PPO learner"""
    def __init__(self, network, optimiser, number_of_epochs, ppo_clip = None , gradient_clip = None ):
        learner.__init__(self,optimiser, number_of_epochs)
        """Initialize parameters and build PPO learner.

        Params
        ======
            ppo_clip (float) : the epsilon ppo parameter.
            gradient_clip (float): gradient_clipping used to handle exploding/vanishing gradients.
        """
        self.ppo_clip  = ppo_clip
        self.gradient_clip = gradient_clip
        self.network = network


    def learn(self, states, actions, log_probs_old, returns, advantages):
        """Implements the learning function for PPO

        Params
        ======
            states ([torch.tensor]) : states in many trajectories.
            actions ([torch.tensor]) : actions in many trajectories.
            log_probs_old ([torch.tensor]) : log_probs_old in many trajectories.
            returns ([torch.tensor]) : returns in many trajectories.
            advantages ([torch.tensor]) : advantages in many trajectories.
        """
        for _ in range(self.number_of_epochs):
            #set up batcher
            self.batcher.get_iter()
            while not self.batcher.end():
                #sample from space of data
                sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages = \
                    self.batcher.sample(states, actions, log_probs_old, returns, advantages)

                #detach data so we can compute things
                sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages = \
                    sampled_states.detach(), sampled_actions.detach(), sampled_log_probs_old.detach(), sampled_returns.detach(), sampled_advantages.detach()

                #re-forward pass through network to find log_probs and values
                _, log_probs, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                if self.ppo_clip:
                    obj_clipped = ratio.clamp(1.0 - self.ppo_clip,1.0 + self.ppo_clip) * sampled_advantages
                    policy_loss = -torch.min(obj, obj_clipped).mean(0)
                else:
                    policy_loss = -obj.mean(0)

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.optmiser.zero_grad()
                #perform optimisation of the sum of policy_loss and value_loss - optimising the same parameters.
                (policy_loss + value_loss).backward()
                if self.gradient_clip: nn.utils.clip_grad_norm_(self.network.parameters(),self.gradient_clip)
                self.optmiser.step()
                del policy_loss

    
class VanillaPolicyGradientOptimisation(learner):
    """Implements the learning function for VPG"""
    def __init__(self, optimiser, number_of_epochs, trajectory):
        learner.__init__(self,optimiser, number_of_epochs)
        self.trajectory = trajectory

    def learn(self, states):
        """Implements the learning function for PPO

        Params
        ======
            states ([torch.tensor]) : states in many trajectories.
        """
        #Collect trajectories
        processed_trajectories, states, all_rewards, episode_rewards = self.trajectory.collect(states)
        states_, all_rewards_, episode_rewards_ = states, all_rewards, episode_rewards
        states, actions, log_probs, returns = map(lambda x: torch.cat(x, dim=0), zip(*processed_trajectories))

        #write up a surrogate function
        surrogate = log_probs * returns

        policy_loss = -surrogate.mean()
        #optimise
        self.optmiser.zero_grad()
        policy_loss.backward()
        self.optmiser.step()
        del policy_loss

