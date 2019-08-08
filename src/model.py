import torch
import torch.nn as nn
import torch.nn.functional #as F
import torch.optim as optim
import torch.nn as nn

import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class FCNetwork(nn.Module):
    """Fully Connected Model."""

    def __init__(self, state_size, output_size,hidden_layers  = [512], seed = 12345, drop_p = 0.0, output_gate=None ):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            output_size (int): Dimension of each output
            seed (int): Random seed
            hidden_layers (array): Hidden number of nodes in each layer
            drop_p (float [0-1]) : Probability of dropping nodes (implementation of dropout)
            output_gate (torch functional) : Output gate function
        """
        super(FCNetwork, self).__init__()
        self.seed = torch.manual_seed(12345)
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # Add the output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        # dropout parameter added in case of concentration on a subset of nodes in NN.
        self.dropout = nn.Dropout(p=drop_p)

        # define the output gate which is any function we wish
        self.output_gate = output_gate



    def forward(self, input):
        """Build a network that maps state -> action values.
        
        Params
        ======
            input (Tensor[torch.Variable]): Input tensor in PyTorch model
        """
        for linear in self.hidden_layers:
            input = F.relu(linear(input))
            input = self.dropout(input)

        output = self.output(input) 
        
        if self.output_gate:           
            output =  self.output_gate(output)
        return output

class ActorCritic(nn.Module):
    """Actor Critic Model."""
    
    def __init__(self, state_size, action_size, hidden_size, device):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of actions
            hidden_size (array): Hidden number of nodes in each layer
            device (PyTorch device) : device
        """
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.device = device

        self.actor_body = FCNetwork(state_size, action_size, hidden_size, output_gate = torch.tanh)
        self.critic_body = FCNetwork(state_size, 1, hidden_size)  
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.to(device)

    def forward(self, obs, action=None):
        """Build a network that maps observation -> selected action, log_prob, value.
        
        Params
        ======
            input (Tensor[torch.Variable]): Input tensor in PyTorch model
        """
        obs = torch.Tensor(obs)
        a = self.actor_body(obs)
        v = self.critic_body(obs)
        
        dist = torch.distributions.Normal(a, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        #return action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), v
        return action, log_prob, v


class PolicyMod(nn.Module):
    """VPG Model."""
    def __init__(self, state_size, action_size, hidden_size, device):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of actions
            hidden_size (array): Hidden number of nodes in each layer
            device (PyTorch device) : device
        """
        super(PolicyMod, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.device = device

        self.actor_body = FCNetwork(state_size, action_size, hidden_size, output_gate = torch.tanh)
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.to(device)

    def forward(self, obs, action=None):
        """Build a network that maps observation -> selected action, log_prob
        
        Params
        ======
            input (Tensor[torch.Variable]): Input tensor in PyTorch model
        """
        obs = torch.Tensor(obs)
        a = self.actor_body(obs)
        
        dist = torch.distributions.Normal(a, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, 0