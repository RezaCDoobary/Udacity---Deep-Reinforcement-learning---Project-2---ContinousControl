import numpy as np
import torch



#This script contains the trajectory classes. trajectories_basic is the base, of which trajectories_returns and trajectories_advantage
#inherit from. trajectories_returns simply processes the direct output of the action selection and the agent stepping to produce
#discounted returns. trajectories_advantage processes the data to give the advatange function via the GAE.

class trajectories_basic:
    """Trajectory collection class"""
    def __init__(self, network, trajectory_length, number_of_agents):
        """Initialize parameters and build trajectory builder.

        Params
        ======
            network (nn.Module): Network used to generator actions, returns etc..
            trajectory_length (int): Length of trajectory.
            number_of_agents (int): The number of asychronous agents.
        """
        self.trajectory_length = trajectory_length
        self.number_of_agents = number_of_agents
        self.all_rewards = np.zeros(self.number_of_agents)
        self.episode_rewards = []
        self.network = network

    def set_env(self, env_):
        """
        Set environment.
        """
        self.env__ = env_

    def set_network(self, network):
        """
        Set network.
        """
        self.network = network

    def _collect_basic(self, states, find_values = True):
        """Collecting the basics provided by the out of a actor policiy, 
        trajectory = array of [states, values, actions, log_probs, rewards, 1 - terminals]
        returns states, array of all rewards, array of all episodic rewards

        Params
        ======
            states ([torch.tensor]): Starting state
            find_Values (boolean): should it output values in the trajectory array (this is the base class for both actor
            critic and vpg - thus we need the choice.) 
        """
        #We set up the trajectory - to filled with [states, values, actions, log_probs, rewards, 1 - terminals] for each state-action
        #pair in an episode.
        trajectory = []
        for _ in range(self.trajectory_length):
            if find_values:

                #one pass through the network
                actions, log_probs, values = self.network(states)
            else:
                actions, log_probs, _ = self.network(states)

            #given the action we take a step in the environment
            next_states, rewards, is_done = self.env__.step(actions.cpu().detach().numpy())

            terminals = np.array(is_done, dtype = int)
            self.all_rewards += rewards 
            
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.all_rewards[i])
                    self.all_rewards[i] = 0

            #append to trajectory
            if find_values:
                trajectory.append([states, values, actions, log_probs, rewards, 1 - terminals])
            else:
                trajectory.append([states, actions, log_probs, rewards, 1 - terminals])
            states = next_states

        if find_values:
            pending_value = self.network(states)[-1]
            trajectory.append([states, pending_value, None, None, None, None])


        return trajectory, states, self.all_rewards, self.episode_rewards

    def collect(self, states):
        """The main collect function = will be overridden by classes inheriting this one.

        Params
        ======
            states ([torch.tensor]): Starting state
        """
        trajectory, states, all_rewards, episode_rewards = self._collect_basic(states)
        return trajectory, states, all_rewards, episode_rewards

class trajectories_returns(trajectories_basic):
    """Trajectory collection for basic returns - used for VPG"""
    def __init__(self, network, trajectory_length, number_of_agents, discount_rate):
        """Initialize parameters and build trajectory builder.

        Params
        ======
            network (nn.Module): Network used to generator actions, returns etc..
            trajectory_length (int): Length of trajectory.
            number_of_agents (int): The number of asychronous agents.
            discount_rate (float) : discount rate for returns.
        """
        trajectories_basic.__init__(self,network, trajectory_length, number_of_agents)
        self.discount_rate = discount_rate

    def _collect_returns(self, trajectory):
        """Processes the trajectories to find the discounted returns.

        Params
        ======
            trajectory (nn.Module): trajectory output of _collect_basic
        """
        processed_trajectory = [None] * (len(trajectory) - 1)
        for i in reversed(range(len(trajectory) - 1)):
            states, actions, log_probs, rewards, terminals = trajectory[i]
            terminals = torch.Tensor(terminals).unsqueeze(1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            actions = torch.Tensor(actions)
            states = torch.Tensor(states)

            if i==len(trajectory) - 2:
                returns = rewards
            returns = rewards + self.discount_rate * terminals * returns

            processed_trajectory[i] = [states, actions, log_probs, returns]
    
        return processed_trajectory

    def collect(self, states):
        """Collects the environment response and processes the data to produce discounted returns.

        Params
        ======
            states ([torch.tensor]): Starting state
        """
        trajectory, states, all_rewards, episode_rewards = self._collect_basic(states, False)
        processed_trajectory = self._collect_returns(trajectory)
        return processed_trajectory, states, all_rewards, episode_rewards


class trajectories_advantage(trajectories_basic):
    """Trajectory collection for advantage returns - used for PPO"""
    def __init__(self, network, trajectory_length, number_of_agents, discount_rate, lambda_):
        """Initialize parameters and build trajectory builder.

        Params
        ======
            network (nn.Module): Network used to generator actions, returns etc..
            trajectory_length (int): Length of trajectory.
            number_of_agents (int): The number of asychronous agents.
            discount_rate (float) : discount rate for returns.
            lambda_ (float [0-1]) : is the GAE hyperparameter
        """
        trajectories_basic.__init__(self,network, trajectory_length, number_of_agents)
        self.discount_rate = discount_rate
        self.lambda_ = lambda_

    def _collect_advantages(self, trajectory):
        """Processes the trajectories to find the advantage function estimated with TD error.

        Params
        ======
            trajectory (nn.Module): trajectory output of _collect_basic
        """
        processed_trajectory = [None] * (len(trajectory) - 1)
        advantages = torch.Tensor(np.zeros((self.number_of_agents, 1)))
        returns =  trajectory[-1][1]
        for i in reversed(range(len(trajectory) - 1)):
            states, value, actions, log_probs, rewards, terminals = trajectory[i]
            terminals = torch.Tensor(terminals).unsqueeze(1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            actions = torch.Tensor(actions)
            states = torch.Tensor(states)
            next_value = trajectory[i + 1][1]
            returns = rewards + self.discount_rate * terminals * returns

            td_error = rewards + self.discount_rate * terminals * next_value - value

            #GAE computation
            advantages = advantages * self.lambda_ * self.discount_rate * terminals + td_error
            processed_trajectory[i] = [states, actions, log_probs, returns, advantages]
    
        return processed_trajectory

    def collect(self, states):
        """Collects the environment response and processes the data to produce advantage functions.

        Params
        ======
            states ([torch.tensor]): Starting state
        """
        trajectory, states, all_rewards, episode_rewards = self._collect_basic(states)
        processed_trajectory = self._collect_advantages(trajectory)
        return processed_trajectory, states, all_rewards, episode_rewards

