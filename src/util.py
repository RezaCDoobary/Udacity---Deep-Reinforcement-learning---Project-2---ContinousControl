import numpy as np

def get_mean_score(env_, policy, number_of_agents = 20):
    """Retreives the mean score given am environment, policy and number of 
    asynchronous agents.

    Params
    ======
        env_ (Environment): An environment wrapping the unity engine environment
        policy (Pytorch nn.Module): A module for the policy
        number_of_agents (int): Number of  asynchrnous agents in the unity environment.
    """
    states = env_.reset(True)              
    scores = np.zeros(number_of_agents)                         
    while True:
        outcome = policy(states)
        if len(outcome) == 4:
                actions, _, _,_ = outcome
        elif len(outcome) == 3:
                actions,_,_ = outcome
        next_states, rewards, is_done = env_.step(actions.cpu().detach().numpy())                   
        scores += rewards                  
        states = next_states                               
        if np.any(is_done):                                  
            break
    return np.mean(scores)