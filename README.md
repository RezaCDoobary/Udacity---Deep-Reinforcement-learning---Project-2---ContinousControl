# Udacity Reinforcement learning Nanodegree project 2 - Continous Control

## Introduction
This is the second project of the reinforcement learning nanodegree offered by udacity.

Broadly speaking the goal of the project is to train an agent which is a robotic arm to to follow a target. We are given the option to use a single agent or 20 asynchronous agents in order to maximise the number of scenarios available, here the second choice is taken.


## Project Description
Being more precise, the task of the project is the train a double jointed robotic arm agent to maintain contact with target location.

* The material goal of the task is reflected on the **reward** function by giving the agent +0.1 for every step the agent maintains contact with the target location, with 0 otherwise. This is the representation of the environments reaction to performing an action.

* The **state space** of the agent is 33 dimensional and contains the position, rotation, velocity, and angular velocities of the arm.

* The space of **actions** of the agent can take is 4 dimensional continous vector and corresponds to the torque applied on the two joints.

* The task is deemed solved if the agent gets an average score of +30 over 100 consecutive episodes.

## Setup
* A complete set-up of python, the machine learning agents, openAI and much more can be found in https://github.com/udacity/deep-reinforcement-learning#dependencies. Of particular relevance will be unityagents from within the python folder in the corresponding repository.

* The reacher application for the 20 agents is downloaded from the following locations:
    * win x64 : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip
    * win x32 : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip
    * Linux : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
    * Mac : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip

However note that we single agents environments can also be downloaded from https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started, although they were not used in this project.

* We employ the use of environment variables so as to not distribute personal computer information. As a result, please set 

    * \_DRL_LOCATION_ : The path of the python subfolder in the deep reinforcement learning rep descirbed above, generically this would be "...\drlnd\deep-reinforcement-learning\python"
    * \_REACHER_LOCATION_ : The path to the banana executable, this generically would look like "...\Reacher_Windows_x86_64\Reacher.exe"

## Code and result structure
There are three components to the solution. The first is the source code itself which implements the agent, the underlying model and further nessecary componenents. The second is the results folder, which contains the results of the training of the various models studied. Finally, the jupyter notebook named continousControl.ipynb which acts as interface between the source code and the results folder, whilst itself displaying the results.

The detailed rundown is as follows:
* **The source code** can be found in the folder `\src` and include four python files:
    * `agent.py` : Contains the complete agent implementation subject the underlying model chosen.
    * `environment.py` : Is a very rudimentary wrapper for the environment to make it feel a little more like the OpenAI environments.
    * `model.py` :  The model implementations - in this case a neural network. A skeleton fully connected model together with a Actor Critic and a basic vanilla policy gradient model are implemented here.
    * `learner.py` : Contains the implementations for learning classes. Since various learning optomisations employ slightly different learning prescriptions, it makes sense to employ the use of the class structures here.
    * `trajectories.py` : Is the class responsible for the collection of all necessary episodes. Both the implementation required for an advantage function and basic returns are provided.
    * `batcher.py` : Part of the learning method chosen is stochastic sampling - thus a batcher class is created to handle this.
    * `util.py` : A script containing functions that are useful but do not fit elsewhere - currently only contains the `play' method which plays the agent with the winning policy.    

* **The results folder** contains the two subfolders, VPG and PPO which stand for the two models we investigate (vanilla policy gradient and proximity policy optimisation). Within these folders is a folder denoting the date of the run, and 'runs' denoting the run for that day. So for example the relevant subfolder might be for example : 'results/PPO/20190727/run1/'. Suc folder contain the following:

    * The checkpoints (`checkpoint_solved.pth`) upon succesfully solving the task. Checkpoints such as `checkpoint_not_solved.pth` may also be present since and optimisation can be picked up from these files.
    * The corresponding scores against episodes plots.
    * A data.json file which containts all hyperparameter details for the run.
N.B. Model details will be fleshed out in the report.pdf.

* **The interfacing jupyter notebook - continousControl.ipynb** is considered the interface layer in which the user can decide on what precise arch
itectures and models to use for the model. With each trained model, the scores against episodes is plotted, with the results forwared to the relevant results subfolder. The modelled agent can also be played from here to the see the solved task at work. 



## Models
* There are two main optimisation methodologies employed in this project. First is the simple vanilla gradient policy. The second is the proximity policy optimisation with an actor/critic architecture together with importance sampling.

Below is the result of the PPO algorithm described above.
![](play.gif)
