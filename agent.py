import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

"""
Contains the definition of the agent that will run in an
environment.
"""

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(2, 32)
        
        self.action_layer = nn.Linear(32, 2)
        self.value_layer = nn.Linear(32, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.actions = []
        

    def forward(self, observation):
        # Convert tuple into tensor
        observation_as_list = []
        observation_as_list.append(observation[0])
        observation_as_list.append(observation[1])
        observation_as_list = np.asarray(observation_as_list)
        observation_as_list = observation_as_list.reshape(1,2)
        observation = observation_as_list
        
        state = torch.from_numpy(observation).float()
        state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        action_parameters = F.tanh(self.action_layer(state))
        action_distribution = Normal(action_parameters[0][0], action_parameters[0][1])
        
        action = action_distribution.sample() # Torch.tensor; action
        
        self.logprobs.append(action_distribution.log_prob(action)+ 1e-6)
        self.state_values.append(state_value)
        return action.item() # Float element
        
        
    
    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)  
            
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]




class RandomAgent():
    def __init__(self):
        """Init a new agent.
        """
        #self.theta = np.zeros((3, 2))
        #self.state = RandomAgent.reset(self,[-20,20])
        
        self.count_episodes = -1
        self.max_position = -0.4
        self.epsilon = 0.9
        self.gamma = 0.99
        self.running_rewards = 0
        self.policy = ActorCritic()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01, betas=(0.9, 0.999))
        self.check_new_episode = 1
        self.count_iter = 0
        
    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        self.epsilon = (self.epsilon * 0.99)
        self.count_episodes += 1
        return (np.random.uniform(x_range[0],x_range[1]), np.random.uniform(-20,20))

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        
#        observation_as_list = []
#        observation_as_list.append(observation[0])
#        observation_as_list.append(observation[1])
#        observation_as_list = np.asarray(observation_as_list)
#        observation_as_list = observation_as_list.reshape(1,2)
#        observation = observation_as_list
        
        
        if np.random.rand(1) < self.epsilon:
            return np.random.uniform(-1,1)
        else:
            action = self.policy(observation)
            return action

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.count_iter +=1
        self.policy.rewards.append(reward)
        self.running_rewards += reward
        if self.count_iter == 100:
            # We want first to update the critic agent:
            self.optimizer.zero_grad()
            self.loss = self.policy.calculateLoss(self.gamma)
            self.loss.backward()
            self.optimizer.step()        
            self.policy.clearMemory()
            
            self.count_iter = 0
        

Agent = RandomAgent



