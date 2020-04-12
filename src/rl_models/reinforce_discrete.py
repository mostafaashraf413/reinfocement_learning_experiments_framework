#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:14:51 2020

@author: mostafa

Implemented from:
    - https://www.manning.com/books/deep-reinforcement-learning-in-action?query=reinfor
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl_model_interface import RLModelInterface
from utils.image_preprocessing import display
from utils.visualization import plot_line_chart
import numpy as np


class REINFORCEModule(nn.Module):

    def __init__(self, h, w, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, self.conv1.kernel_size[0], self.conv1.stride[0]), self.conv2.kernel_size[0], self.conv2.stride[0]), self.conv3.kernel_size[0], self.conv3.stride[0])
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, self.conv1.kernel_size[0], self.conv1.stride[0]), self.conv2.kernel_size[0], self.conv2.stride[0]), self.conv3.kernel_size[0], self.conv3.stride[0])
       
        linear_input_size = convw * convh * 32
        # self.hidden1 = nn.Linear(linear_input_size, 512)
        # self.hidden2 = nn.Linear(512, 256)
        # self.hidden3 = nn.Linear(256, 64)
        self.head = nn.Linear(linear_input_size, output_size) 

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.hidden1(x.view(x.size(0), -1)))
        # x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        x = x.view(x.size(0), -1)
        x = F.softmax(self.head(x))
        return x
    
    
    
    
    
class REINFORCE(RLModelInterface):
    
    def __init__(self, action_space, reward_range, state_height, state_width):
        super().__init__('reinforce_model_discrete', action_space, reward_range, state_height, state_width)
        
        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actions = list(range(action_space.n))
        self.reinforce_module = REINFORCEModule(self.state_height, self.state_width, self.action_space.n).to(self.device)
        self.__initialize_model_params()
        self.memory = []
        self.analysis_dic = {'losses':[]}
        
    def __initialize_model_params(self):
        self.batch_size = self.model_config['batch_size'] #200
        self.GAMMA = self.model_config['gamma'] #0.999
        self.optimizer = optim.Adam(self.reinforce_module.parameters()) # optim.RMSprop(self.reinforce_module.parameters()) 
      
        
    def discount_rewards(self, rewards):
        lenr = len(rewards)
        disc_return = torch.pow(self.GAMMA,torch.arange(lenr).float()).to(self.device) * rewards
        disc_return /= disc_return.max()
        return disc_return
    
    
    def loss_fn(self, preds, r):
        return -1 * torch.sum(r * torch.log(preds))
    
    
    def accumulate_rewards(self, rewards):
        rewards = np.add.accumulate(rewards)
        return rewards
    
    def __optimize_model(self):
        
        self.reinforce_module.train()
        
        state1_batch, action_batch, reward_batch = [],[],[]
        for (s1,a,r) in self.memory:
            state1_batch.append(s1)
            action_batch.append(a)
            reward_batch.append(r)
            
        state1_batch = torch.cat(state1_batch).to(self.device)
        action_batch = torch.Tensor(action_batch).to(self.device)
        
        reward_batch = self.accumulate_rewards(reward_batch)
        reward_batch = torch.Tensor(reward_batch).to(self.device)
        reward_batch = reward_batch.flip(dims=(0,))
        
        predictions = self.reinforce_module(state1_batch)
        prob_batch = predictions.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
        
        disc_returns = self.discount_rewards(reward_batch)
        loss = self.loss_fn(prob_batch, disc_returns)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.analysis_dic['losses'].append(loss.item())
        
        # just for debug:
        # self.visualize_training_performance()

        
    def __preprocess_state(self, raw_state):
        state = raw_state[-1]
        for i in reversed(range(0, len(raw_state)-1)):
            state = state - raw_state[i]
        
        # state = state / len(raw_state)
        
        if self.model_config['verbose']:
            display([i[0].transpose(0,2).transpose(0,1) for i in raw_state])
            display([state[0].transpose(0,2).transpose(0,1)])            
        return state
    
    
    def get_action(self, state): 
        state = self.__preprocess_state(state)
        state = state.to(self.device)
        
        self.reinforce_module.eval()
        pred = self.reinforce_module(state)[0]
        action = np.random.choice(self.actions, p=pred.cpu().detach().data.numpy())
        
        return action
        
        
    def add_feedback_sample(self, state1, action, reward, state2, done):
        state1_ = self.__preprocess_state(state1)  
        
        # reward = 1 if reward > 0 else (-1 if reward < 0 else 0) 
        
        self.memory.append((state1_, action, reward))
        
        if done:
            self.__optimize_model()
            self.memory = []
        
   
    def visualize_training_performance(self):
        plot_line_chart(y_lst = [self.analysis_dic['losses']], y_names_lst = ['loss'], x_label = 'iterations', y_label = 'loss', title = 'training loss')
     
    
    def save(self, file_name):
        torch.save(self.reinforce_module.state_dict(), file_name)
    

    def load(self, file_name):
        self.reinforce_module.load_state_dict(torch.load(file_name)) 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        