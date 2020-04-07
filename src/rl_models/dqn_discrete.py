#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:23:19 2020

@author: mostafa

Implemented from:
    - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    - https://www.manning.com/books/deep-reinforcement-learning-in-action?query=reinfor
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl_model_interface import RLModelInterface
import copy
from collections import deque
import math
import random



class DQNModule(nn.Module):

    def __init__(self, h, w, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, output_size) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
    
    



class DQN(RLModelInterface):
    
    def __init__(self, action_space, reward_range, state_height, state_width):
        super().__init__('dqn_model_discrete', action_space, reward_range, state_height, state_width)
        
        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dqn_module = DQNModule(self.state_height, self.state_width, self.action_space.n).to(self.device)
        self.dqn_module_target = copy.deepcopy(self.dqn_module).to(self.device)
        self.__initialize_model_params()
        
        
    def __initialize_model_params(self):
        self.sync_freq = self.model_config['sync_freq'] #50
        self.sync_counter = 1
        self.mem_size = self.model_config['mem_size'] #1000
        self.batch_size = self.model_config['batch_size'] #200
        self.GAMMA = self.model_config['gamma'] #0.999
        self.EPS_START = self.model_config['EPS_START']
        self.EPS_END = self.model_config['EPS_END']
        self.EPS_DECAY = self.model_config['EPS_DECAY']
        self.steps_done = 0
        
        self.replay = deque(maxlen= self.mem_size)        
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn_module.parameters())
      
        
    def __optimize_model(self):
        if len(self.replay) > self.batch_size:
            minibatch = random.sample(self.replay, self.batch_size)
            
            self.dqn_module.train()
            self.dqn_module_target.eval()
            
            state1_batch, action_batch, reward_batch, state2_batch, done_batch = [],[],[],[],[]
            for (s1,a,r,s2,d) in minibatch:
                state1_batch.append(s1)
                action_batch.append(a)
                reward_batch.append(r)
                state2_batch.append(s2)
                done_batch.append(d)
                
            state1_batch = torch.cat(state1_batch).to(self.device)
            action_batch = torch.Tensor(action_batch).to(self.device)
            reward_batch = torch.Tensor(reward_batch).to(self.device)
            state2_batch = torch.cat(state2_batch).to(self.device)
            done_batch = torch.Tensor(done_batch).to(self.device)
            
            q1 = self.dqn_module(state1_batch)
            with torch.no_grad():
                q2 = self.dqn_module_target(state2_batch)
                
            y = reward_batch + self.GAMMA * ((1-done_batch) * torch.max(q2,dim=1)[0])
            y_predict = q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = self.loss_fn(y_predict, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.sync_counter % self.sync_freq == 0:
                self.dqn_module_target.load_state_dict(self.dqn_module.state_dict())
                self.sync_counter = 1
            self.sync_counter += 1

        
    
    def get_action(self, state):
        self.__optimize_model()
        state = state.to(self.device)
        
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            self.dqn_module.eval()
            with torch.no_grad():
                return self.dqn_module(state).max(1)[1].view(1, 1).item()
        else:
            return torch.tensor([[random.randrange(2)]], device = self.device, dtype=torch.long).item()
        
          

    def add_feedback_sample(self, state1, action, reward, state2, done):
        self.replay.append((state1, action, reward, state2, done))
        
   
    def get_analysis_dataframe(self):
        pass    
     
    
    def save(self, file_name):
        torch.save(self.dqn_module.state_dict(), file_name)
    

    def load(self, file_name):
        self.dqn_module.load_state_dict(torch.load(file_name))    