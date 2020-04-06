#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:23:19 2020

@author: mostafa
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from rl_model_interface import RLModelInterface
import copy



class DQNModule(nn.Module):

    def __init__(self, h, w):
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
        self.head = nn.Linear(linear_input_size, 2) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
    
    



class DQN(RLModelInterface):
    
    def __init__(self, action_space, reward_range, state_height, state_width):
        super().__init__('dqn_model', action_space, reward_range, state_height, state_width)
        
        self.dqn_module = DQNModule(self.state_height, self.state_width, )
        self.dqn_module_target = copy.deepcopy(self.dqn_module)
        
        
    def get_action(self, state):
        return self.dqn_module(state).max(1)[1].view(1, 1).item()
    

    def add_feedback_sample(self, state1, action, reward, state2):
#        print('%s has received a feedback!'%(self.model_name))
        pass
    
    
    def save(self):
        print('%s does not need to be saved!'%(self.model_name))
    

    def load(self):
        print('%s does not need to be loaded!'%(self.model_name))  