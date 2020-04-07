#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:32:37 2020

@author: mostafa
"""

from abc import ABC, abstractmethod
from utils.config_manager import ConfigManager

class RLModelInterface(ABC):
    
    def __init__(self, model_name, action_space, reward_range, state_height, state_width):
        self.model_name = model_name
        self.action_space = action_space
        self.reward_range = reward_range
        self.state_height = state_height
        self.state_width = state_width 
        
        self.model_config = ConfigManager('models_config.json').get(self.model_name)
        
    
    @abstractmethod
    def get_action(self, state):
        pass
    
    
    @abstractmethod
    def add_feedback_sample(self, state1, action, reward, state2, done):
        pass
    
    
    @abstractmethod
    def save(self, file_name):
        pass
    
    
    @abstractmethod
    def load(self, file_name):
        pass 
    
    @abstractmethod
    def get_analysis_dataframe(self):
        pass
    
    
    def __str__(self):
        return self.model_name