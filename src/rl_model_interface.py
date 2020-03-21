#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:32:37 2020

@author: mostafa
"""

from abc import ABC, abstractmethod

class RLModelInterface(ABC):
    
    def __init__(self, model_name, action_space, reward_range):
        self.model_name = model_name
        self.action_space = action_space
        self.reward_range = reward_range
        
    
    @abstractmethod
    def get_action(state):
        pass
    
    
    @abstractmethod
    def add_feedback_sample(state1, action, reward, state2):
        pass