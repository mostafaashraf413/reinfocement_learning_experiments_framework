#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:55:18 2020

@author: mostafa
"""

from rl_model_interface import RLModelInterface
import random


class RandomModel(RLModelInterface):
    
    def __init__(self, action_space, reward_range):
        super().__init__('random_model', action_space, reward_range)
        
        
    def get_action(self, state):
        random.random.randint(0, self.action_space-1)
    

    def add_feedback_sample(self, state1, action, reward, state2):
        print('%s has received a feedback!'%(self.model_name))
    
    
    def save(self):
        print('%s does not need to be saved!'%(self.model_name))
    

    def load(self):
        print('%s does not need to be loaded!'%(self.model_name))  