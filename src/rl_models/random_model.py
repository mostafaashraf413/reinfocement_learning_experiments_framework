#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:55:18 2020

@author: mostafa
"""

from rl_model_interface import RLModelInterface


class RandomModel(RLModelInterface):
    
    def __init__(self, action_space, reward_range, state_width, state_height):
        super().__init__('random_model', action_space, reward_range, state_width, state_height)
        
        
    def get_action(self, state):
        return self.action_space.sample()
    

    def add_feedback_sample(self, state1, action, reward, state2, done):
#        print('%s has received a feedback!'%(self.model_name))
        pass
    
    
    def save(self, file_name):
        pass
    
    
    def load(self, file_name):
        pass   
    
    def get_analysis_dataframe(self):
        return None