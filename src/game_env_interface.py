#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:31:52 2020

@author: mostafa
"""


from abc import ABC, abstractmethod

class GameEnvInterface(ABC):

    def __init__(self, name):
        self.name = name


    @abstractmethod
    def new_episode(self):
        pass
    
    
    @abstractmethod
    def step(self, action):
        """
        input: new action
        output: reward, observation, is_done
        """
        pass
    
    
    @abstractmethod
    def render(self):
        pass
    
    
    @abstractmethod
    def close_env(self):
        pass
    

    @abstractmethod
    def action_space(self):
        pass    
    

    @abstractmethod
    def get_random_action(self):
        pass 
    
    
    @abstractmethod
    def get_reward_range(self):
        pass