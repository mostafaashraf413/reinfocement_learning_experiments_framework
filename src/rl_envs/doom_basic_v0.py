#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:44:32 2020

@author: mostafa
"""

from rl_env_interface import RLEnvInterface
import gym
import gym_pull

class DoomBasic(RLEnvInterface):
    
    def __init__(self):
        super().__init__('doom_basic_v0')
        gym_pull.pull('github.com/ppaquette/gym-doom')
        self.env = gym.make('ppaquette/DoomCorridor-v0') 

    def new_episode(self):
        self.env.reset()
    
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return reward, done
    
    
    def render(self):
        return self.env.render(mode='rgb_array')
    
    
    def close_env(self):
        self.env.close()
    

    def action_space(self):
        return self.env.action_space   
    

    def get_random_action(self):
        return self.env.action_space.sample() 
    
    
    def get_reward_range(self):
        return self.env.reward_range
    
    
if __name__ == '__main__':
    import time
    doom = DoomBasic()
    
    for i in range(10000):
        doom.new_episode()    
        done = True
        while(done):
            reward, done = doom.step(doom.get_random_action())
            doom.render()
            time.sleep(1)
        
    doom.close_env()