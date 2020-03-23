#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:17:27 2020

@author: mostafa
"""

from rl_env_interface import RLEnvInterface
import gym_super_mario_bros as gym

class SuperMario(RLEnvInterface):
    
    def __init__(self):
        super().__init__('super_mario_basic_v0')
        self.env = gym.make('SuperMarioBros-v0') 

    def new_episode(self):
        self.env.reset()
    
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return reward, done
    
    
    def render(self):
        return self.env.render()
    
    
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
    mario = SuperMario()
    
    for i in range(10000):
        mario.new_episode()    
        done = True
        while(done):
            reward, done = mario.step(mario.get_random_action())
            mario.render()
            time.sleep(1)
            print(done)
        
    mario.close_env()