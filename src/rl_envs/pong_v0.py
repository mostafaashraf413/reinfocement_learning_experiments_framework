#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:49:24 2020

@author: mostafa
"""

from rl_env_interface import RLEnvInterface
import gym

class Pong(RLEnvInterface):
    
    def __init__(self):
        super().__init__('pong_v0')
        self.env = gym.make('Pong-v0')
        self.current_observation = self.env.observation_space.sample()
        

    def new_episode(self):
        self.env.reset()
    
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.current_observation = observation
        return reward, done
    
    
    def render(self):
        self.env.render(mode = 'human') #(mode = 'rgb_array')
        return self.current_observation
    
    
    def close_env(self):
        self.env.close()
    

    def action_space(self):
        return self.env.action_space  
    

    def get_random_action(self):
        return self.env.action_space.sample() 
    
    
    def get_reward_range(self):
        return self.env.reward_range
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    pong = Pong()
    print(pong.action_space())
    
    for i in range(10000):
        pong.new_episode()    
        done = False
        
        while(not done):
            action = pong.get_random_action()
            reward, done = pong.step(action)
            print(reward, ' ', action, ' ', done)
            
            pong.render()
            # plt.imshow(pong.render())
            # plt.show()
        
    pong.close_env()