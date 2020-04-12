#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:01:17 2020

@author: mostafa
"""

from rl_env_interface import RLEnvInterface
import gym


class Boxing(RLEnvInterface):
    
    def __init__(self):
        super().__init__('Boxing-v0')
        self.env = gym.make('Boxing-v0')
        self.current_observation = self.env.observation_space.sample()
        

    def new_episode(self):
        self.env.reset()
    
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.current_observation = observation
        return reward, done
    
    
    def render(self):
        self.env.render(mode = 'human') #(mode = 'rgb_array')
        # return self.current_observation
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
    import matplotlib.pyplot as plt
    
    box = Boxing()
    
    print(box.action_space())
    
    for i in range(10000):
        box.new_episode()    
        done = False
        
        while(not done):
            action = box.get_random_action()
            reward, done = box.step(action)
            print(reward, ' ', action, ' ', done)
            box.render()
            if reward != 0:
                print('wooooooooooooooooow')
            # print(enduro.render())
            # plt.imshow(pong.render())
            # plt.show()
        
    box.close_env()