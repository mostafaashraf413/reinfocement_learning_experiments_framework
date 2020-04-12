#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:47:36 2020

@author: mostafa
"""

from rl_env_interface import RLEnvInterface
import gym

class MountainCar(RLEnvInterface):
    
    def __init__(self):
        super().__init__('mountain_car_v0')
        self.env = gym.make('MountainCar-v0').unwrapped

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
    car = MountainCar()
    
    for i in range(10000):
        car.new_episode()    
        done = False
        total_reward = 0
        while(not done):
            reward, done = car.step(car.get_random_action())
            car.render()
            total_reward += reward
        print(total_reward)
        
    car.close_env()