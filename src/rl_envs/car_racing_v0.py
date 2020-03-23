#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:38:45 2020

@author: mostafa
"""

from rl_env_interface import RLEnvInterface
import gym

class CarRacing(RLEnvInterface):
    
    def __init__(self):
        super().__init__('car_racing_v0')
        self.env = gym.make('CarRacing-v0') 
#        self.env = wrappers.Monitor(self.env, None, video_callable=False ,force=True)

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
    car = CarRacing()
    
    for i in range(10000):
        car.new_episode()    
        done = True
        while(done):
            reward, done = car.step(car.get_random_action())
            car.render()
            time.sleep(1)
        
    car.close_env()