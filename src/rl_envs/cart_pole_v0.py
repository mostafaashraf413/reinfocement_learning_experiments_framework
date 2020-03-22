#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:32:15 2020

@author: mostafa
"""


from rl_env_interface import RLEnvInterface
import gym
from gym import wrappers

class CartPole(RLEnvInterface):
    
    def __init__(self):
        super().__init__('cart_pole_v0')
        self.env = gym.make('CartPole-v0')
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
        return self.env.action_space.n   
    

    def get_random_action(self):
        return self.env.action_space.sample() 
    
    
    def get_reward_range(self):
        return self.env.reward_range
    
    
if __name__ == '__main__':
    cart = CartPole()
    
    for i in range(10000):
        cart.new_episode()    
        done = True
        while(done):
            reward, done = cart.step(cart.get_random_action())
            cart.render()
        
    cart.close_env()