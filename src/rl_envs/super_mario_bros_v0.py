#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:17:27 2020

@author: mostafa
"""

from rl_env_interface import RLEnvInterface
import gym_super_mario_bros as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# URL: https://pypi.org/project/gym-super-mario-bros/
class SuperMario(RLEnvInterface):
    
    def __init__(self):
        super().__init__('super_mario_basic_v0')
        self.env = gym.make('SuperMarioBros-v0') 
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.current_state = None

    def new_episode(self):
        self.env.reset()
    
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.current_state = observation
        
        # TODO: render option must be cofigured
        self.env.render()
        
        return reward, done
    
    
    def render(self):
        if self.current_state is None:
            return self.env.observation_space.sample()
        return self.current_state
        # return self.env.render('rgb_array')
    
    
    def close_env(self):
        self.env.close()
    

    def action_space(self):
        return self.env.action_space   
    

    def get_random_action(self):
        action = self.env.action_space.sample() 
        return action
    
    
    def get_reward_range(self):
        return self.env.reward_range
    
    
if __name__ == '__main__':
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render('rgb_array')
        print(state)
    
    env.close()