#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:33:39 2020

@author: mostafa
"""

from utils.config_manager import ConfigManager
from utils.image_preprocessing import ImagePreprocessing
from rl_env_interface import RLEnvInterface
from rl_model_interface import RLModelInterface


class GameManager():
    
    def __init__(self, config_file = 'game_config.json'):
        self.config = ConfigManager(config_file)
        self.get_env()        
        self.img_preprocess = ImagePreprocessing(image_size = (self.config.get('state_image_height'), self.config.get('state_image_width')))        
        self.get_rl_models(self.config.get('state_image_height'), self.config.get('state_image_width'))
        
        
    def get_env(self):
        env_name = self.config.get('env_name')
        exec('from rl_envs.%s import %s '%(env_name[0], env_name[1]))
        self.env : RLEnvInterface = eval(env_name[1])()
    
    
    def get_rl_models(self, state_height, state_width):
        rl_models_names = self.config.get('rl_models')
        self.rl_models = []
        for i in rl_models_names:
            exec('from rl_models.%s import %s '%(i[0], i[1]))
            rl_model: RLModelInterface = eval(i[1])(self.env.action_space(), self.env.get_reward_range(), 
                                                    state_height = state_height, state_width = state_width)
            
            if self.config.get('load_saved_models'):
                rl_model.load()
            
            self.rl_models.append(rl_model)
            
    
    def run(self):
        print('env name : ', self.env.name)
        print('rl models : ', self.rl_models)
        
        for rl_model in self.rl_models:
            print('%s training has been started!'%(rl_model.model_name))
            
            for episode in range(self.config.get('episodes')):
                print('%s started episode number %d'%(rl_model.model_name, episode))
                self.env.new_episode()
                done = False
                total_rewards = 0
                
                while not done:
                    current_state = self.img_preprocess.preprocess_screen(self.env.render())
                    action = rl_model.get_action(current_state)
                    reward, done = self.env.step(action)
                    next_state = self.img_preprocess.preprocess_screen(self.env.render())
                    rl_model.add_feedback_sample(current_state, action, reward, next_state)
                    total_rewards += reward
                print('%s , episode %d, total rewards = %f'%(rl_model.model_name, episode, total_rewards))
                    
            print('%s training has been finished!'%(rl_model.model_name))
            print('########################################################')
        self.env.close_env()
                    
    
    


if __name__ == '__main__':
    game_man = GameManager()
    game_man.run()