#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:33:39 2020

@author: mostafa
"""

from utils.config_manager import ConfigManager
from utils.image_preprocessing import ImagePreprocessing
from utils.visualization import plot_line_chart
from rl_env_interface import RLEnvInterface
from rl_model_interface import RLModelInterface
from collections import deque

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
            
            self.rl_models.append(rl_model)
      
    def visualize_models_performance(self, all_rewards):
        y_names = []
        y_lst = []
        
        for y_name in all_rewards.keys():
            y_names.append(y_name)
            y_lst.append(all_rewards[y_name])
            
        plot_line_chart(y_lst = y_lst, y_names_lst = y_names, x_label = 'episodes', y_label = 'rewards', title = 'rewards graph')
      
    
    
    def run(self):
        print('env name : ', self.env.name)
        print('rl models : ', self.rl_models)
        
        all_rewards = {}
        
        for rl_model in self.rl_models:
            
            model_rewards_lst = []
            model_file_name = '../models/'+rl_model.model_name+'__'+self.env.name+'.model'
            
            if self.config.get('load_saved_models'):
                rl_model.load(model_file_name)
                print('%s has been loaded'%(rl_model.model_name))
            
            print('%s training has been started!'%(rl_model.model_name))
            
            for episode in range(self.config.get('episodes')):
                print('%s started episode number %d'%(rl_model.model_name, episode))
                self.env.new_episode()
                done = False
                total_rewards = 0
                
                frames_bag = deque(maxlen= rl_model.frames_per_state()) 
                frames_bag.append(self.img_preprocess.preprocess_screen(self.env.render()))
                
                while not done:
                    current_state = list(frames_bag)
                    
                    action = rl_model.get_action(current_state)
                    reward, done = self.env.step(action)
                    
                    frames_bag.append(self.img_preprocess.preprocess_screen(self.env.render()))
                    next_state = list(frames_bag)
                    
                    rl_model.add_feedback_sample(current_state, action, reward, next_state, done)
                    total_rewards += reward
                print('%s , episode %d, total rewards = %f'%(rl_model.model_name, episode, total_rewards))
                model_rewards_lst.append(total_rewards)
                
                # save model:
                rl_model.save(model_file_name)
                print('%s has been saved'%(rl_model.model_name))
                
            # TODO: collect models analysis
            rl_model.visualize_training_performance()
            all_rewards[rl_model.model_name] = model_rewards_lst
                
            print('%s training has been finished!'%(rl_model.model_name))
            print('########################################################')
        
        self.env.close_env()
        self.visualize_models_performance(all_rewards = all_rewards)             
    
    


if __name__ == '__main__':
    game_man = GameManager()
    game_man.run()