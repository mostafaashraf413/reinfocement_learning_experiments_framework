#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:33:39 2020

@author: mostafa
"""

from utils.config_manager import ConfigManager
from rl_env_interface import RLEnvInterface
from rl_model_interface import RLModelInterface

class GameManager():
    
    def __init__(self, config_file = 'game_config.json'):
        self.config = ConfigManager(config_file)
        self.get_env()
        self.get_rl_models()
        
    def get_env(self):
        env_name = self.config.get('env_name')
        exec('from rl_envs.%s import %s '%(env_name[0], env_name[1]))
        self.env : RLEnvInterface = eval(env_name[1])()
    
    def get_rl_models(self):
        rl_models_names = self.config.get('rl_models')
        self.rl_models = []
        for i in rl_models_names:
            exec('from rl_models.%s.%s import %s '%(i[0], i[0], i[1]))
            rl_model: RLModelInterface = eval(i[1])(self.env.action_space(), self.env.get_reward_range())
            self.rl_models.append(rl_model)
    
    def run(self):
        print('env name : ', self.env.name)
        print('rl models : ', self.rl_models)
    
    


if __name__ == '__main__':
    game_man = GameManager()
    game_man.run()