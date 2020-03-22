#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:33:35 2020

@author: mostafa
"""

import json


class ConfigManager:
    def __init__(self, config_file):
        self.config = None
        with open(config_file) as json_data_file:
            self.config = json.load(json_data_file)
        print('configuration file has been loaded:')
        print(self.config)
        
    
    def get(self, key):
        return self.config[key]
    
    def __str__(self):
        return str(self.config)
    
    
    
if __name__ == '__main__':
    config = ConfigManager('game_config.json')
    print('episodes', config.get('episodes'))