#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:53:22 2020

@author: mostafa
"""

import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch



class ImagePreprocessing:

    def __init__(self, image_size = (40,40)): 
        self.resize = T.Compose([T.ToPILImage(),
                            T.Resize((image_size[0],image_size[1]), interpolation=Image.CUBIC),
                            T.ToTensor()])


    def preprocess_screen(self, screen):
        screen = screen.transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
          
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0)