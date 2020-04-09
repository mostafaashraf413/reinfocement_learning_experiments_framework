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
import matplotlib.pyplot as plt


class ImagePreprocessing:

    def __init__(self, image_size = (40,40)): 
        self.resize = T.Compose([T.ToPILImage(),
                            T.Resize((image_size[0],image_size[1]), interpolation=Image.CUBIC),
                            T.ToTensor()])


    def preprocess_screen(self, screen):
        screen = screen.transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
        # screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
          
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0)
        
    
    

# URL: https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
def display(img_lst):
    fig=plt.figure(figsize=(8, 8))
    for i, img in enumerate(img_lst):
        fig.add_subplot(1, len(img_lst), i+1)
        plt.imshow(img)
    plt.show()
    
    
if __name__ == '__main__':   
    display([np.random.randint(10, size=(100,100)), np.random.randint(10, size=(10,10))])