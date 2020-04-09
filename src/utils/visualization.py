#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:14:26 2020

@author: mostafa
"""
import matplotlib.pyplot as plt

def plot_line_chart(x = None, y_lst = None, y_names_lst = None, x_label = None, y_label = None, title = None):

    if x == None:
        x = list(range(len(y_lst[0])))
    
    for i, y in enumerate(y_lst):
        plt.plot(x, y, label = y_names_lst[i])
    
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    
    
    
    
if __name__ == '__main__':
    year = [1960, 1970, 1980, 1990, 2000, 2010]
    pop_pakistan = [44.91, 58.09, 78.07, 107.7, 138.5, 170.6]
    pop_india = [449.48, 553.57, 696.783, 870.133, 1000.4, 1309.1]
    
    plot_line_chart(x = None, y_lst = [pop_pakistan, pop_india], y_names_lst = ['pakistan', 'india'], 
                    x_label = 'year', y_label = 'pop', title = 'line chart')