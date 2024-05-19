# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:49:00 2024

@author: marce
"""
import numpy as np

def rgb_gray(rgb):
    img = np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    img = (16 - (img*16).astype(int))
    img = img.flatten()
    return img