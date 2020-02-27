# -*- coding: utf-8 -*-
"""
test
fun:
env:
"""
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cv2
import scipy.misc

matrix = np.load("A_Hour_145_Band_08.npy") # A_Hour_145_Band_08.npy A_Hour_597_Band_08.npy
print("matrix.shape",matrix.shape)

min_val_set = 2500
max_val_set = 4000
max_val = np.max(matrix)
min_val = np.min(matrix)
print("max_val",max_val) # 3980 4007
print("min_val",min_val) # 2689 2718

matrix = np.maximum(matrix,min_val_set) # min_val
matrix = np.minimum(matrix,max_val_set) # max_val

max_val = np.max(matrix)
min_val = np.min(matrix)
print("after trim max_val",max_val)
print("after trim min_val",min_val)

matrix_scale = (max_val_set - matrix)/(max_val_set - min_val_set)
print("matrix_scale",matrix_scale)
max_val = np.max(matrix_scale)
min_val = np.min(matrix_scale)
print("after scale max_val",max_val)
print("after scale min_val",min_val)

# img = Image.fromarray(matrix_scale)
# img.save("your_file.jpeg")
matplotlib.image.imsave('matplot.png', matrix_scale)
cv2.imwrite("opencv.png",matrix_scale*255)
scipy.misc.imsave('scipy.png',matrix_scale)