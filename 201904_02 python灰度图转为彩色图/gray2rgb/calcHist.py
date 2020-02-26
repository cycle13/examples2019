# -*- coding: utf-8 -*-
""" 
calcHist.py
fun:
	draw calcHist pictures 
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3,opencv-python3.4.4.19

"""
from __future__ import print_function
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('ball.png', 0)

hist = cv.calcHist([img], [0], None, [256], [0,256])
print('hist :\n',hist)#hist是一个256x1阵列，每个值对应于该图像中的像素值及其对应的像素值。
print('length hist :\n',len(hist))

plt.hist(img.ravel(), 256, [0,256])
plt.show()
