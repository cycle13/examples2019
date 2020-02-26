# -*- coding: utf-8 -*-
""" 
img_ssim_test
fun:
	get ssim metric 
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3;,opencv-python3.4.4.19;scikit-image 0.14.0
"""
from __future__ import print_function
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

#in_img_path = './gray1.png'
in_img_path = './cat.jpg'
out_img_path1 = './rgb1.png'
out_img_path2 = './rgb2.png'

def main():
	gray_img1 = Image.open(in_img_path)
	gray_img1 = gray_img1.convert('L')#RGB转灰度
	rgb1 = gray_img1.convert('RGB')#灰度转RGB
	rgb1.save(out_img_path1)

	gray_img2 = cv2.imread(in_img_path,cv2.IMREAD_COLOR) #cv2.IMREAD_COLOR  cv2.IMREAD_GRAYSCALE
	gray_img2 = cv2.cvtColor(gray_img2, cv2.COLOR_BGR2GRAY)
	rgb2 = cv2.cvtColor(gray_img2, cv2.COLOR_GRAY2BGR)
	cv2.imwrite(out_img_path2, rgb2)
   

if __name__ == '__main__':
   main()



   
