# -*- coding: utf-8 -*-
""" 
subplots_matplot_test01
fun:
	draw multi-pictures 
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3
"""
#from __future__ import print_function

import os
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(os.path.join('./images/cat.jpg'))


plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()


gray = img.convert('L')
plt.figure("Image Gray")
# 这里必须加 cmap='gray' ,否则尽管原图像是灰度图（下图1），但是显示的是伪彩色图像（下图2）（如果不加的话）
plt.imshow(gray,cmap='gray')
plt.axis('on')
plt.title('image')
plt.show()

#一个窗口显示多幅图像，要用到subplot
gray = img.convert('L')
r,g,b = img.split()
img_merged = Image.merge('RGB', (r, g, b))

plt.figure(figsize=(10,5)) #设置窗口大小
plt.suptitle('Multi_Image') # 图片名称

plt.subplot(2,3,1), plt.title('image')
plt.imshow(img), plt.axis('off')

plt.subplot(2,3,2), plt.title('gray')
plt.imshow(gray,cmap='gray'), plt.axis('off') #这里显示灰度图要加cmap

plt.subplot(2,3,3), plt.title('img_merged')
plt.imshow(img_merged), plt.axis('off')

plt.subplot(2,3,4), plt.title('r')
plt.imshow(r,cmap='gray'), plt.axis('off')

plt.subplot(2,3,5), plt.title('g')
plt.imshow(g,cmap='gray'), plt.axis('off')

plt.subplot(2,3,6), plt.title('b')
plt.imshow(b,cmap='gray'), plt.axis('off')

plt.show()

