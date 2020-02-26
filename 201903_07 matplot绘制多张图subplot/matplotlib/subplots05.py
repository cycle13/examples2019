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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import matplotlib
matplotlib.style.use('bmh')

# plt.figure()  绘图对象
# plt.figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True, 
# FigureClass=<class 'matplotlib.figure.Figure'>, clear=False, **kwargs)


fig1 = plt.figure(num=1,figsize=(4, 2))  # 创建一个图像对象
plt.plot(np.random.rand(50).cumsum(),'-.k')    # 图像会显示在上面离它最近的图像对象中
fig2 = plt.figure(num=2,figsize=(4,2))
plt.plot(np.random.randint(1,11,size=5).cumprod(),'-.g')
# num：图表序号，可以试试不写或都为同一个数字的情况，图表如何显示
# figsize：图表大小
# 当我们调用plot时，如果设置plt.figure()，
# 则会自动调用figure()生成一个figure, 严格的讲，是生成subplots(111)

plt.show()

