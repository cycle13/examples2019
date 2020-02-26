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

import matplotlib.pyplot as plt
import numpy as np

#创建一个画中画
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
#plt.show()

#面向对象的画图接口中类似的命令由fig.add_axes()，下面用这个命令创建两个竖直排列的坐标轴
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], ylim=(-1.2, 1.2))
x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))
plt.show()
