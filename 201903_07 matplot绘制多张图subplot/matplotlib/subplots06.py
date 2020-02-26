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

# 子图创建1 - 先建立子图然后一次填充其他子图


fig = plt.figure(figsize=(10,6), facecolor = 'gray')

ax1 = fig.add_subplot(2,2,1)  # 创建一个图表，该图在2行2列的子图中的第一个位置，即一行左图
plt.plot(np.random.rand(50).cumsum(),'--g')

ax4 = fig.add_subplot(2,2,4)    #创建一个图表，位于2行2列，即二行右图
ax4.hist(np.random.rand(50).cumsum(),alpha=0.5,color='b')
# 先创建图表figure，然后生成子图，(2,2,1)代表创建2*2的矩阵表格，然后选择第一个，顺序是从左到右从上到下
# 创建子图后绘制图表，绘制到最后一个子图

ax2 = fig.add_subplot(2,2,2)
df2 = pd.DataFrame(np.random.rand(10,4), columns=['a','b','c','d'])
ax2.plot(df2, linestyle='--',marker='.')
# 也可以直接在子图后用图表创建函数直接生成图表

plt.show()


# 子图创建2 - 创建一个新的figure，并返回一个subplot对象的numpy数组 → plt.subplot

fig,axes = plt.subplots(2,3,figsize=(10,4))
print(type(fig), '\n', type(axes)) 

ax1=axes[0, 1]
ax1.plot(np.random.rand(100))

ax2 = axes[1,2]
ax2.plot(np.random.randint(10,size=10).cumsum(),'--')

plt.show()

