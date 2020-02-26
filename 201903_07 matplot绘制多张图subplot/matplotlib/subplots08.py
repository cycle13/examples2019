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

# 子图创建3 - 多系列图，分别绘制

df = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))
df = df.cumsum()

df.plot(style ='--', alpha=0.5, grid=True,figsize=(8,6),
       subplots = True,
       layout=(2, 2))
plt.subplots_adjust(wspace=0.2,hspace=0.2)
# plt.plot()基本图表绘制函数 → subplots，是否分别绘制系列（子图）
# layout：绘制子图矩阵，按顺序填充

plt.show()

