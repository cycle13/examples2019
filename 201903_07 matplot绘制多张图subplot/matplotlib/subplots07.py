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

# plt.subplot()参数调整
fig,axes = plt.subplots(2,2,figsize=(6,4),
                       sharex=True,
                       sharey=True)
# sharex,sharey：是否共享x，y刻度

for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500),color='b', alpha=0.8) 
# 利用循环来填充数据

plt.subplots_adjust(wspace=0.5, hspace=0.1)
# wspace,hspace：用于控制宽度和高度的百分比，比如subplot之间的间距

plt.show()


