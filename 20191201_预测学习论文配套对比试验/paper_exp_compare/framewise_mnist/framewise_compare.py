# -*- coding: utf-8 -*-
""" 
subplots_matplot_test01
fun:
	draw multi-pictures 
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

mse_per_seq1 = [
80,
95,
105,
112,
120,
128,
135,
140,
135,
132
]
mse_per_seq2 = [
70,
88,
95,
105,
110,
120,
125,
128,
125,
122
]
mse_per_seq3 = [
29,
36,
44,
51,
62,
72,
81,
85,
89,
92
]
mse_per_seq4 = [
22,
26,
32,
39,
44,
49,
55,
60,
65,
72
]
mse_per_seq5 = [
22,
25,
33,
38,
43,
48,
51,
57,
62,
70
]

ssim1 = [
0.8,
0.77,
0.73,
0.7,
0.68,
0.67,
0.66,
0.65,
0.63,
0.61
]
ssim2 = [
0.83,
0.79,
0.76,
0.73,
0.7,
0.69,
0.67,
0.65,
0.63,
0.61
]
ssim3 = [
0.940,
0.922,
0.904,
0.888,
0.873,
0.859,
0.846,
0.833,
0.822,
0.811
]
ssim4 = [
0.938,
0.931,
0.919,
0.909,
0.897,
0.887,
0.876,
0.866,
0.856,
0.846
]
ssim5 = [
0.941,
0.938,
0.925,
0.915,
0.908,
0.897,
0.887,
0.877,
0.869,
0.856
]
 
if __name__ == '__main__' :
    #print('ssim6',ssim6)
    #print('mse_per_seq6',mse_per_seq6)
    #for index,val in enumerate(ssim6):
    #   print('index',index)
    #print (ssim6.index)
    fig = plt.figure("expermenets",figsize=(10,4))
    #plt.suptitle("Comparisons of different models on test set.")  # 图片名称
    plt.subplot(1,2,1)
    plt.plot(np.arange(1, 11, 1),mse_per_seq1, 'bx-',
             np.arange(1, 11, 1),mse_per_seq2, 'gs-',
             np.arange(1, 11, 1),mse_per_seq3, 'r*-',
             np.arange(1, 11, 1),mse_per_seq4, 'c^-',
             np.arange(1, 11, 1),mse_per_seq5, 'ko-')
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('(a) Frame-wise MSE')  # 图像题目
    plt.legend(["FC-LSTM","ConvLSTM","VPN","Causal LSTM","Ours"],loc='best')
    plt.grid(False)
    plt.xlabel('time steps')
    plt.ylabel('MSE')
    plt.xticks(np.arange(1, 11, step=1), rotation=0) #xticks>xlim
    plt.yticks(np.arange(0, 141, step=20), rotation=0) #yticks>ylim
    plt.xlim(1,10)
    plt.ylim(0, 140)


    plt.subplot(1,2,2)
    plt.plot(np.arange(1, 11, 1),ssim1, 'bx-',
             np.arange(1, 11, 1),ssim2, 'gs-',
             np.arange(1, 11, 1),ssim3, 'r*-',
             np.arange(1, 11, 1),ssim4, 'c^-',
             np.arange(1, 11, 1),ssim5, 'ko-')
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('(b) Frame-wise SSIM')  # 图像题目
    plt.legend(["FC-LSTM","ConvLSTM","VPN","Causal LSTM","Ours"]) #,loc='lower right'
    plt.xlabel('time steps')
    plt.ylabel('SSIM')
    plt.xticks(np.arange(1, 11, step=1),rotation=0)#设置x轴刻度的表现方式
    plt.yticks(np.arange(0.6, 1.1, step=0.1), rotation=0)
    plt.xlim(1,10) #xticks>xlim
    plt.ylim(0.6, 1)

    plt.savefig("result.jpg")
    plt.show()


