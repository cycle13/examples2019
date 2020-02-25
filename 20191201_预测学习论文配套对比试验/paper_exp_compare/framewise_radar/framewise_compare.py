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

psnr_per_seq1 = [
15.33,
13.73,
12.8,
12.19,
11.79,
11.47,
11.19,
11,
10.92,
10.87
]
psnr_per_seq2 = [
16.74,
15.58,
14.91,
14.41,
13.94,
13.57,
13.27,
13.06,
12.95,
12.84
]
psnr_per_seq3 = [
17.55,
16.86,
16.47,
16.23,
15.94,
15.63,
15.45,
15.3,
15.17,
15.02
]
psnr_per_seq4 = [
18.7,
18.18,
17.85,
17.64,
17.41,
16.92,
16.79,
16.68,
16.62,
16.58
]
psnr_per_seq5 = [
18.77,
18.22,
17.91,
17.68,
17.43,
17.22,
17.04,
16.89,
16.82,
16.77
]

ssim1 = [
0.669,
0.648,
0.636,
0.628,
0.621,
0.615,
0.608,
0.605,
0.603,
0.603
]
ssim2 = [
0.752,
0.721,
0.705,
0.695,
0.689,
0.682,
0.677,
0.674,
0.67,
0.668
]
ssim3 = [
0.781,
0.765,
0.756,
0.75,
0.744,
0.738,
0.735,
0.733,
0.73,
0.729
]
ssim4 = [
0.819,
0.807,
0.799,
0.795,
0.789,
0.778,
0.776,
0.774,
0.772,
0.77
]
ssim5 = [
0.82,
0.81,
0.8,
0.799,
0.792,
0.787,
0.785,
0.785,
0.783,
0.784
]
 
if __name__ == '__main__' :
    #print('ssim6',ssim6)
    #print('psnr_per_seq6',psnr_per_seq6)
    #for index,val in enumerate(ssim6):
    #   print('index',index)
    #print (ssim6.index)
    fig = plt.figure("expermenets",figsize=(10,4))
    #plt.suptitle("Comparisons of different models on test set.")  # 图片名称
    plt.subplot(1,2,1)
    plt.plot(np.arange(1, 11, 1),psnr_per_seq1, 'bx-',
             np.arange(1, 11, 1),psnr_per_seq2, 'gs-',
             np.arange(1, 11, 1),psnr_per_seq3, 'r*-',
             np.arange(1, 11, 1),psnr_per_seq4, 'c^-',
             np.arange(1, 11, 1),psnr_per_seq5, 'ko-')
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('(a) Frame-wise PSNR')  # 图像题目
    plt.legend(["ConvLSTM","TrajGRU","ST-LSTM","Causal LSTM","Ours"],loc='best')
    plt.grid(False)
    plt.xlabel('time steps')
    plt.ylabel('PSNR')
    plt.xticks(np.arange(1, 11, step=1), rotation=0) #xticks>xlim
    plt.yticks(np.arange(10, 26, step=5), rotation=0) #yticks>ylim
    plt.xlim(1,10)
    plt.ylim(10, 25)


    plt.subplot(1,2,2)
    plt.plot(np.arange(1, 11, 1),ssim1, 'bx-',
             np.arange(1, 11, 1),ssim2, 'gs-',
             np.arange(1, 11, 1),ssim3, 'r*-',
             np.arange(1, 11, 1),ssim4, 'c^-',
             np.arange(1, 11, 1),ssim5, 'ko-')
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('(b) Frame-wise SSIM')  # 图像题目
    plt.legend(["ConvLSTM","TrajGRU","ST-LSTM","Causal LSTM","Ours"]) #,loc='lower right'
    plt.xlabel('time steps')
    plt.ylabel('SSIM')
    plt.xticks(np.arange(1, 11, step=1),rotation=0)#设置x轴刻度的表现方式
    plt.yticks(np.arange(0.6, 1.1, step=0.1), rotation=0)
    plt.xlim(1,10) #xticks>xlim
    plt.ylim(0.6, 1)

    plt.savefig("result.jpg")
    plt.show()


