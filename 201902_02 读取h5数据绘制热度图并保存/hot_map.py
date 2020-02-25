#coding=utf-8
from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

sys.setrecursionlimit(10000000)#手动设置递归调用深度
# 数据流入量
taxiin_dir = 'taxi_in'
# 数据流出量
taxiout_dir = 'taxi_out'

#读取数据
fname = 'BJ13_M32x32_T30_InOut.h5'
f = h5py.File(fname, 'r')
print ("f.keys() = ",f.keys())
data = f['data'].value
print ("data.shape = ",data.shape)
print ("data.dtype = ",data.dtype)

#创建流入量图文件路径
if not os.path.exists(taxiin_dir):
    os.mkdir(taxiin_dir)
for i in range(100):#10 for test ; data.shape[0] = 4888
    print ("data[%04d,0,:,:].shape is "%(i),data[ i, 0, :, :].shape)# 0 is in
    plt.subplots()
    sns.heatmap(data[i, 0, :, :],vmin=0,vmax=500)
    #save png
    plt.savefig(taxiin_dir + '/' + '%04d_contrast.png' % i)
    #plt.show()

#创建流出量图文件路径
if not os.path.exists(taxiout_dir):
    os.mkdir(taxiout_dir)
for i in range(100):#10 for test ; data.shape[0] = 4888
    print("data[%04d,1,:,:].shape is "%(i),data[ i, 1,:,:].shape)
    plt.subplots()
    sns.heatmap(data[i,1,:,:],vmin=0,vmax=500)#1 is out
    plt.savefig(taxiout_dir + '/' + '%04d_contrast.png' % i)
    #plt.show()

