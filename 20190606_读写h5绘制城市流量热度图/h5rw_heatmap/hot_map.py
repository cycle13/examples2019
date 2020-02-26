#coding=utf-8
from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import time

sys.setrecursionlimit(10000)#手动设置递归调用深度
# 数据流入量
taxiin_dir = 'taxi_in'

#读取数据
frname = "BJ13_M32x32_T30_InOut.h5"
fwname = "small.h5"
print("#################read and write h5######################")
print("reading ",frname)
fr = h5py.File(frname, 'r')
fw = h5py.File(fwname, 'w')
print ("fr.keys() = ",fr.keys())
data = fr['data'].value
print ("data.shape = ",data.shape)
print ("data.dtype = ",data.dtype)
fw['small_data'] = data[0:48,0,:,:]
fr.close()
fw.close()
print("save ",fwname)
print("#################read small h5######################")
time.sleep(5)
print("reading ",fwname)
fr = h5py.File(fwname, 'r')
print ("fr.keys() = ",fr.keys())
data = fr['small_data'].value
print ("data.shape = ",data.shape)
print ("data.dtype = ",data.dtype)
print("max data = ", np.max(data))
print("min data = ", np.min(data))
fr.close()

print("#################plt and save h5######################")
#创建流入量图文件路径
if not os.path.exists(taxiin_dir):
    os.mkdir(taxiin_dir)
for i in range(len(data)):#data.shape[0] = 48
    print ("data[%02d,:,:].shape is "%(i),data[i, :, :].shape)
    print("data[%02d,:,:].dtype is " % (i), data[i, :, :].dtype)
    print("data[%02d,:,:] = \n" % (i), data[i, :, :]/np.max(data))#归一化，（0-1），最大值为1

    plt.subplots()
    sns.heatmap(data[ i, :, :],vmin=0,vmax=np.max(data))
    #save png
    plt.savefig(taxiin_dir + '/' + '%02d_contrast.png' % i)
    #plt.show()


