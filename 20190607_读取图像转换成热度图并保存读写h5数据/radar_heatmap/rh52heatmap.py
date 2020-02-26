# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fwname = "./out/imgs.h5"
heatmap_dir = "heatmap"
print("reading ",fwname)
fr = h5py.File(fwname, 'r')
print ("fr.keys() = ",fr.keys())
# 可以查看所有的主键
for key in fr.keys():
    print(fr[key].name)
    print(fr[key].shape)
    print(fr[key].dtype)
    #print(fr[key].value)
data = fr['RAD_206482464212541'][()]
print ("data.shape = ",data.shape)
print ("data.dtype = ",data.dtype)
print("max data = ", np.max(data))
print("min data = ", np.min(data))
fr.close()

print("#################plt and save h5######################")
#创建流入量图文件路径
if not os.path.exists(heatmap_dir):
    os.mkdir(heatmap_dir)
for i in range(len(data)):#data.shape[0] = 48
    print ("data[%02d,:,:].shape is "%(i),data[i, :, :].shape)
    print("data[%02d,:,:].dtype is " % (i), data[i, :, :].dtype)
    #print("data[%02d,:,:] = \n" % (i), data[i, :, :])#归一化，（0-1），最大值为1

    plt.subplots()
    sns.heatmap(data[ i, :, :],vmin=0,vmax=1)
    #save png
    plt.savefig(heatmap_dir + '/' + '%02d_heatmap.png' % i)
    #plt.show()