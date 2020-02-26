# -*- coding: utf-8 -*-
"""
指定一个方框的左下角和右上角经纬度坐标和切分的行数列数
返回切分后的每个小方框中心点的坐标值shape=(row,col)= [lat,lon]

读取h5文件获取shape数据

数据做维度变换做成move_data格式要求的数据
Author: elesun
"""
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import time

loc_downleft = [37.9, 116.35]#latitude,longitude
loc_upright = [39.9, 118.35]
row = 32
col = 32

def area_split(loc_downleft,loc_upright,row,col):
    location = np.zeros(shape=(row,col,2),dtype=float)# 3 mean [lat,lon,val]
    #print("location_init\n",location)
    row_length = (loc_upright[0] - loc_downleft[0])/row
    col_length = (loc_upright[1] - loc_downleft[1])/col
    print("row_length",row_length)
    print("col_length",col_length)
    for i in range(row) :
        for j in range(col) :
            # location[i, j, :] = [37.9, 118.35, 1]
            location[i,j] = [loc_downleft[0]+(i+0.5)*row_length,loc_downleft[1]+(j+0.5)*col_length]
    return (location)

print("#################area split######################")
location = area_split(loc_downleft,loc_upright,row,col)
#print("location\n",location)
print("location[0,0,:] = \n",location[0,0,:])
print("location.shape ",location.shape)

print("#################read small h5######################")
#h5 data shape = (48, 32, 32) = (step, h, w)
frname = "small.h5"
print("reading ",frname)
fr = h5py.File(frname, 'r')
print ("fr.keys() = ",fr.keys())
data = fr['small_data'][()]
fr.close()
print ("data.shape = ",data.shape)
print ("data.dtype = ",data.dtype)
print("max data = ", np.max(data))
print("min data = ", np.min(data))
data = (data-np.min(data))/(np.max(data)-np.min(data))
#print ("data = ",data)
print ("data[0,0,:3] = \n",data[0,0,:3] )

print("#################array reshape######################")
#move_data shape (48, 1024, 3) (step, samples, data)
#data shape 3 = (lat,lon,val)
#转换过程：h5 data(48, 32, 32) to (48,32*32,3),3 =(lat,lon,val)
data = data.reshape(48,32*32,1)
location = location.reshape(32*32,2)
#print("location\n",location)
temp = np.insert(data, 0, location[:,1], axis=2)
move_data = np.insert(temp, 0, location[:,0], axis=2)
print ("move_data.shape ",move_data.shape)
#print ("move_data = \n",move_data)
print ("move_data[0,0:3,:] = \n",move_data[0,0:3,:])