# -*- coding: utf-8 -*-
"""
指定一个方框的左下角和右上角经纬度坐标和切分的行数列数
返回切分后的每个小方框中心点的坐标值shape=(row,col,3)= [lat,lon,val]
Author: elesun
将loc_downleft loc_upright指定的区域，划分成row*col个小方格
"""
from __future__ import print_function
import os
import numpy as np

loc_downleft = [37.9, 116.35]#latitude,longitude
loc_upright = [39.9, 118.35]
row = 32
col = 32

def area_split(loc_downleft,loc_upright,row,col):
    location = np.zeros(shape=(row,col,3),dtype=float)# 3 mean [lat,lon,val]
    print("location_init\n",location)
    row_length = (loc_upright[0] - loc_downleft[0])/row
    col_length = (loc_upright[1] - loc_downleft[1])/col
    print("row_length",row_length)
    print("col_length",col_length)
    for i in range(row) :
        for j in range(col) :
            # location[i, j, :] = [37.9, 118.35, 1]
            location[i,j,:] = [loc_downleft[0]+(i+0.5)*row_length,loc_downleft[1]+(j+0.5)*col_length,1]
    return (location)

location = area_split(loc_downleft,loc_upright,row,col)
print("location\n",location)




