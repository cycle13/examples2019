# -*- coding: utf-8 -*-
"""
haikou traffice
fun:
	numpy数组变换海口流量数据并用folium生成热度图map
env:
	Win7 64bit anaconda;python 3.5;tensorflow1.10.1;Keras2.2.4
	pip3,matplotlib2.2.3,seaborn0.9.0
"""
from __future__ import print_function
import os
import sys
import numpy as np
import shutil
import time
import seaborn as sns
import numpy as np
import pandas as pd
import folium
import folium.plugins as plugins
import webbrowser
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import rcParams
import re
import h5py
import time
import webbrowser

#sys.setrecursionlimit(10000000)#手动设置递归调用深度

def area_split(loc_downleft,loc_upright,row,col):
    location = np.zeros(shape=(row,col,2),dtype=float)# 3 mean [lat,lon,val]
    #print("location_init\n",location)
    row_length = (loc_upright[0] - loc_downleft[0])/row
    col_length = (loc_upright[1] - loc_downleft[1])/col
    #print("row_length",row_length)
    #print("col_length",col_length)
    for i in range(row) :
        for j in range(col) :
            # location[i, j, :] = [37.9, 118.35, 1]
            location[i,j] = [loc_upright[0]-(i+0.5)*row_length,loc_downleft[1]+(j+0.5)*col_length]
    return (location)

def gen_data_fr_h5(in_dir,out_dir):
    loc_downleft = [20.08, 110.1]  # latitude,longitude
    loc_upright = [19.9, 110.5]
    row = 50
    col = 75
    # print("#################area split######################")
    location = area_split(loc_downleft, loc_upright, row, col)
    # print("location\n",location)
    # print("location[0,0,:] = \n",location[0,0,:])
    print("location.shape ", location.shape)

    fr = h5py.File(in_dir, 'r')
    print("reading ", in_dir)
    # print("fr.keys() = ", fr.keys()) # python2
    print("fr.keys() = ", [key for key in fr.keys()])  # python3
    key_list = [key for key in fr.keys()]
    print('num of key_list =', len(key_list))
    # key_list.sort()
    for key in key_list:
        print("------------------", key, "------------------")
        h5_data_2chan = fr[key][()]  # fr[key].value  dataset.value has been deprecated.
        print("h5_data_2chan.shape = ", h5_data_2chan.shape)  # (24, 2, 75, 50)
        #h5_data_2chan = h5_data_2chan.transpose((0, 1, 3, 2))
        #h5_data_2chan = h5_data_2chan.swapaxes(2, 3)
        #print("h5_data_2chan.shape after swap", h5_data_2chan.shape)  # (24, 2, 75, 50)

        # channel start
        h5_data = h5_data_2chan[:,0,:,:]# channel start
        h5_data = np.maximum(h5_data, 0)  # 逐位比较取较大者
        h5_data = np.minimum(h5_data, 255)  # 逐位比较取较小者
        print("h5_data.dtype = ", h5_data.dtype) # int32
        print("max h5_data = ", np.max(h5_data)) # max data =  415
        #print("mean h5_data = ", np.mean(h5_data))  # mean data =
        print("min h5_data = ", np.min(h5_data)) # min data =  0
        #h5_data = (h5_data - float(np.min(h5_data))) / (np.max(h5_data) - np.min(h5_data))# 归一化，（0-1），最大值为1
        h5_data = h5_data / 256.0
        #print("h5_data\n", h5_data)
        print("h5_data.dtype = ", h5_data.dtype)  # float64
        print("max h5_data after scale", np.max(h5_data))  # max data =  1
        #print("mean h5_data after scale", np.mean(h5_data))  # mean data =
        print("min h5_data after scale", np.min(h5_data))  # min data =  0
        #print("h5_data\n", h5_data)
        print("np.argwhere(h5_data == 1.0):", np.argwhere(h5_data == 1.0))
        # print("#################array reshape######################")
        # 转换过程：h5 data(24, 75, 50) to move_data(24,75*50,3)
        # move_data shape (24, 75*50) = (step, samples, data)
        # move_data shape 3 = (lat,lon,val)
        data = h5_data.reshape((h5_data.shape[0], col * row, 1), order='F')#最后一维表示value
        location = location.reshape((col * row, 2), order='C')
        #print("location\n",location)
        temp = np.insert(data, 0, location[:, 1], axis=2)#插入经度
        move_data = np.insert(temp, 0, location[:, 0], axis=2)#插入纬度
        #print ("move_data.shape ",move_data.shape)
        # print ("move_data[0,0:3,:] = \n",move_data[0,0:3,:])
        move_data = move_data[:,:,:]  # [20:40,]
        print("move_data.shape ", move_data.shape)
        print("np.argwhere(move_data == 1.0):", np.argwhere(move_data == 1.0))
        # type <class 'numpy.ndarray'> [[   7 2512    2]] shape(1,3)
        # print("np.where(move_data == 1.0)",np.where(move_data == 1.0))
        # type <class 'tuple'> (array([7], dtype=int64), array([2512], dtype=int64), array([2], dtype=int64))
        move_data = move_data.tolist()

        # generate and save html
        m = folium.Map([19.99, 110.3], zoom_start=12)  # zoom_start small or big  tiles='stamentoner'
        hm = plugins.HeatMapWithTime(move_data,#shape(step,x*y,value) value=全1全0均匀成红色，密集成紫色
                                     radius=22, # 10点太小有间隙，22看不到点间隙
                                     min_opacity=0, max_opacity=0.45,#透明度 0完全透明，1完全不透明看不到底图
                                     scale_radius=False # True不显示热度图 Scale the radius of the points based on the zoom level.
                                     )
        # zoom_start small or big  tiles='stamentoner'
        hm.add_to(m)
        save_path = os.path.join(out_dir, "map_%s_start_24frames.html"%(key))
        # 保存为html文件
        m.save(save_path)
        print(save_path,"has been saved!")

        # channel dest
        h5_data = h5_data_2chan[:, 1, :, :]  # dest start
        h5_data = np.maximum(h5_data, 0)  # 逐位比较取较大者
        h5_data = np.minimum(h5_data, 255)  # 逐位比较取较小者
        print("h5_data.dtype = ", h5_data.dtype)  # int32
        print("max h5_data = ", np.max(h5_data))  # max data =  415
        #print("mean h5_data = ", np.mean(h5_data))  # mean data =
        print("min h5_data = ", np.min(h5_data))  # min data =  0
        #h5_data = (h5_data - float(np.min(h5_data))) / (np.max(h5_data) - np.min(h5_data))  # 归一化，（0-1），最大值为1
        h5_data = h5_data / 256.0
        #print("h5_data\n", h5_data)
        print("h5_data.dtype = ", h5_data.dtype)  # float64
        print("max h5_data after scale", np.max(h5_data))  # max data =  1
        #print("mean h5_data after scale", np.mean(h5_data))  # mean data =
        print("min h5_data after scale", np.min(h5_data))  # min data =  0
        # print("h5_data\n", h5_data)
        print("np.argwhere(h5_data == 1.0):", np.argwhere(h5_data == 1.0))
        # print("#################array reshape######################")
        # 转换过程：h5 data(24, 75, 50) to move_data(24,75*50,3)
        # move_data shape (24, 75*50) = (step, samples, data)
        # move_data shape 3 = (lat,lon,val)
        data = h5_data.reshape((h5_data.shape[0], col * row, 1), order='F')  # 最后一维表示value order= C F A
        location = location.reshape((col * row, 2), order='C') # order= C F A
        #print("location\n",location)
        temp = np.insert(data, 0, location[:, 1], axis=2)  # 插入经度
        move_data = np.insert(temp, 0, location[:, 0], axis=2)  # 插入纬度
        #print("move_data.shape ", move_data.shape)
        # print ("move_data[0,0:3,:] = \n",move_data[0,0:3,:])
        move_data = move_data[:, :, :]  # [20:40,]
        print("move_data.shape ", move_data.shape)
        print ("np.argwhere(move_data == 1.0):",np.argwhere(move_data == 1.0)) #[0,:] elesun
        #type <class 'numpy.ndarray'> [[   7 2512    2]] shape(1,3)
        #print("np.where(move_data == 1.0)",np.where(move_data == 1.0))
        #type <class 'tuple'> (array([7], dtype=int64), array([2512], dtype=int64), array([2], dtype=int64))
        #print ("move_data(np.argwhere(move_data == 1.0)[0,:] \n",move_data(list(np.argwhere(move_data == 1.0)[0,:])))
        #elesun
        #h5_data[14 45 23] step x y --> step n 2 move_data[14 1770 2]
        move_data = move_data.tolist()
        # generate and save html
        m = folium.Map([19.99, 110.3], zoom_start=12)  # zoom_start small or big  tiles='stamentoner'
        hm = plugins.HeatMapWithTime(move_data,
                                     radius=22, #
                                     min_opacity=0, max_opacity=0.45,
                                     scale_radius=False
                                     )
        hm.add_to(m)
        save_path = os.path.join(out_dir, "map_%s_dest_24frames.html" % (key))
        # 保存为html文件
        m.save(save_path)
        print(save_path, "has been saved!")
    fr.close()

def test_beijing_traffic():
    loc_downleft = [39.7938, 116.2033]  # latitude,longitude
    loc_upright = [40.0403, 116.5358]
    row = 32
    col = 32

    print("#################area split######################")
    location = area_split(loc_downleft, loc_upright, row, col)
    # print("location\n",location)
    print("location[0,0,:] = \n", location[0, 0, :])
    print("location.shape ", location.shape)

    print("#################read small h5######################")
    # h5 data shape = (48, 32, 32) = (step, h, w)
    frname = "small.h5"
    print("reading ", frname)
    fr = h5py.File(frname, 'r')
    print("fr.keys() = ", fr.keys())
    data = fr['small_data'][()]
    fr.close()
    print("data.shape = ", data.shape)
    print("data.dtype = ", data.dtype)
    print("max data = ", np.max(data))
    print("min data = ", np.min(data))
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # print ("data = ",data)
    print("data[0,0,:] = \n", data[0, :, 0])

    print("#################array reshape######################")
    # move_data shape (48, 1024, 3) (step, samples, data)
    # data shape 3 = (lat,lon,val)
    # 转换过程：h5 data(48, 32, 32) to (48,32*32,3),3 =(lat,lon,val)
    data = data.reshape((48, 32 * 32, 1), order='C') #
    location = location.reshape((32 * 32, 2), order='C') #
    # print("location\n",location)
    temp = np.insert(data, 0, location[:, 1], axis=2)
    move_data = np.insert(temp, 0, location[:, 0], axis=2)
    print("move_data.shape ", move_data.shape)
    print("move_data[0,0:3,:] = \n", move_data[0, 0:3, :])
    data2 = move_data[20:40, ]
    print("data2.shape ", data2.shape)
    # print ("data2[0,0:33,:] \n",data2[0,0:33,:])
    data2 = data2.tolist()

    m = folium.Map([39.9, 116.35], zoom_start=11)  # zoom_start small or big  tiles='stamentoner' #data1 data2
    # m = folium.Map([35, 110], zoom_start=5)#zoom_start small or big  tiles='stamentoner'  #data3
    hm = plugins.HeatMapWithTime(data2, radius=22)  # data1 data2
    # hm = plugins.HeatMapWithTime(data3,radius=8) # data3
    hm.add_to(m)
    save_path = os.path.join(out_dir, "test_beijing.html")
    # 保存为html文件
    m.save(save_path)
    print(save_path, "has been saved!")
    # 默认浏览器打开
    webbrowser.open(save_path)

if __name__ == '__main__':
    in_dir = "stage_03/haikou_shape.h5"
    out_dir = "stage_05"
    if not os.path.exists(in_dir):
        print("there is not directory", in_dir)
    print("input directory is ", in_dir)
    if os.path.exists(out_dir):
        # os.remove(out_dir) #删除整个目录
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    time1 = time.time()
    #test_beijing_traffic()#用北京城市流量测试
    gen_data_fr_h5(in_dir, out_dir)

    time2=time.time()
    print ('time use: ' + str(time2 - time1) + ' s')