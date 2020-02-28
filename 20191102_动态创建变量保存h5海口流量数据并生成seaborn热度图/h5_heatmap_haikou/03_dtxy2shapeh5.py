# -*- coding: utf-8 -*-
"""
haikou traffice
fun:
	dtxy2shapeh5
env:
	Win7 64bit anaconda;python 3.5;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3,seaborn0.9.0
"""
from __future__ import print_function
import os
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

def dtxy2shapeh5(in_dir,out_dir):
    save_path = os.path.join(out_dir, "haikou_shape.h5")
    fw = h5py.File(save_path, 'w')
    file_list = os.listdir(in_dir)
    print('file_list is ',file_list)
    print('num of file_list =',len(file_list))
    for filename in file_list:
        file_path = os.path.join(in_dir, filename)
        print ("%s has been find!"%file_path)
        if filename[-10:-5] == "start" :
            start_df2 = pd.read_csv(file_path)
            for index, row in start_df2.iterrows():
                #print("index",index)
                if not ("data_" + str(row["st_date"])) in locals().keys():
                    #print("var not defined")
                    exec("data_" + str(row["st_date"]) + "= np.zeros([24, 2, 75, 50], dtype=int)")# elesun shape
                #print("data_" + str(row["st_date"]) + "[row['st_time'], 0, row['st_X'], row['st_Y']] += 1")
                #data_20190519[time,0,x,y] += 1
                exec("data_" + str(row["st_date"]) + "[row['st_time'], 0, row['st_X'], row['st_Y']] += 1")# elesun shape
                # 0 is start channel
        elif filename[-9:-5] == "dest" :
            dest_df2 = pd.read_csv(file_path)
            for index, row in dest_df2.iterrows():
                #print("index", index)
                if not ("data_" + str(row["ed_date"])) in locals().keys():
                    #print("var not defined")
                    exec("data_" + str(row["ed_date"]) + "= np.zeros([24, 2, 75, 50], dtype=int)")# elesun shape
                #print("data_" + str(row["ed_date"]) + "[row['ed_time'], 1, row['ed_X'], row['ed_Y']] += 1")
                # data_20190519[time,1,x,y] += 1
                exec("data_" + str(row["ed_date"]) + "[row['ed_time'], 1, row['ed_X'], row['ed_Y']] += 1")# elesun shape
                # 1 is dest channel
        print("%s has been finished!" % file_path)
    # 获取局部命名空间所有变量名称
    #print("locals().keys()",type(locals().keys()))#<class 'dict_keys'>
    #print("locals().keys()", list(locals().keys()))  #['row', 'data_20170519', 'data_20170508']
    #print([elem for elem in list(locals().keys()) if ("data_" in elem)])#过滤选择含有data_的变量名称字符串的列表
    var_list = [elem for elem in list(locals().keys()) if ("data_" in elem)]
    #print("for var list")
    for i in var_list :
        fw[i[-8:]] = locals()[i]
        #print("locals()[i]",i,locals()[i])
    #打印h5中所有键
    #print("fw.keys() = ", fw.keys()) # python2
    print([key for key in fw.keys()]) #python3
    fw.close()
    print("save h5 ", save_path)


if __name__ == '__main__':
    in_dir = "stage_02"
    out_dir = "stage_03"
    if not os.path.exists(in_dir):
        print("there is not directory", in_dir)
    print("input directory is ", in_dir)
    if os.path.exists(out_dir):
        # os.remove(out_dir) #删除整个目录
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    time1 = time.time()
    dtxy2shapeh5(in_dir, out_dir)
    time2=time.time()
    print ('time use: ' + str(time2 - time1) + ' s')