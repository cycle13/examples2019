# -*- coding: utf-8 -*-
"""
haikou traffice
fun:
	save csv
env:
	Win7 64bit anaconda;python 3.6;tensorflow1.10.1;Keras2.2.4
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
import time

def dtxy2csv(in_dir,out_dir):
    file_list = os.listdir(in_dir)
    print('file_list is ',file_list)
    print('num of file_list =',len(file_list))
    for filename in file_list:
        file_path = os.path.join(in_dir, filename)
        print ("%s has been find!"%file_path)
        if filename[-9:-4] == "start" :
            start_df = pd.read_csv(file_path)
            #######################创建CSV文件（包含日期、小时序号、经度所在X，维度所在Y）##########################
            st_date_list = []
            st_time_list = []
            st_lng_list = []
            st_lat_list = []
            for index, row in start_df.iterrows():
                # print("index",index)
                # 转为时间数组
                timeArray = time.strptime(row['%s.arrive_time'%(filename[:-10])], "%Y-%m-%d %H:%M:%S")
                # print ("timeArray",timeArray)
                # timeArray可以调用tm_year等
                # print("timeArray.tm_year", timeArray.tm_year)  # 2017
                # print("timeArray.tm_mon", timeArray.tm_mon)  # 5
                # print("timeArray.tm_mday", timeArray.tm_mday)  # 19
                # print("timeArray.tm_hour", timeArray.tm_hour)  # 1
                # print("timeArray.tm_min", timeArray.tm_min)  # 12
                # print("timeArray.tm_sec", timeArray.tm_sec)  # 4
                st_date_list.append('%04d'%(timeArray.tm_year)+'%02d'%(timeArray.tm_mon)+'%02d'%(timeArray.tm_mday))
                st_time_list.append(timeArray.tm_hour)
                if ((row['%s.starting_lng'%(filename[:-10])] - 110.10) / (110.50 - 110.10)) >= 1 :
                    print("error value",row['%s.starting_lng'%(filename[:-10])])
                if ((row['%s.starting_lat'%(filename[:-10])] - 19.9) / (20.08 - 19.9)) >= 1 :
                    print("error value", row['%s.starting_lat' % (filename[:-10])])
                st_lng_list.append(int(75.0*(row['%s.starting_lng'%(filename[:-10])] - 110.10) / (110.50 - 110.10))) #海口经度75公里
                st_lat_list.append(int(50.0 * (row['%s.starting_lat'%(filename[:-10])] - 19.9) / (20.08 - 19.9)))  #海口纬度50公里
            #print("st_date_list",st_date_list)
            #print("st_time_list",st_time_list)
            #print("st_lng_list",st_lng_list)
            #print("st_lat_list",st_lat_list)
            #start_df2 = pd.DataFrame(columns=["st_date"], data=st_date_list)
            start_df2_dict = {
                "st_date" : st_date_list,
                "st_time" : st_time_list,
                "st_X"    : st_lng_list,
                "st_Y"    : st_lat_list
            }
            start_df2 = pd.DataFrame(data=start_df2_dict)
            print("start_df2.describe\n",start_df2.describe())
            save_path = os.path.join(out_dir, filename[:-10] + "_start2.csv")
            start_df2.to_csv(save_path,index=True,header=True) #
        elif filename[-8:-4] == "dest" :
            dest_df = pd.read_csv(file_path)
            ed_date_list = []
            ed_time_list = []
            ed_lng_list = []
            ed_lat_list = []
            for index, row in dest_df.iterrows():
                # print("index",index)
                # 转为时间数组
                timeArray = time.strptime(row['%s.departure_time'%(filename[:-9])], "%Y-%m-%d %H:%M:%S")
                # print ("timeArray",timeArray)
                # timeArray可以调用tm_year等
                # print("timeArray.tm_year", timeArray.tm_year)  # 2017
                # print("timeArray.tm_mon", timeArray.tm_mon)  # 5
                # print("timeArray.tm_mday", timeArray.tm_mday)  # 19
                # print("timeArray.tm_hour", timeArray.tm_hour)  # 1
                # print("timeArray.tm_min", timeArray.tm_min)  # 12
                # print("timeArray.tm_sec", timeArray.tm_sec)  # 4
                ed_date_list.append('%04d'%(timeArray.tm_year)+'%02d'%(timeArray.tm_mon)+'%02d'%(timeArray.tm_mday))
                ed_time_list.append(timeArray.tm_hour)
                if ((row['%s.dest_lng' % (filename[:-9])] - 110.10) / (110.50 - 110.10)) >= 1:
                    print("error value", row['%s.dest_lng' % (filename[:-9])])
                if ((row['%s.dest_lat' % (filename[:-9])] - 19.9) / (20.08 - 19.9)) >= 1:
                    print("error value", row['%s.dest_lat' % (filename[:-9])])
                ed_lng_list.append(int(75.0*(row['%s.dest_lng'%(filename[:-9])] - 110.10) / (110.50 - 110.10))) #海口经度75公里
                ed_lat_list.append(int(50.0 * (row['%s.dest_lat'%(filename[:-9])] - 19.9) / (20.08 - 19.9)))  # 海口纬度50公里
            #print("ed_date_list",ed_date_list)
            #print("ed_time_list",ed_time_list)
            #print("ed_lng_list",ed_lng_list)
            #print("ed_lat_list",ed_lat_list)
            #dest_df2 = pd.DataFrame(columns=["ed_date"], data=ed_date_list)
            dest_df2_dict = {
                "ed_date" : ed_date_list,
                "ed_time" : ed_time_list,
                "ed_X"    : ed_lng_list,
                "ed_Y"    : ed_lat_list
            }
            dest_df2 = pd.DataFrame(data=dest_df2_dict)
            print("dest_df2.describe\n",dest_df2.describe())
            save_path = os.path.join(out_dir, filename[:-9] + "_dest2.csv")
            dest_df2.to_csv(save_path,index=True,header=True) #

if __name__ == '__main__':
    in_dir = "stage_01"
    out_dir = "stage_02"
    if not os.path.exists(in_dir):
        print("there is not directory", in_dir)
    print("image directory is ", in_dir)
    if os.path.exists(out_dir):
        # os.remove(out_dir) #删除整个目录
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    time1 = time.time()
    dtxy2csv(in_dir, out_dir)
    time2=time.time()
    print ('time use: ' + str(time2 - time1) + ' s')