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

def extract_inf(in_dir,out_dir):
    file_list = os.listdir(in_dir)
    print('file_list is ',file_list)
    print('num of file_list =',len(file_list))
    for filename in file_list:
        file_path = os.path.join(in_dir, filename)
        print ("%s has been find!"%file_path)
        print("---------------------数据概览-----------------------")
        df = pd.read_csv(file_path,sep='\t')
        #print("df\n",df)
        print("df.shape",df.shape)
        df.dropna(axis=0, how='any',inplace=True) # 去除空值
        print("df.shape after dropna",df.shape)
        #print("df.describe\n",df.describe())
        print("df.index.values\n",df.index.values)
        print("df.columns.values\n",df.columns.values)
        print("df.head(10)\n",df.head(10))
        print("---------------------经纬度最值-----------------------")
        starting_lng_max = df['%s.starting_lng'%(filename[:-4])].describe()['max']
        starting_lng_min = df['%s.starting_lng'%(filename[:-4])].describe()['min']
        starting_lat_max = df['%s.starting_lat'%(filename[:-4])].describe()['max']
        starting_lat_min = df['%s.starting_lat'%(filename[:-4])].describe()['min']
        print("starting_lng_max:", starting_lng_max)
        print("starting_lng_min:", starting_lng_min)
        print("starting_lat_max:", starting_lat_max)
        print("starting_lat_min:", starting_lat_min)

        dest_lng_max = df['%s.dest_lng'%(filename[:-4])].describe()['max']
        dest_lng_min = df['%s.dest_lng'%(filename[:-4])].describe()['min']
        dest_lat_max = df['%s.dest_lat'%(filename[:-4])].describe()['max']
        dest_lat_min = df['%s.dest_lat'%(filename[:-4])].describe()['min']
        print("dest_lng_max:", dest_lng_max)
        print("dest_lng_min:", dest_lng_min)
        print("dest_lat_max:", dest_lat_max)
        print("dest_lat_min:", dest_lat_min)

        dest_lat_mean = df['%s.dest_lat'%(filename[:-4])].mean()
        dest_lat_max = df['%s.dest_lat'%(filename[:-4])].max()
        dest_lat_min = df['%s.dest_lat'%(filename[:-4])].min()
        print("dest_lat_mean:", dest_lat_mean)
        print("dest_lat_max:", dest_lat_max)
        print("dest_lat_min:", dest_lat_min)
        ########################创建CSV文件（包含日期时间、经度、纬度）##########################
        start_df = df[['%s.arrive_time'%(filename[:-4]), '%s.starting_lng'%(filename[:-4]), '%s.starting_lat'%(filename[:-4])]]
        dest_df = df[['%s.departure_time'%(filename[:-4]), '%s.dest_lng'%(filename[:-4]), '%s.dest_lat'%(filename[:-4])]]
        #print("start_df\n",start_df)
        print("start_df.shape",start_df.shape)
        #print("dest_df\n",dest_df)
        print("dest_df.shape",dest_df.shape)
        start_df =start_df.drop(start_df[(start_df["%s.arrive_time"%(filename[:-4])] == "0000-00-00 00:00:00")].index)
        dest_df =dest_df.drop(dest_df[(dest_df["%s.departure_time"%(filename[:-4])] == "0000-00-00 00:00:00")].index)

        start_df =start_df.drop(start_df[(start_df["%s.starting_lng"%(filename[:-4])] >= 110.5)].index)
        start_df =start_df.drop(start_df[(start_df["%s.starting_lng"%(filename[:-4])] < 110.1)].index)
        start_df =start_df.drop(start_df[(start_df["%s.starting_lat"%(filename[:-4])] >= 20.08)].index)
        start_df =start_df.drop(start_df[(start_df["%s.starting_lat"%(filename[:-4])] < 19.9)].index)
        #print("start_df\n",start_df)
        print("start_df.shape after inside area",start_df.shape)
        dest_df =dest_df.drop(dest_df[(dest_df["%s.dest_lng"%(filename[:-4])] >= 110.5)].index)
        dest_df =dest_df.drop(dest_df[(dest_df["%s.dest_lng"%(filename[:-4])] < 110.1)].index)
        dest_df =dest_df.drop(dest_df[(dest_df["%s.dest_lat"%(filename[:-4])] >= 20.08)].index)
        dest_df =dest_df.drop(dest_df[(dest_df["%s.dest_lat"%(filename[:-4])] < 19.9)].index)
        #print("dest_df\n",dest_df)
        print("dest_df.shape after inside area",dest_df.shape)
        save_path = os.path.join(out_dir,filename[:-4] + "_start.csv")
        start_df.to_csv(save_path,index=True,header=True) #
        save_path = os.path.join(out_dir,filename[:-4] + "_dest.csv")
        dest_df.to_csv(save_path,index=True,header=True) #

if __name__ == '__main__':
    in_dir = "datasets"
    out_dir = "stage_01"
    if not os.path.exists(in_dir):
        print("there is not directory", in_dir)
    print("image directory is ", in_dir)
    if os.path.exists(out_dir):
        # os.remove(out_dir) #删除整个目录
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    time1 = time.time()
    extract_inf(in_dir, out_dir)
    time2=time.time()
    print ('time use: ' + str(time2 - time1) + ' s')
