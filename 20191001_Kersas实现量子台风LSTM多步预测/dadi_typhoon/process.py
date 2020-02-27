# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import re

#根据训练数据集生成网络所需要的序列
def generate_train_seq(train_images_folder_path,train_track_folder_path,
                 seq_len=16,input_len=12,interval=6,valid_len=10) :
    train_size = 0
    for lists in os.listdir(train_images_folder_path):
        sub_path = os.path.join(train_images_folder_path, lists)
        if os.path.isdir(sub_path):
            train_size = train_size + 1
    train_size = 176 # elesun
    print("train_size",train_size)

    typhoon_seq = []
    for id_lists in os.listdir(train_track_folder_path):#dataset/Train/Track
        txt_path = os.path.join(train_track_folder_path, id_lists)#dataset/Train/Track/Tracktxt.
        print("txt_path", txt_path)
        df = pd.read_csv(txt_path, delim_whitespace=True, header=None, index_col=None,
                         names=["ID", "TIME", "I", "LAT", "LON", "PRES", "WND"],
                         dtype={"ID": np.int, "TIME": np.int,"I": np.int, "LAT": np.int, "LON": np.int,
                                "PRES": np.int,"WND": np.int,})
        # df.columns = ["ID", "TIME", "I", "LAT", "LON", "PRES", "WND"]
        # df.index
        # df.columns
        for typhoon_id in range(1,train_size+1,1):
            df1 = df.loc[(df["ID"] == typhoon_id), ["TIME","I", "LAT", "LON", "PRES", "WND"]]
            temp_list = []
            for time_id in range(0, 91, 6): #
                #print("typhoon_id",typhoon_id,"time_id",time_id)
                frame = np.squeeze(df1.loc[(df1["TIME"] == time_id),["I", "LAT", "LON", "PRES", "WND"]].values[:])
                temp_list.append(frame)
            typhoon_seq.append(temp_list)
        #print("typhoon_seq",typhoon_seq)#
        typhoon_seq = np.array(typhoon_seq,np.float)
        #print("typhoon_seq", typhoon_seq)
        print("type typhoon_seq",type(typhoon_seq))
        print("typhoon_seq.shape", typhoon_seq.shape) # typhoon_seq.shape (176, 16, 5)
        # typhoon_seq =
        # Y_train_aims =

        i_max_val = np.max(typhoon_seq[:, :, 0])
        i_min_val = np.min(typhoon_seq[:, :, 0])
        print("i_max_val", i_max_val)
        print("i_min_val", i_min_val)
        i_val_span =  i_max_val - i_min_val
        typhoon_seq[:, :, 0] = (typhoon_seq[:, :, 0] - i_min_val) / (i_max_val - i_min_val)

        lat_max_val = np.max(typhoon_seq[:,:,1])
        lat_min_val = np.min(typhoon_seq[:,:,1])
        print("lat_max_val", lat_max_val)
        print("lat_min_val", lat_min_val)
        lat_val_span = lat_max_val - lat_min_val
        #print(typhoon_seq[:, :, 1])
        typhoon_seq[:, :, 1] = (typhoon_seq[:, :, 1] - lat_min_val) / (lat_max_val - lat_min_val)
        #print(typhoon_seq[:, :, 1])

        lon_max_val = np.max(typhoon_seq[:, :, 2])
        lon_min_val = np.min(typhoon_seq[:, :, 2])
        print("lon_max_val", lon_max_val)
        print("lon_min_val", lon_min_val)
        lon_val_span = lon_max_val - lon_min_val

        typhoon_seq[:, :, 2] = (typhoon_seq[:, :, 2] - lon_min_val) / (lon_max_val - lon_min_val)
        press_max_val = np.max(typhoon_seq[:, :, 3])
        press_min_val = np.min(typhoon_seq[:, :, 3])
        print("press_max_val", press_max_val)
        print("press_min_val", press_min_val)
        press_val_span = press_max_val - press_min_val
        typhoon_seq[:, :, 3] = (typhoon_seq[:, :, 3] - press_min_val) / (press_max_val - press_min_val)
        wnd_max_val = np.max(typhoon_seq[:, :, 4])
        wnd_min_val = np.min(typhoon_seq[:, :, 4])
        print("wnd_max_val", wnd_max_val)
        print("wnd_min_val", wnd_min_val)
        wnd_val_span = wnd_max_val - wnd_min_val
        typhoon_seq[:, :, 4] = (typhoon_seq[:, :, 4] - wnd_min_val)/(wnd_max_val - wnd_min_val)
    
    train_seq = typhoon_seq[0:typhoon_seq.shape[0]-valid_len,:,:]
    valid_seq = typhoon_seq[-valid_len:,:,:]
    #print("train_seq\n",train_seq)
    return train_seq,valid_seq,i_max_val,i_min_val,lat_max_val,lat_min_val,lon_max_val,lon_min_val,press_max_val,press_min_val,wnd_max_val,wnd_min_val

#根据测试数据集生成网络所需要的序列
def generate_test_seq(images_folder_path,track_folder_path,
                 i_max_val,i_min_val,
                 lat_max_val,lat_min_val,lon_max_val,lon_min_val,
                 press_max_val,press_min_val,wnd_max_val,wnd_min_val,
                 seq_len=16,input_len=12,interval=6) :

    test_size = 0
    for lists in os.listdir(images_folder_path):
        sub_path = os.path.join(images_folder_path, lists)
        if os.path.isdir(sub_path):
            test_size = test_size + 1
    print("test_size", test_size)

    id_lists = os.listdir(track_folder_path)
    # id_lists.sort()  #
    id_lists.sort(key=lambda x: int(x))  # elesun
    print("len id_lists", len(id_lists))
    print("id_lists\n", id_lists)
    typhoon_seq = []
    for id_list in id_lists:
        id_path = os.path.join(track_folder_path, id_list)
        for txt_lists in os.listdir(id_path):
            txt_path = os.path.join(id_path, txt_lists)
            #print("txt_path", txt_path)
            df = pd.read_csv(txt_path, delim_whitespace=True, header=None, index_col=None,
                             names=["ID", "TIME", "I", "LAT", "LON", "PRES", "WND"],
                             dtype={"ID": np.int, "TIME": np.int, "I": np.int, "LAT": np.int, "LON": np.int,
                                    "PRES": np.int, "WND": np.int, })
            #typhoon_id = int(re.findall("\d*",txt_lists)[0])
            typhoon_id = int(id_list)
            #print("typhoon_id = ",typhoon_id)
            df1 = df.loc[(df["ID"] == typhoon_id), ["TIME","I", "LAT", "LON", "PRES", "WND"]]
            temp_list = []
            for time_id in range(48, 67, 6):#start 48
                #print("typhoon_id",typhoon_id,"time_id",time_id)
                frame = np.squeeze(df1.loc[(df1["TIME"] == time_id),["I", "LAT", "LON", "PRES", "WND"]].values[:])
                #print("frame",frame)
                temp_list.append(frame)
        typhoon_seq.append(temp_list)
    #print("typhoon_seq",typhoon_seq)#
    typhoon_seq = np.array(typhoon_seq,np.float)
    #print("typhoon_seq", typhoon_seq)
    print("type typhoon_seq",type(typhoon_seq))
    print("typhoon_seq.shape", typhoon_seq.shape) # typhoon_seq.shape (8, 4, 5)
    #print("typhoon_seq\n", typhoon_seq)
    typhoon_seq_norm = typhoon_seq.copy()
    typhoon_seq_norm[:, :, 0] = (typhoon_seq[:, :, 0] - i_min_val) / (i_max_val - i_min_val)
    typhoon_seq_norm[:, :, 1] = (typhoon_seq[:, :, 1] - lat_min_val) / (lat_max_val - lat_min_val)
    typhoon_seq_norm[:, :, 2] = (typhoon_seq[:, :, 2] - lon_min_val) / (lon_max_val - lon_min_val)
    typhoon_seq_norm[:, :, 3] = (typhoon_seq[:, :, 3] - press_min_val) / (press_max_val - press_min_val)
    typhoon_seq_norm[:, :, 4] = (typhoon_seq[:, :, 4] - wnd_min_val) / (wnd_max_val - wnd_min_val)
    #print("max",np.max(typhoon_seq_norm))#不要出现超过1
    #print("min",np.min(typhoon_seq_norm))#不要出现小于0
    typhoon_seq_norm = np.maximum(typhoon_seq_norm, 0.0)
    typhoon_seq_norm = np.minimum(typhoon_seq_norm, 1.0)
    #print("typhoon_seq", typhoon_seq)
    return typhoon_seq,typhoon_seq_norm

if __name__ == "__main__":
    ####################  ######################
    train_track_folder_path = "dataset/Train/Track"  # 训练集台风路径文件路径
    train_images_folder_path = "dataset/Train/Image"  # 训练集卫星云图文件路径
    results_folder_path = "result/result.csv"  # 结果输出文件路径
    (train_seq, valid_seq,
     i_max_val, i_min_val,
     lat_max_val, lat_min_val,
     lon_max_val, lon_min_val,
     press_max_val, press_min_val,
     wnd_max_val, wnd_min_val) = generate_train_seq(train_images_folder_path, train_track_folder_path)
    print(i_max_val, i_min_val,
     lat_max_val, lat_min_val,
     lon_max_val, lon_min_val,
     press_max_val, press_min_val,
     wnd_max_val, wnd_min_val)

    track_folder_path = "dataset/Test/Track"  # #测试集台风路径文件路径
    images_folder_path = "dataset/Test/Image"  # 测试集卫星云图文件路径
    results_folder_path = "result/result.csv"  # 结果输出文件路径
    predict = generate_test_seq(images_folder_path,track_folder_path,
                 i_max_val,i_min_val,
                 lat_max_val,lat_min_val,lon_max_val,lon_min_val,
                 press_max_val,press_min_val,wnd_max_val,wnd_min_val,
                 seq_len=16,input_len=12,interval=6)


