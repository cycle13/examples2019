# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.setrecursionlimit(10000000)#手动设置递归调用深度

def data_process(dataset_dir,step=100):
    # 读取整个csv文件
    # df = pd.read_csv("air_data.csv", header=0, index_col=None) # dtype={"station_id":np.int,"PM25_AQI_value":np.int}
    df = pd.read_csv(dataset_dir, header=0, index_col=None)
    df = df.dropna(axis=0, how="any") # 去除空值nan

    # 对数据中的某一列进行归一化处理
    pm25_max = np.max(df["PM25_AQI_value"].values[:]) # 取PM25里面的所有值的最大值
    pm25_min = np.min(df["PM25_AQI_value"].values[:])
    print("pm25_max", pm25_max)
    print("pm25_min", pm25_min)
    x = df["PM25_AQI_value"].values
    x_scale = (x - pm25_min)/(pm25_max - pm25_min)  # 归一化
    # x_scale = 1 - (x - pm25_min)/(pm25_max - pm25_min)  # 归一化
    print("x_scale.shape:", x_scale.shape)
    print("x_scale:", x_scale)
    df["PM25_AQI_value"] = x_scale

    station_id = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014]
    wq_data = []
    for i in station_id:
        wq_data.append(df.loc[(df['station_id'] == i), ['PM25_AQI_value']].values[0:step])
    #print('wq_data\n', wq_data)
    wq_data = np.array(wq_data)
    print('wq_data.shape:', wq_data.shape)
    #print('wq_data:\n', wq_data)

    row = 10
    col = 16
    ddo = np.zeros((row, col, step))
    for i in range(row):
        for j in range(col):
            # for t in range(step):
            if i == 5 and j == 0:
                ddo[i, j, :] = wq_data[8, :, 0]  # s9
            elif i == 1 and j == 2:
                ddo[i, j, :] = wq_data[9, :, 0]  # s10
            elif i == 2 and j == 3:
                ddo[i, j, :] = wq_data[10, :, 0]  # s11
            elif i == 3 and j == 3:
                ddo[i, j, :] = wq_data[11, :, 0]  # s12
            elif i == 9 and j == 4:
                ddo[i, j, :] = wq_data[13, :, 0]  # s14
            elif i == 6 and j == 5:
                ddo[i, j, :] = wq_data[5, :, 0]  # s6
            elif i == 6 and j == 7:
                ddo[i, j, :] = wq_data[6, :, 0]  # s7
            elif i == 2 and j == 8:
                ddo[i, j, :] = wq_data[0, :, 0]  # s1
            elif i == 3 and j == 10:
                ddo[i, j, :] = wq_data[12, :, 0]  # s13
            elif i == 0 and j == 12:
                ddo[i, j, :] = wq_data[7, :, 0]  # s8
            elif i == 6 and j == 13:
                ddo[i, j, :] = wq_data[4, :, 0]  # s5
            elif i == 4 and j == 14:
                ddo[i, j, :] = wq_data[2, :, 0]  # s3
            elif i == 1 and j == 15:
                ddo[i, j, :] = wq_data[1, :, 0]  # s2
            elif i == 2 and j == 15:
                ddo[i, j, :] = wq_data[3, :, 0]  # s4
            else:
                ddo[i, j, :] = 0
        # print("ddo\n",ddo)
        # print("ddo.shape", ddo.shape)
    # c3_data.append(ddo)
    c3_data = ddo
    # print('c3_data:\n', c3_data)
    c3_data = np.array(c3_data)
    print('c3_data.shape', c3_data.shape)
    # print('c3_data:\n', c3_data)
    return c3_data

def plt_heat(data,out_dir,step=100):
    #创建流入量图文件路径
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i in range(step):#10 for test ; data.shape[0] = 4888
        #print ("data[%04d,:,:].shape is "%(i),data[ i, :, :].shape)#
        print("data[:,:,%04d].shape is " % (i), data[:, :, i].shape) #
        plt.subplots()
        #sns.heatmap(data[i, :, :],vmin=0,vmax=500)
        sns.heatmap(data[:, :, i],vmin=0,vmax=1)
        #save png
        plt.savefig(out_dir + '/' + '%04d_contrast.png' % i)
        #plt.show()


if __name__ == '__main__':
    dataset_dir = "air_data.csv"
    out_dir = "out"
    step = 1000  # max = 2400
    c3_data = data_process(dataset_dir,step)
    plt_heat(c3_data, out_dir, step)