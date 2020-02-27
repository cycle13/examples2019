#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import model



if __name__ == "__main__":

    #################### 不可修改区域开始 ######################
    test_path = "/home/data"						#测试集路径。存储形式同“训练集”，下含Track和Image两个文件夹
    track_folder_path = "Test/Track"#"/home/data/Track"			#测试集台风路径文件路径
    images_folder_path = "Test/Image"#"/home/data/Image"			#测试集卫星云图文件路径
    results_folder_path = "result/result.csv"#"/code/result/result.csv"	#结果输出文件路径
    #################### 不可修改区域结束 ######################

    ### 参考代码1开始：统计测试集中包含的台风数量
    test_size = 0
    for lists in os.listdir(images_folder_path):
        sub_path = os.path.join(images_folder_path, lists)
        if os.path.isdir(sub_path):
            test_size = test_size + 1
    print("test_size",test_size)

    lat_list = []
    lon_list = []
    for id_lists in os.listdir(track_folder_path):
        id_path = os.path.join(track_folder_path, id_lists)
        for txt_lists in os.listdir(id_path):
            txt_path = os.path.join(id_path, txt_lists)
            print("txt_path",txt_path)
            df = pd.read_csv(txt_path,delim_whitespace=True,header=None,index_col=None,
                             names=["ID", "TIME", "I", "LAT", "LON", "PRES", "WND"],
                             dtype={"ID": np.int, "TIME": str, "LAT": np.int, "LON": np.int})
            #df.columns = ["ID", "TIME", "I", "LAT", "LON", "PRES", "WND"]
            #df.index
            #df.columns
            last_lat = np.squeeze(df.loc[(df["TIME"] == "066"), ["LAT"]].values[:])
            last_lon = np.squeeze(df.loc[(df["TIME"] == "066"), ["LON"]].values[:])
            for i in range(4) :
                lat_list.append(last_lat)
                lon_list.append(last_lon)
            #print(type(df.loc[(df["TIME"] == "066"), ["LAT"]].values[:])) # <class 'numpy.ndarray'>
            #print("df\n", df)
    #lat_list = np.array(lat_list)
    #lon_list = np.array(lon_list)
    print("lat_list",lat_list)
    print("lon_list",lon_list)
    ### 参考代码1结束：统计测试集中包含的台风数量

    ### 调用自己的工程文件
    # 小例子
    model.helloworld()

    ### 参考代码2开始：输出标准结果文件
    '''
    1. 输出文件格式为CSV，内容包括两列，分别为：预测位置纬度，预测位置经度
    2. 注意！！！单位分别为0.1°N和0.1°E，故选手应提交“默认的经纬度数值*10”
    3. 结果文件为有序文件，按台风id排主序，按时间先后顺序排副序（每个台风有四行数据）
    4. 输出文件不含首行表头和首列时间。
    '''
    # 案例中为随机生成的数字，实际应该是选手模型预测的结果。
    # data = {'lat':np.arange(test_size*4),
    #          'lon':np.arange(test_size*4)}
    # print("np.arange(test_size*4)",np.arange(test_size*4))
    # data = {'lat': [224,224,224,224],
    #         'lon': [1218,1218,1218,1218]}
    data = {'lat': lat_list,
            'lon': lon_list}
    df = pd.DataFrame(data)

    # 注意路径不能更改，index和header都需要设置为None
    df.to_csv(results_folder_path, index=None,header=None)
    ### 参考代码2结束：输出标准结果文件
