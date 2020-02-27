#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
fun:
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3
"""
from __future__ import print_function
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
import process
import model

def train(train_seq,valid_seq,seq_len=16,input_len=12,interval=6,step=4) :
    print("********train************")
    assert type(train_seq) == np.ndarray
    assert train_seq.ndim == 3
    print("train_seq.shape",train_seq.shape)
    assert train_seq.shape[1] == seq_len
    assert type(valid_seq) == np.ndarray
    assert valid_seq.ndim == 3
    print("valid_seq.shape",valid_seq.shape)
    assert valid_seq.shape[1] == seq_len
    predict = []
    """
    # Xt+n = Xt
    for i in range(train_seq.shape[0]) :
        temp2_predict = []
        for n in range(1,step+1,1) :
            lon_predict = train_seq[i, input_len-1, 2]
            lat_predict = train_seq[i, input_len-1, 1]
            temp1_predict = [lat_predict,lon_predict] # y,x
            temp2_predict.append(temp1_predict)
        predict.append(temp2_predict)
    """  
    # Xt+1 = Xt + n*(Xt - Xt-1)
    for i in range(train_seq.shape[0]) :
        temp2_predict = []
        for n in range(1,step+1,1) :
            delta_x = train_seq[i, input_len-1, 2] - train_seq[i, input_len-1-1, 2]
            delta_y = train_seq[i, input_len-1, 1] - train_seq[i, input_len-1-1, 1]
            lon_predict = train_seq[i, input_len-1, 2] + n * delta_x # Xt+1 = Xt + n*(Xt - Xt-1)
            lat_predict = train_seq[i, input_len-1, 1] + n * delta_y
            temp1_predict = [lat_predict,lon_predict] # y,x
            temp2_predict.append(temp1_predict)
        predict.append(temp2_predict)

    truth = train_seq[:,:,1:3]
    print("truth.shape",truth.shape)
    print("truth\n",truth)
    predict = np.array(predict)
    print("predict.shape",predict.shape)
    print("predict\n",predict)
    return truth,predict
    
def evaluate(truth,predict,seq_len=16,input_len=12) :
    print("********evaluate************")
    assert type(truth) == np.ndarray
    assert truth.ndim == 3
    print("truth.shape",truth.shape)
    assert truth.shape[1] == seq_len
    assert truth.shape[2] == 2
    assert type(predict) == np.ndarray
    assert predict.ndim == 3
    print("predict.shape",predict.shape) # 10,4,2
    assert predict.shape[1] == seq_len - input_len
    assert predict.shape[2] == 2
    rmse_temp4 = []
    for i in range(predict.shape[0]) : # 10
        rmse_temp2 = []
        for j in range(predict.shape[1]) : # 4
            rmse_temp1 = np.sqrt((predict[i,j,0]-truth[i,j-(seq_len-input_len),0])**2+(predict[i,j,1]-truth[i,j-(seq_len-input_len),1])**2)
            rmse_temp2.append(rmse_temp1)
        rmse_temp2 = np.array(rmse_temp2)
        rmse_temp3 = np.sum(rmse_temp2)
        rmse_temp4.append(rmse_temp3)
    rmse_temp4 = np.array(rmse_temp4)
    rmse_temp3 = np.mean(rmse_temp4)
    return rmse_temp3

def draw_route(truth,predict,draw_dir="draw",seq_len=16,input_len=12,interval=6) :
    if tf.gfile.Exists(draw_dir):
        tf.gfile.DeleteRecursively(draw_dir)
    tf.gfile.MakeDirs(draw_dir)
    
    assert truth.shape[1] == seq_len
    assert predict.shape[1] == seq_len-input_len
    for i in range(0,truth.shape[0]) :# 一个台风ID一张图
        fig = plt.figure("expermenets", figsize=(12.80, 9.60))  # 实际像素值放大100倍
        for j in range(seq_len): #前面12张后面4张颜色不同
            if j < input_len :
                curve_1, = plt.plot(truth[i, j, 1], truth[i, j, 0], 'bo-')
            else :
                curve_2, = plt.plot(truth[i, j, 1], truth[i, j, 0], 'bx-')
                curve_3, = plt.plot(predict[i,j-input_len,1], predict[i,j-input_len,0], 'r^-')
        plt.axis('on')  # 关掉坐标轴为 off
        plt.title('typhoon %03d route '%i)  # 图像题目
        #plt.legend(["history","ground truth","predict"], loc='best')
        #plt.legend(handles=[l1, l2], labels=['up', 'down'], loc='best')
        plt.legend(handles=[curve_1, curve_2, curve_3], labels=["history","ground truth","predict"], loc='best')

        #pl.legend([plot1, plot2], ("red line", "green circles"), "best", numpoints=1)
        plt.grid(False)
        plt.xlabel('lon')
        plt.ylabel('lat')
        plt.xticks(np.arange(1000, 2001, step=100), rotation=0)  # xticks>xlim
        plt.yticks(np.arange(90, 501, step=50), rotation=0)  # yticks>ylim
        plt.xlim(1000, 2000)
        plt.ylim(90, 500)
        save_path = os.path.join(draw_dir, "ID_%03d_route.jpg" %(i+1))
        plt.savefig(save_path)
        # plt.show()
        plt.close()  # 避免产生数量过多warning
        print("%s haved been ploted !"%save_path)

def test_predict_linemove(images_folder_path,track_folder_path,
                 seq_len=16,input_len=12,interval=6,step=4):
    print("********test_predict************")
    print("########linear move########")
    test_size = 0
    for lists in os.listdir(images_folder_path):
        sub_path = os.path.join(images_folder_path, lists)
        if os.path.isdir(sub_path):
            test_size = test_size + 1
    print("test_size",test_size)

    predict1 = []
    id_lists = os.listdir(track_folder_path)
    # id_lists.sort()  #
    id_lists.sort(key=lambda x: int(x)) # elesun
    print("len id_lists", len(id_lists))
    print("id_lists\n", id_lists)
    for id_list in id_lists :
        id_path = os.path.join(track_folder_path, id_list)
        temp2_predict = []
        for txt_lists in os.listdir(id_path):
            txt_path = os.path.join(id_path, txt_lists)
            #print("txt_path",txt_path)
            df = pd.read_csv(txt_path,delim_whitespace=True,header=None,index_col=None,
                             names=["ID", "TIME", "I", "LAT", "LON", "PRES", "WND"],
                             dtype={"ID": np.int, "TIME": str, "LAT": np.int, "LON": np.int})
            #df.columns = ["ID", "TIME", "I", "LAT", "LON", "PRES", "WND"]
            #df.index
            #df.columns
            last_lat = np.squeeze(df.loc[(df["TIME"] == "066"), ["LAT"]].values[:])
            #print("last_lat",last_lat)
            last_last_lat = np.squeeze(df.loc[(df["TIME"] == "060"), ["LAT"]].values[:])
            last_lon = np.squeeze(df.loc[(df["TIME"] == "066"), ["LON"]].values[:])
            #print("last_lon", last_lon)
            last_last_lon = np.squeeze(df.loc[(df["TIME"] == "060"), ["LON"]].values[:])
            delta_x = last_lon - last_last_lon
            delta_y = last_lat - last_last_lat
            #print("delta_x",delta_x)
            #print("delta_y", delta_y)
            for i_step in range(1, step + 1, 1):
                lon_predict = last_lon + i_step * 1.15 * delta_x # Xt+1 = Xt + n*(Xt - Xt-1)
                lat_predict = last_lat + i_step * 1.15 * delta_y
                temp1_predict = [lat_predict, lon_predict]  # y,x
                temp2_predict.append(temp1_predict)
        predict1.append(temp2_predict)
    predict1 = np.array(predict1,np.float) # 线性叠加预测结果
    print("predict1.shape", predict1.shape)
    print("predict1[0,0,:]\n", predict1[0,0,:])
    return predict1 # predict predict1 predict2

def generate_result(predict,results_folder_path,seq_len=16,input_len=12,interval=6,step=4):
    assert type(predict) == np.ndarray
    assert predict.ndim == 3
    assert predict.shape[1] == seq_len - input_len

    lat_list = np.ravel(predict[:,:,0])
    lon_list =  np.ravel(predict[:,:,1])
    #print("lon_list\n",lon_list)
    #print("lat_list\n", lat_list)
    ###输出标准结果文件
    '''
    1. 输出文件格式为CSV，内容包括两列，分别为：预测位置纬度，预测位置经度
    2. 注意！！！单位分别为0.1°N和0.1°E，故选手应提交“默认的经纬度数值*10”
    3. 结果文件为有序文件，按台风id排主序，按时间先后顺序排副序（每个台风有四行数据）
    4. 输出文件不含首行表头和首列时间。
    '''
    # 案例中为随机生成的数字，实际应该是选手模型预测的结果。
    # data = {'lat':np.arange(test_size*4),
    #           'lon':np.arange(test_size*4)}
    # print("np.arange(test_size*4)",np.arange(test_size*4))
    # data = {'lat': [224,224,224,224],
    #         'lon': [1218,1218,1218,1218]}
    data = {'lat': lat_list,
            'lon': lon_list}
    df = pd.DataFrame(data)
    # 注意路径不能更改，index和header都需要设置为None
    df.to_csv(results_folder_path, index=None,header=None)
    print(results_folder_path,"has been saved!")
    ###输出标准结果文件

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "-1" #elesun "1,0"
    seq_len=16
    input_len=12
    interval=6
    step=4
    ####################  ######################
    mode = "train" #train test docker
    print("#################work mode",mode,"#######################")
    if mode == "train" :
        # 数据预处理typhoon
        train_track_folder_path = "dataset/Train/Track"  #训练集台风路径文件路径
        train_images_folder_path = "dataset/Train/Image"  #训练集卫星云图文件路径
        results_folder_path = "result/result.csv"  # 结果输出文件路径
        (train_seq, valid_seq,
         i_max_val, i_min_val,
         lat_max_val, lat_min_val,
         lon_max_val, lon_min_val,
         press_max_val, press_min_val,
         wnd_max_val, wnd_min_val) = process.generate_train_seq(train_images_folder_path,train_track_folder_path)

        #数据预处理mnist
        n_input = 5 # 28
        n_step = 4 # 28
        n_hidden = 64 # 128
        output_units = 2 # 10
        # (trainX, trainY), (validX, validY) = mnist.load_data()
        # trainX = trainX.reshape(-1, n_step, n_input)
        # validX = validX.reshape(-1, n_step, n_input)
        # trainX = trainX.astype('float32')
        # validX = validX.astype('float32')
        # trainX /= 255
        # validX /= 255
        # trainY = keras.utils.to_categorical(trainY, n_classes)
        # validY = keras.utils.to_categorical(validY, n_classes)
        trainX = train_seq[:,-8:-4,:] # [:,-8:-4,:]根据所有数据预测经纬度 [:,-8:-4,1:3]根据经纬度预测经纬度
        trainY = train_seq[:,-4:,1:3]
        validX = valid_seq[:,-8:-4,:] # [:,-8:-4,:] [:,-8:-4,1:3] 对应修改
        validY = valid_seq[:,-4:,1:3]
        print('trainX.shape:', trainX.shape)
        print('trainY.shape:', trainY.shape)
        print('validX.shape:', validX.shape)
        print('validY.shape:', validY.shape)
        print('trainX[0]\n', trainX[0])
        print('trainY[0]\n:', trainY[0])
        # 搭建网络训练模型
        model.helloworld()
        network = model.build_model(n_step=n_step,n_input=n_input)
        history = model.train_model(trainX, trainY,validX, validY,network)
        model.plt_result(history)
        model.model_evaluate(validX, validY)

        #line move train predict and evaluate
        #truth,predict = train(train_seq,valid_seq)
        #rmse = evaluate(truth,predict)
        #print("rmse = ",rmse)
        #draw_route(truth,predict)
        #generate_result(predict,results_folder_path)
    elif mode == "test" :
        #test_path = "/home/data"						#测试集路径。存储形式同“训练集”，下含Track和Image两个文件夹
        track_folder_path = "dataset/Test/Track"#			#测试集台风路径文件路径
        images_folder_path = "dataset/Test/Image"			#测试集卫星云图文件路径
        results_folder_path = "result/result.csv"	#结果输出文件路径
        i_max_val, i_min_val = 6.0, 0.0
        lat_max_val, lat_min_val = 485.0, 92.0
        lon_max_val, lon_min_val = 1885.0, 1052.0
        press_max_val, press_min_val = 1010.0, 888.0
        wnd_max_val, wnd_min_val = 72.0, 10.0
        model.helloworld()
        # 线性叠加预测，便于后面对比、融合、作图
        predict1 = test_predict_linemove(images_folder_path, track_folder_path)
        # LSTM深度学习模型预测，便于后面融合、作图
        print("########lstm model########")
        testX, testX_norm = process.generate_test_seq(images_folder_path, track_folder_path, i_max_val, i_min_val,
                                                      lat_max_val, lat_min_val, lon_max_val, lon_min_val,
                                                      press_max_val, press_min_val, wnd_max_val, wnd_min_val)
        #print("testX",testX)
        predict2 = model.model_predict(testX_norm, lat_max_val, lat_min_val, lon_max_val, lon_min_val)  # LSTM预测结果
        print("predict2.shape", predict2.shape)
        print("predict2[0,0,:]\n", predict2[0, 0, :])
        #两种模型融合
        predict = 0 * predict1 + 1 * predict2  # 调整权重系数
        print("predict.shape", predict.shape)
        print("predict\n", predict)  # 融合后的预测结果
        generate_result(predict2,results_folder_path)
        # test阶段绘图，真实值4帧已知+4帧线性叠加预测4帧为truth，4帧LSTM预测结果为predict
        #print("testX.shape",testX.shape) # (8, 4, 5) testX[:,:,1:3]为经纬度数据
        #predict1.shape (8, 4, 2)
        truth = []
        for i in range(predict1.shape[0]):
            # concatente 矩阵水平叠加的效果，列数要保持一致，行数增加
            truth.append(np.concatenate((testX[i,:,1:3], predict1[i,:,:]), axis=0))
        #print("testX[:,:,1:3]",testX[:,:,1:3])
        #print("predict1", predict1)
        truth = np.array(truth,np.float)
        #print("truth after concat predict1\n", truth)
        #print("truth.shape",truth.shape)
        draw_route(truth, predict2, draw_dir="draw_predict", seq_len=8, input_len=4, interval=6)#test时seq_len=4, input_len=0由于未来的真实值无法获取
    elif mode == "docker" :
        test_path = "/home/data"						#测试集路径。存储形式同“训练集”，下含Track和Image两个文件夹
        track_folder_path = "/home/data/Track"			#测试集台风路径文件路径
        images_folder_path = "/home/data/Image"			#测试集卫星云图文件路径
        results_folder_path = "/code/result/result.csv"	#结果输出文件路径 
        model.helloworld()
        predict = test_predict_linemove(images_folder_path, track_folder_path)
        generate_result(predict,results_folder_path)
    else :
        print("mode error!")
    ####################  ######################
    
    ####################  ######################
        


