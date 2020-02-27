# -*- coding: utf-8 -*-
"""
Vehicle plate recognition
using keras
Author: elesun
https://cloud.tencent.com/developer/article/1005199
# -*- coding: utf-8 -*-
"""
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from keras.models import Sequential,Input,Model
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Activation,Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import cv2


#os.environ["CUDA_VISIBLE_DEVICES"] = "0" #"1,0"
#####################车牌数据生成器，################################################
#用于深度神经网络的数据输入
#开源的车牌生成器，随机生成的车牌达到以假乱真的效果
#国内机动车车牌7位，第一位是各省的汉字，第二位是 A-Z 的大写字母，3-7位则是数字、字母混合
from genplate import *

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ]
M_strIdx = dict(zip(chars, range(len(chars))))
#print("M_strIdx\n",M_strIdx)
Ge = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")
model_dir = "./model"
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

def gen(batch_size=32):
    while True:
        l_plateStr, l_plateImg = Ge.genBatch(batch_size, 2, range(31, 65), "plate", (272, 72))
        #print('l_plateStr type :', type(l_plateStr))
        #print('l_plateStr = ', l_plateStr)
        #print('l_plateImg type = ', type(l_plateImg))
        #print('l_plateImg len :', len(l_plateImg))
        X = np.array(l_plateImg, dtype=np.uint8)
        #print 'X type :',type(X)
        #print 'X.dtype :',X.dtype
        #print 'X.shape :',X.shape
        #print np.array(list(map(lambda a: [a for a in list(x)], l_plateStr)))#,dtype=np.float32)
        #ytmp = np.array(list(map(lambda a: [a for a in list(x)], l_plateStr)))#, dtype=np.uint8)# x: [M_strIdx[a]
        temp = list(map(lambda x: [a for a in list(x)], l_plateStr))#elesun TypeError: object of type 'map' has no len()
        #print("temp\n",temp)
        #print('temp type :', type(temp))    # <type 'list'>
        #print("temp[0]\n",temp[0])
        #print('temp[0] type :', type(temp[0])) # <type 'list'>
        #print("temp[0][0]\n",temp[0][0])
        #print('temp[0][0] type :', type(temp[0][0])) # <type 'str'>
        #print("temp[0][0] + temp[0][1] + temp[0][2] :", (temp[0][0] + temp[0][1] + temp[0][2]))
        temp2 = [] #list的第一层
        for i in range(len(temp)):
            temp1 = [] #list的第二层
            for j in range(len(temp[i])):
                if j == 0 :
                    temp1.append(temp[i][0] + temp[i][1] + temp[i][2]) #拼接字符串形成汉字 闽
                elif 1 <= j <= 2 :
                    continue # 只拼接前三个字符为汉字
                else :
                    temp1.append(temp[i][j]) #后面只追加 车牌数字和字符
            temp2.append(temp1)
        #print("temp2\n",temp2)
        #打印字典对应值是否正确
        #for i in range(len(temp2)):
        #    for j in range(len(temp2[i])):
        #        print("temp2[%d][%d]=" % (i, j),temp2[i][j],"; M_strIdx[(temp2[%d][%d])]="%(i,j),M_strIdx[(temp2[i][j])])
        #print('temp2 type :', type(temp2))  # <type 'numpy.ndarray'>
        #print("M_strIdx['A']",M_strIdx['A'])
        #print("M_strIdx['\xe6\xb9\x98']", M_strIdx['\xe6\xb9\x98'])
        #print("M_strIdx['\xe5']", M_strIdx['\xe5']) # error
        #ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], l_plateStr)), dtype=np.uint8)
        ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in x], temp)), dtype=np.uint8)#elesun temp2 for python2 ubuntu
        #print('ytmp\n', ytmp)
        #print ('ytmp type :',type(ytmp)) # <type 'numpy.ndarray'>
        #print ('ytmp.dtype :',ytmp.dtype) # uint8
        #print ('ytmp.shape :',ytmp.shape) # (32, 7)
        y = np.zeros([ytmp.shape[1],batch_size,len(chars)])# 7,32,65
        #print 'y type :',type(y)
        #print 'y.dtype :',y.dtype
        #print 'y.shape :',y.shape
        for batch in range(batch_size):
            for idx,row_i in enumerate(ytmp[batch]):
                y[idx,batch,row_i] = 1
        yield X, [yy for yy in y]

#########################定义网络并训练###########################################
def model_build_train(lr=0.001, epochs=25, batch_size=32, model_name="model_best.h5"):
    print("building network ...")
    #用一个 一组卷积层+7个全链接层 的架构，来对应输入的车牌图片
    input_tensor = Input((72, 272, 3))
    x = input_tensor
    for i in range(3):
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    n_class = len(chars) #elesun len(chars)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(7)]

    model = Model(inputs=input_tensor, outputs=x)
    model.summary()
    print("save network picture")
    #SVG(model_to_dot(model=model, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    print("training network ...")
    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    best_model = ModelCheckpoint(os.path.join(model_dir, model_name), monitor='val_loss', verbose=0, save_best_only=True)
    #print("gen(batch_size)",list(gen(batch_size)))
    #fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
    model.fit_generator(gen(batch_size), steps_per_epoch=200, epochs=epochs,
                   validation_data=gen(batch_size), validation_steps=20,
                   verbose=2,callbacks=[best_model]) #每个epoch输出一行记录
#########################读取测试车牌图片###########################################
def load_plate_data(data_dir="./recognize_samples"):
    print("loading plate data ...")
    plateStr = []
    plateImg = []
    file_list = os.listdir(data_dir)
    #print(file_list)
    for filename in file_list:
        path = ''
        path = os.path.join(data_dir, filename)
        image = cv2.imread(path)  #读取图片 cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
        #print("image.shape:",image.shape) #(72, 272, 3)
        if image.shape != (72, 272, 3) :
            # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
            print("picture %s size error, maybe resize before load !"%(filename))
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print ("%s has been read!"%filename)
        plateStr.append(filename[:-4])
        plateImg.append(image)
    return plateStr, plateImg
##########################展示模型预测结果########################################
def model_load_predict_plt(model_name,test_Img):
    # 加载模型
    print('load the trained model')
    model = load_model(os.path.join(model_dir, model_name))
    print("###############model predict###############")
    results = model.predict(np.array(test_Img))
    print('results type :', type(results)) #<type 'list'>
    results = np.array(results)
    print ('results type :',type(results)) #<type 'numpy.ndarray'>
    print ('results.dtype :',results.dtype) #float32
    print ('results.shape :',results.shape) #(7, num, 65)
    results = np.argmax(results, axis = 2)
    results = results.T
    print ('results.dtype :',results.dtype) #int64
    print ('results.shape :',results.shape) #(num, 7)
    print('results\n', results)  #
    #print("M_strIdx[0]",M_strIdx[0])
    #results = "".join([M_strIdx[xx] for xx in results.T])
    predict_plate_str = [] # list的第一层
    for i in range(results.shape[0]):
        temp = []  # list的第二层
        for j in range(results.shape[1]):
            for key, value in M_strIdx.items():
                if value == results[i,j]:
                    print("key",key)
                    temp.append(key)
        predict_plate_str.append(temp)
    print('predict_plate_str type :', type(predict_plate_str))  #
    print('predict_plate_str\n', predict_plate_str)
    # predict_plate_str = np.array(predict_plate_str)
    # print('predict_plate_str type :', type(predict_plate_str))
    # print ('predict_plate_str.dtype :',predict_plate_str.dtype) #
    # print ('predict_plate_str.shape :',results.shape) #
    # print('predict_plate_str\n', predict_plate_str)  #
    print("###############plt results###############")
    myfont = FontProperties(fname='./font/Lantinghei.ttc')
    # 用来正常显示中文标签，SimHei是字体名称，字体必须再系统中存在，字体的查看方式和安装第三部分
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
    fig = plt.figure(figsize=(12,12))
    #l_titles = list(map(lambda x: "".join([M_idxStr[xx] for xx in x]), np.argmax(np.array(model.predict( np.array(l_plateImg) )), 2).T))
    for idx,img in enumerate(test_Img[0:12]):
        ax = fig.add_subplot(4,3,idx+1)
        ax.imshow(img)
        ax.set_title(predict_plate_str[idx],fontproperties=myfont)
        ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    model_name = "model_best.h5"
    model_build_train(lr=0.001, epochs=30, batch_size=16, model_name="model_best.h5")
    test_data_dir = "./recognize_samples"
    test_name, test_Img = load_plate_data(test_data_dir)
    print("test_name",test_name)
    model_load_predict_plt(model_name, test_Img)





