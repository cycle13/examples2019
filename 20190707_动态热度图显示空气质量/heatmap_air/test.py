# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
import folium
import folium.plugins as plugins
import webbrowser


# 读取整个csv文件
# df = pd.read_csv("air_data.csv", header=0, index_col=None) # dtype={"station_id":np.int,"PM25_AQI_value":np.int}
df = pd.read_csv("air_data.csv", header=0, index_col=None)
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
    wq_data.append(df.loc[(df['station_id'] == i), ['PM25_AQI_value']].values[0:2400])
#print('wq_data\n', wq_data)
wq_data = np.array(wq_data)
print('wq_data.shape:', wq_data.shape)
#print('wq_data:\n', wq_data)


location = np.array([
    [22.548958, 114.016356], [22.560689, 114.026827], [22.559262, 114.029402], [22.556726, 114.029230],
    [22.549116, 114.067339], [22.548799, 114.080042], [22.511220, 114.054121], [22.567347, 114.086222],
    [22.565603, 114.105105], [22.574480, 114.122614], [22.574163, 114.143385], [22.569725, 114.144759],
    [22.557518, 114.143042], [22.548641, 114.139265]
])  # [lat,lon]

c1_data = []
for i in range(wq_data.shape[1]):# 2400
    for j in range(wq_data.shape[0]):# 14
        temp = np.insert(location, 2, wq_data[:, i].reshape(1, 14), axis=1)
        # temp = np.insert(location, 2, wq_data[:,i,0], axis=1)
        # temp = np.insert(location, 2, wq_data[:, i])
        # print ("wq_data[j,i,0]",j,i,wq_data[:,i,0])
        # print("temp\n",temp)
    c1_data.append(temp)
c1_data = np.array(c1_data)
#print("c1_data\n",c1_data)
print("c1_data.shape",c1_data.shape)
c1_data = c1_data.tolist()

c2_data = []
weight = np.random.rand(14, 10)  # [weight,frame]
# print ("weight.shape",weight.shape)
# print("weight\n", weight)
for i in range(weight.shape[1]):
    temp = np.insert(location, 2, weight[:, i], axis=1)
    # print ("temp\n",temp)
    c2_data.append(temp)
# print("c2_data\n",c2_data)
c2_data = np.array(c2_data)  # list to array
# print("c2_data\n", c2_data)
print("c2_data.shape", c2_data.shape) #(10, 14, 3) (frame,location,weight)
c2_data = c2_data.tolist()

m = folium.Map([22.549116, 114.067339], zoom_start=12) # zoom_start small or big  tiles='stamentoner'
hm = plugins.HeatMapWithTime(c1_data,radius=22)
hm.add_to(m)

file_path = "AirQualityMap.html" #r"D:\AirQualityMap.html"
# 保存为html文件
m.save(file_path)
# 默认浏览器打开
webbrowser.open(file_path)

