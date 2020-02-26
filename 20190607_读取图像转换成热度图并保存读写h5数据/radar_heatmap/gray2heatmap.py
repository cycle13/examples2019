# -*- coding: utf-8 -*-
""" 
plt_heatmap_radar_img
fun:
	transform radar img to heatmap
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3,seaborn0.9.0
"""
from __future__ import print_function
import os
import numpy as np
import shutil
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

img_dir = './results/radar_predrnn_pp_radar500_train001'
out_dir = "./out"
fwname = "imgs.h5"
h = w = 200

def trans_img_heatmap(img_dir,out_dir):
	fw = h5py.File(out_dir+'/'+fwname, 'w')
	area_file_list = os.listdir(img_dir)
	#print('area_file_list is ',area_file_list)
	#print('num of area_file =',len(area_file_list))
	for area_filename in area_file_list:
		temp = np.zeros([1,h, w], dtype=float)
		imgfile_path = ''
		imgfile_path = img_dir+'/'+area_filename
		#print ("%s has been find!"%imgfile_path)
		img_list = os.listdir(imgfile_path)
		img_list.sort()#img_list.sort(key=lambda x:int(x[4:])) #gt02.png pd16.png
		#print('img_list is ',img_list)
		#print('num of file =',len(img_list))
		for i,imgname in enumerate(img_list):
			img_path = ''
			img_path = imgfile_path+'/'+imgname
			img = Image.open(img_path)
			print ("%s has been find!"%imgname)
			img_array = np.array(img)
			#print ("%s has been find!"%img_path)
			#print("img_array.shape",img_array.shape)
			#print("img_array.dtype", img_array.dtype)
			img_array = (255-img_array)/255.0#归一化处理，（0-1）
			#temp = np.concatenate((temp,img_array),axis=0)
			#temp = np.row_stack((temp,img_array))
			if i == 0 :
				temp = img_array[np.newaxis,]
			else :
				temp = np.concatenate((temp, img_array[np.newaxis,]), axis=0)
			print("temp.shape",temp.shape)
			#print("img_array", img_array)
			# if (imgname[0:2] == 'GT'):
			# 	print("GT")
			# elif (imgname[0:2] == 'PD'):
			# 	print("PD")
			plt.subplots()
			sns.heatmap(img_array, vmin=0, vmax=1)#np.max(img_array)
			# save png
			plt.savefig(out_dir +'/'+area_filename+'_'+imgname)
			print("heatmap has been saved!")
			# plt.show()
		fw[area_filename] = temp
	fw.close()

if __name__ == '__main__':
	time1 = time.time()
	if not os.path.exists(img_dir):
		print("there is not directory",img_dir)
	print ("image directory is ",img_dir)
	if os.path.exists(out_dir):
		#os.remove(out_dir) #删除整个目录
		shutil.rmtree(out_dir) 
	os.makedirs(out_dir)
	trans_img_heatmap(img_dir,out_dir)
	time2=time.time()
	print ('time use: ' + str(time2 - time1) + ' s')
