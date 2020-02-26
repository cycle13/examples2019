# -*- coding: utf-8 -*-
""" 
subplots_matplot_test
fun:
	draw multi-pictures 
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3
"""
from __future__ import print_function
import cv2
import os
import time
import shutil
import numpy as np
from matplotlib import pyplot as plt

dataset_dir = './data'
out_dir = './out'

def gray2RGB(dataset_dir,out_dir):
	area_list = os.listdir(dataset_dir)
	print('area_list is ',area_list)
	print('num of area_list =',len(area_list))
	for areas in area_list:
		img_file_path = ''
		img_file_path = dataset_dir+'/'+areas
		img_list = os.listdir(img_file_path)	
		#print('img_list is ',img_list)
		print('num of img_list =',len(img_list))
		for imgname in img_list:
			img_path = ''
			img_path = img_file_path+'/'+imgname	
			gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)#cv2默认为bgr顺序
			print('%s has been read !'%imgname)
			#print ('gray_img.shape :',gray_img.shape)
			#print ('gray_img :\n',gray_img)
			#plt.hist(gray_img.ravel(), 256, [0,256])
			#plt.show()
			row = gray_img.shape[0]
			col = gray_img.shape[1]
			rgb_img_mat = np.zeros((row, col, 3),dtype=np.uint8)
			#print ('rgb_img_mat:',rgb_img_mat)
			for i in range(row) :
				for j in range(col) :
					gray_val = gray_img[i][j]#红255,0,0 橙255,165,0 黄255,255,0 绿0,255,0 青0,255,255 蓝0,0,255
					#print ('gray_val',gray_val)#0-255 数值越小越红越会下雨 数值越大越蓝越不会下雨 255-白色
					if   0  <= gray_val < 5 :
						rgb_img_mat[i,j,2],rgb_img_mat[i,j,1],rgb_img_mat[i,j,0] = 0,0,255 #蓝色
					elif 5 <= gray_val < 8:
						rgb_img_mat[i,j,2],rgb_img_mat[i,j,1],rgb_img_mat[i,j,0] = 0,255,255 #青色
					elif 8 <= gray_val < 10:
						rgb_img_mat[i,j,2],rgb_img_mat[i,j,1],rgb_img_mat[i,j,0] = 0,255,0 #绿色
					elif 10 <= gray_val < 13:
						rgb_img_mat[i,j,2],rgb_img_mat[i,j,1],rgb_img_mat[i,j,0] = 255,255,0 #黄色	
					elif 13<= gray_val < 17:
						rgb_img_mat[i,j,2],rgb_img_mat[i,j,1],rgb_img_mat[i,j,0] = 255,165,0 #橙色
					elif 17<= gray_val < 22:
						rgb_img_mat[i,j,2],rgb_img_mat[i,j,1],rgb_img_mat[i,j,0] = 255,0,0 #红色
					elif 22<= gray_val < 200:
						rgb_img_mat[i,j,2],rgb_img_mat[i,j,1],rgb_img_mat[i,j,0] = 160,32,240 #紫色
					elif 200<= gray_val < 256:
						rgb_img_mat[i,j,2],rgb_img_mat[i,j,1],rgb_img_mat[i,j,0] = 255,255,255 #白色
					else :
						print('invalid value') #
					#print (i,j,rgb_img_mat[i,j])
			out_img_file = 	out_dir+'/'+areas	
			if not os.path.exists(out_img_file):
				os.makedirs(out_img_file)			
			cv2.imwrite(out_img_file+'/'+imgname,rgb_img_mat)
			print('%s has been saved !'%imgname)
			
			
if __name__ == '__main__':
	time1 = time.time()
	if not os.path.exists(dataset_dir):
		print("there is not directory",dataset_dir)
	print ("image directory is ",dataset_dir)
	if os.path.exists(out_dir):
		shutil.rmtree(out_dir) #删除整个目录
	os.makedirs(out_dir)
	gray2RGB(dataset_dir,out_dir)
	time2=time.time()
	print ('time use: ' + str(time2 - time1) + ' s')
