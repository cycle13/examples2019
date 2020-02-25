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

import os
import numpy as np
import shutil
import time
from PIL import Image
import matplotlib.pyplot as plt
print(plt.style.available)
plt.style.use('grayscale')
import re

in_dir = 'input'
out_dir = "out_mnist"
seq_length = 20
input_length = 10

def visual_compare(in_dir,out_dir):
	dataset_file_list = os.listdir(in_dir)
	print('dataset_file_list is ',dataset_file_list)
	print('num of dataset_file =',len(dataset_file_list))
	for index_dataset,dataset_filename in enumerate(dataset_file_list):
		#将不同模型下预测图片放到同一个exampleid_filename，构成对比图片文件夹，并保存到out路径下
		model_path = ''
		model_path = in_dir+'/'+dataset_filename
		model_file_list = os.listdir(model_path)
		model_file_list.sort(key=lambda x: int(x[0:2]))  # '01FC-LSTM2000', '02ConvLSTM6000', '03TrajGRU10000'
		#print('model_file_list is ',model_file_list)
		#print('num of model_file =',len(model_file_list))
		for index_model,model_filename in enumerate(model_file_list):
			exampleid_path = ''
			exampleid_path = model_path+'/'+model_filename
			exampleid_file_list = os.listdir(exampleid_path)
			exampleid_file_list.sort(key=lambda x:int(x[:])) #1 2 3 ... 10
			#print('exampleid_file_list is ',exampleid_file_list)
			#print('num of exampleid_file =',len(exampleid_file_list))
			for index_example,exampleid_filename in enumerate(exampleid_file_list):
				imgfile_path = ''
				imgfile_path = exampleid_path+'/'+exampleid_filename
				#print ("%s has been find!"%imgfile_path)
				outpath = os.path.join(out_dir,dataset_filename,exampleid_filename)
				if not os.path.exists(outpath):
					os.makedirs(outpath)
				img_list = os.listdir(imgfile_path)
				#img_list.sort(key=lambda x:int(x[2:4])) #gt02.png pd16.png
				img_list.sort()  #
				#print('img_list is ',img_list)
				#print('num of imgfile =',len(img_list))
				for index_img,imgname in enumerate(img_list):
					img_path = ''
					img_path = imgfile_path+'/'+imgname
					img = Image.open(img_path)
					#print ("%s has been find!"%imgname)
					#print ("%s has been find!"%img_path)
					if ((imgname[0:2] == 'gt') and (index_model != 0)):#
						#print("%s not save!" % img_path)
						pass
					else :
						print("%s has been saved!" % img_path)
						img.save(outpath+"/"+model_filename+"_"+imgname)

		#matplot绘图，每个exampleid_filename文件夹下不同模型的预测图片，上下形成对比图
		for index_example, exampleid_filename in enumerate(exampleid_file_list):
			plt_outpath = os.path.join(out_dir, "plt_contrast", dataset_filename)
			if not os.path.exists(plt_outpath):
				os.makedirs(plt_outpath)
			imgfile_path = os.path.join(out_dir,dataset_filename,exampleid_filename)
			img_list = os.listdir(imgfile_path)
			# img_list.sort(key=lambda x:int(x[2:4]))
			img_list.sort()  #
			#print("img_list\n",img_list)
			fig = plt.figure("%s Image" % exampleid_filename,figsize = (seq_length - input_length, len(img_list) / (seq_length - input_length)))  # 图像窗口名称#设置窗口大小
			figsize = seq_length - input_length, len(img_list) / (seq_length - input_length)
			print("figsize = ",figsize)
			# plt.suptitle('|<------------Input seq------>|\
			# |<------Ground truth and predictions----->|')  # 图片名称
			for index_img, imgname in enumerate(img_list):
				img_path = ''
				img_path = os.path.join(imgfile_path,imgname)
				img = Image.open(img_path)
				#print ("%s has been find!"%imgname)
				#print ("%s has been find!"%img_path)
				ax = plt.subplot(len(img_list) / (seq_length - input_length), seq_length - input_length, index_img + 1)
				if index_img == 0 :#第1行第1列
					str = "Inputs"
					ax.text(-110, 35, str, fontsize=12, ha='left', va='bottom')
					#y=0字体在图片靠上 y=50字体在图片中间 y=100字体在图片以下
					#ax.set_title('t=%d' % (index_img + 1))
					#ax.set_axis_on()
					#ax.set_ylabel(imgname[2:-9],rotation=0)
				elif index_img == input_length :#第2行第1列
					str = "Ground Truth"
					ax.text(-110, 35, str, fontsize=12, ha='left', va='bottom')
				elif (index_img >= seq_length) and (index_img%(seq_length-input_length) == 0) : #其余行第1列
					str = "".join((re.findall("(?<=\d\d)\D*",imgname))[0])#正则化字符串 01FC-LSTM2000_gt01.png
					print("str = ",str)
					ax.text(-110, 35, str, fontsize=12, ha='left',va='bottom')
					#ha='left'设定的坐标值为字体的左边
					#va='center'设定的坐标值为字体的中间
				plt.imshow(img)
				plt.axis('off')#此处控制着axis+label是否显示
				#plt.xlabel("abc")#rotation=0
			plt.subplots_adjust(wspace=0, hspace=0.1)#控制左右图像间隙
				#参数解释：
				#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)
				# wspace,hspace：用于控制宽度和高度的百分比，比如subplot之间的间距
			plt.savefig(plt_outpath+'/'+'contrast_'+exampleid_filename,bbox_inches="tight",dpi=300,pad_inches = 0)
				#参数解释：
				#bbox_inches="tight"时没有边框空白
				#dpi=100时分辨率为824x370，dpi=300时分辨率为2473x1108
				#transparent=False时背景为白色，transparent=True时背景为马赛克
				#pad_inches = 0时四周边框空白大小
				# format='png'为默认值，supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
			#plt.show()
			plt.close()#避免产生数量过多warning
			print("%s haved been ploted !"%imgfile_path)

if __name__ == '__main__':
	time1 = time.time()
	if not os.path.exists(in_dir):
		print("there is not directory",in_dir)
	print ("image directory is ",in_dir)
	if os.path.exists(out_dir):
		#os.remove(out_dir) #删除整个目录
		shutil.rmtree(out_dir) 
	os.makedirs(out_dir)
	visual_compare(in_dir,out_dir)
	time2=time.time()
	print ('time use: ' + str(time2 - time1) + ' s')
