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
import shutil
import time
from PIL import Image
import matplotlib.pyplot as plt

img_dir = './results/radar_predrnn_pp_radar500_train001'
out_dir = "./out"
seq_length = 20
input_length = 10

def Visual_GroundTruth_Prediction(img_dir,out_dir):
	iter_file_list = os.listdir(img_dir)	
	#print('iter_file_list is ',iter_file_list)
	#print('num of iter_file =',len(iter_file_list))
	for iter_filename in iter_file_list:
		batchid_path = ''
		batchid_path = img_dir+'/'+iter_filename
		batchid_file_list = os.listdir(batchid_path)	
		#print('batchid_file_list is ',batchid_file_list)
		#print('num of batchid_file =',len(batchid_file_list))
		for batchid_filename in batchid_file_list:
			imgfile_path = ''
			imgfile_path = batchid_path+'/'+batchid_filename
			#print ("%s has been find!"%imgfile_path)
			img_list = os.listdir(imgfile_path)	
			img_list.sort()#img_list.sort(key=lambda x:int(x[4:])) #gt02.png pd16.png
			#print('img_list is ',img_list)
			#print('num of file =',len(img_list))
			plt.figure("%s Image"%iter_filename,figsize=(12,2)) #图像窗口名称#设置窗口大小 
			plt.suptitle('|<------------------------Input seq-------------------------------------->|\
|<-------------Ground truth and predictions-------------->|') #图片名称
			for i,imgname in enumerate(img_list):
				img_path = ''	
				img_path = imgfile_path+'/'+imgname
				img = Image.open(img_path)
				#print ("%s has been find!"%imgname)
				#print ("%s has been find!"%img_path)
				if (imgname[0:2] == 'gt'):
					plt.subplot(2,seq_length,i+1)
					plt.title('t=%d'%(i+1)) #plt.title(imgname)
				elif (imgname[0:2] == 'pd'):
					plt.subplot(2,seq_length,i+1+input_length)
					#plt.title(imgname)#plt.title('t=%d'%(i+1))
				plt.imshow(img), plt.axis('off')
			ax = plt.subplot(2,seq_length,seq_length+input_length-1)
			plt.axis('off')
			ax.text(0.5, 0.5, 'predrnn++',fontsize=12, ha='center')
			plt.subplots_adjust(wspace=0.0, hspace=0.0)
			# wspace,hspace：用于控制宽度和高度的百分比，比如subplot之间的间距
			plt.savefig(out_dir+'/'+'contrast_'+iter_filename+'_'+batchid_filename+'.png')
			#plt.show()
	

if __name__ == '__main__':
	time1 = time.time()
	if not os.path.exists(img_dir):
		print("there is not directory",img_dir)
	print ("image directory is ",img_dir)
	if os.path.exists(out_dir):
		#os.remove(out_dir) #删除整个目录
		shutil.rmtree(out_dir) 
	os.makedirs(out_dir)
	Visual_GroundTruth_Prediction(img_dir,out_dir)
	time2=time.time()
	print ('time use: ' + str(time2 - time1) + ' s')
