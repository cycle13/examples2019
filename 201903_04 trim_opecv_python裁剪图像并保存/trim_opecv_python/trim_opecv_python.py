# coding = utf-8
import os
import cv2
import time

dataset_dir = "./data"
dataset_out_dir = "./out"


def trim_opecv_python(in_dir,out_dir):
    if not os.path.exists(out_dir):
           os.makedirs(out_dir)
    file_list = os.listdir(in_dir)
    print(file_list)
    for filename in file_list:
        path = ''
        path = in_dir+'/'+filename
		####################读入图像###############################
        image = cv2.imread(path,cv2.IMREAD_COLOR)#以rgb的方式读取图片 cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
        #print ('image.shape:',image.shape) # (531, 870, 3) 高*宽*通道
        #print ('image.shape:',image.shape[0]) # 531 高
		####################图像裁剪###############################
        #cropped = image[0:128, 0:512]
        #cropped = image[int(image.shape[0]/2.0)-100:int(image.shape[0]/2.0)+100, int(image.shape[1]/2.0)-100:int(image.shape[1]/2.0)+100]
        cropped = image[316:402, 229:370]		
		#裁剪坐标为[y0:y1, x0:x1]，x0y0为左上角坐标，x1y1为右下角坐标
		#裁剪出的图像大小为：高（y1-y0）*宽（x1-x0）
        ####################写入图像########################		
        path = out_dir+'/'+filename
        cv2.imwrite(path,cropped)        
        print ("%s has been trimed!"%filename)

if __name__ == '__main__':
   time1 = time.time()
   file_list = os.listdir(dataset_dir)
   print(file_list)
   for dirname in file_list:
        in_path = ''
        in_path = dataset_dir+'/'+dirname
        print (in_path)
        trim_opecv_python(in_path,dataset_out_dir+'/'+dirname+'_trimed')
   time2=time.time()
   print (u'总共耗时：' + str(time2 - time1) + 's')
