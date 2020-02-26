#coding = utf-8
'''
kth dataset convert avi to pngs
$ nohup python avi2png.py >out_64x64.log &
'''
from __future__ import print_function
import os
import time
dataset_dir = "/home/sun/sun/kth/kthdata"
dataset_out_dir = "./out"
image_size = 128#64 128
frame_rate = 25
classes = {'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'}


def avi2png(dataset_dir,out_dir):
    if not os.path.exists(out_dir):
           os.makedirs(out_dir)
    for _, classname in enumerate(classes):
           print(' ------------------------ ',classname,' ------------------------ ')
           in_path = dataset_dir+'/raw/'+classname
           file_list = os.listdir(in_path)
           print('num of file =',len(file_list))
           print(file_list)
           for filename in file_list:
               path = ''
               path = in_path+'/'+filename
               #print ("%s has been find!"%path)
               print ("%s has been find!"%filename)
               filename1 = filename[0:-11] #person13_boxing_d1_uncomp.avi --> person13_boxing_d1
               os.system('mkdir -p %s/%s/%s'%(out_dir, classname, filename1))
               #/home/sun/sun/kth/kthdata/raw/boxing/person01_boxing_d1_uncomp.avi
               os.system('ffmpeg -i %s/raw/%s/%s -r %d -f image2 -s %dx%d  %s/%s/%s/image-%%03d_%dx%d.png'%(dataset_dir, classname, filename, frame_rate, image_size, image_size, out_dir, classname, filename1, image_size, image_size))
 


if __name__ == '__main__':
   time1 = time.time()
   if not os.path.exists(dataset_dir):
          print("there is not directory",dataset_dir)
   print ("dataset directory is ",dataset_dir)
   avi2png(dataset_dir,dataset_out_dir+'_'+str(image_size)+'x'+str(image_size))
   time2=time.time()
   print ('time use: ' + str(time2 - time1) + ' s')

