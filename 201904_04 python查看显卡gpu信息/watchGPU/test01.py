# -*- coding: utf-8 -*-
""" 
get_gpuinf_test
fun:
	get_gpuinf
env:
	Linux ubuntu supermapBJAIGPU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,nvidia-ml-py375.53.1
"""
from __future__ import print_function
import pynvml
pynvml.nvmlInit()
# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(1)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print('meminfo.total',meminfo.total) #第二块显卡总的显存大小
print('meminfo.used',meminfo.used)#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
print('meminfo.free',meminfo.free) #第二块显卡剩余显存大小

print('meminfo.total',round(meminfo.total/1024.0/1024.0,2),'Mbytes') #第二块显卡总的显存大小
print('meminfo.used',round(meminfo.used/1024.0/1024.0,2),'Mbytes')#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
print('meminfo.free',round(meminfo.free/1024.0/1024.0,2),'Mbytes') #第二块显卡剩余显存大小
