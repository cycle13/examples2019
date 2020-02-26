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
from pynvml import *
nvmlInit()
print ("GPU Driver Version:", nvmlSystemGetDriverVersion())#显卡驱动版本
deviceCount = nvmlDeviceGetCount()#几块显卡
for i in range(deviceCount):
	handle = nvmlDeviceGetHandleByIndex(i)
	print ("Device", i, ":", nvmlDeviceGetName(handle)) #具体是什么显卡
	print("meminfo.used", nvmlDeviceGetMemoryInfo(handle).used)  # 这里是字节bytes，
nvmlShutdown()
