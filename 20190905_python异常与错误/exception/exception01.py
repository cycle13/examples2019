# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'elesun'

try:
    fh = open("testfile", "r")
    #fh.write("这是一个测试文件，用于测试异常!!")
except IOError:
    print ("Error: 没有找到文件或读取文件失败")
else:
    print ("内容写入文件成功")
    fh.close()

raise Exception('主动抛出异常') #主动抛出异常

