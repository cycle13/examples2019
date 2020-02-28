# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'elesun'

#data 未定义
print ('data' in locals().keys())
#输出：False

if not ("data" in locals().keys()):
    print("var not defined")

#data 定义
data = 1
print ('data' in locals().keys())
#输出：True
