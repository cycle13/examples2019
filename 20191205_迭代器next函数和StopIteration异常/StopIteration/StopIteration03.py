# -*- coding: utf-8 -*-
'''

'''
from __future__ import print_function
__author__ = 'elesun'

li=[1,2,3,4]
it=iter(li)

import sys  #while循环需要带异常处理
while True:
    try:
        print(next(it))
    except StopIteration:
        print("StopIteration")
        sys.exit()

