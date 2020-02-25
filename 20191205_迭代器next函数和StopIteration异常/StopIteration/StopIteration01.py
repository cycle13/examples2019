# -*- coding: utf-8 -*-
'''

'''
from __future__ import print_function
__author__ = 'elesun'

li=[1,2,3,4]
it=iter(li)

print(next(it))
print(next(it))
print(next(it))
print(next(it))
print(next(it))# next()完成后引发StopIteration异常
