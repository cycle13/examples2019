# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'elesun'

class UserError(Exception)  :  # Exception：所有的异常
    # UserError 继承异常Exception
    # 简单理解就是现在UserError和其他异常一样，是个自定义异常
    def __init__(self, msg):
        self.message = msg

try:
    raise UserError('数据库连不上')
    # ('数据库连不上')作为 UserError 异常的形参 msg = 数据库连不上
    # raise触发BurgessError异常
except UserError as e :  # 抓取UserError异常里自定义的信息
    print(e)  # 得到自定义的异常信息
# 自定义异常的名字最好不要和本身系统的异常名字一样，会导致抓取效果不一致，且又不能完全覆盖
