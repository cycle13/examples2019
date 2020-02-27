#coding=utf-8
"""
logging模块
Author: elesun
logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等；
相比print，具备如下优点：
    可以通过设置不同的日志等级，在release版本中只输出重要信息，而不必显示大量的调试信息；
    print将所有信息都输出到标准输出中，严重影响开发者从标准输出中查看其它数据；logging则可以由开发者决定将信息输出到什么地方，以及怎么输出
"""
from __future__ import print_function
import os
import logging
import time

class SubModuleClass(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__) # "mainModule.sub.module"
        self.logger.info("creating an instance in Class")
    def method(self):
        self.logger.info("do something in Class method")
        n = []
        n.append(1)
        self.logger.debug("list a = " + str(n))
        self.logger.info("finish something in Class method")