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
import sub1 #以函数的方式实现
import sub2 #以类的方式实现

logging.basicConfig(filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s %(module)s %(lineno)04d :%(levelname)10s: %(message)s",
                    level=logging.DEBUG)
# handlers：决定将日志记录分配至正确的目的地。举个例子，一个应用可以将所有的日志消息发送至日志文件，所有的错误级别（error）及以上的日志消息发送至标准输出，所有的严重级别（critical）日志消息发送至某个电子邮箱。在这个例子中需要三个独立的处理器，每一个负责将特定级别的消息发送至特定的位置。
#         常用的有4种：
#         logging.StreamHandler -> 控制台输出
#         logging.FileHandler  -> 文件输出
#         logging.handlers.RotatingFileHandler -> 按照大小自动分割日志文件，一旦达到指定的大小重新生成文件
#         logging.handlers.TimedRotatingFileHandler  -> 按照时间自动分割日志文件
logger = logging.getLogger(__name__) # filename  __name__
#################handlers屏幕上输出#########################
screen = logging.StreamHandler()  # 往屏幕上输出
fmt = "%(asctime)s %(name)s %(module)s :%(levelname)10s: %(message)s"
format_str = logging.Formatter(fmt)#设置日志格式
screen.setFormatter(format_str)  # 设置屏幕上显示的格式
screen.setLevel(logging.INFO)
logger.addHandler(screen) #把对象加到logger里

#logger.setLevel(level=logging.WARN) # logging.DEBUG
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')
logger.debug('format print : Hello {0}, {1}!'.format('World', 'Congratulations'))#不推荐
logger.debug('persent print : Hello %s, %s!'%('World', 'Congratulations'))#不推荐
logger.debug('advice print : Hello %s, %s!', 'World', 'Congratulations')#推荐

logger.debug("sleep cycle in %s",__name__)
for t in range(10):
    time.sleep(1)
    print("print fun second %d in"%(t),__name__)
    logger.debug("second %s",t)
    #logger.debug("second %02d", t)

sub1.somefun()
a = sub2.SubModuleClass()
a.method()
