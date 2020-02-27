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

logging.basicConfig(filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s %(module)s %(lineno)04d :%(levelname)10s: %(message)s",
                    level=logging.DEBUG)
# filename：指定日志文件名，+—name—+是将log命名为当前py的文件名，不设置则打印到控制台
# filemode：和file函数意义相同，指定日志文件的打开模式，'w'或者'a'；
# format：指定输出的格式和内容，format可以输出很多有用的信息，
#         %(levelno)s：打印日志级别的数值
#         %(levelname)s：打印日志级别的名称
#         %(pathname)s：打印当前执行程序的路径，其实就是sys.argv[0]
#         %(filename)s：打印当前执行程序名
#         %(funcName)s：打印日志的当前函数
#         %(lineno)d：打印日志的当前行号
#         %(asctime)s：打印日志的时间
#         %(thread)d：打印线程ID
#         %(threadName)s：打印线程名称
#         %(process)d：打印进程ID
#         %(message)s：打印日志信息
# datefmt：指定时间的输出格式
# level：设置日志级别，默认为logging.WARNNING；，程序会输出优先级大于等于此级别的信息。
#         优先级：日志等级：使用范围
#         FATAL：致命错误
#         50：CRITICAL：特别糟糕的事情，如内存耗尽、磁盘空间为空，一般很少使用
#         40：ERROR：发生错误时，如IO操作失败或者连接问题
#         30：WARNING：发生很重要的事件，但是并不是错误时，如用户登录密码错误
#         20：INFO：处理请求或者状态变化等日常事务
#         10：DEBUG：调试过程中使用DEBUG等级，如算法中每个循环的中间状态
#         0 ：NOTSET
# stream：指定将日志的输出流，可以指定输出到sys.stderr，sys.stdout或者文件，默认输出到sys.stderr，当stream和filename同时指定时，stream被忽略；
# style：如果 format 参数指定了，这个参数就可以指定格式化时的占位符风格，如 %、{、$ 等。
# handlers：决定将日志记录分配至正确的目的地。举个例子，一个应用可以将所有的日志消息发送至日志文件，所有的错误级别（error）及以上的日志消息发送至标准输出，所有的严重级别（critical）日志消息发送至某个电子邮箱。在这个例子中需要三个独立的处理器，每一个负责将特定级别的消息发送至特定的位置。
#         常用的有4种：
#         logging.StreamHandler -> 控制台输出
#         logging.FileHandler  -> 文件输出
#         logging.handlers.RotatingFileHandler -> 按照大小自动分割日志文件，一旦达到指定的大小重新生成文件
#         logging.handlers.TimedRotatingFileHandler  -> 按照时间自动分割日志文件
logger = logging.getLogger(__name__) # filename  __name__
#logger.setLevel(level=logging.WARN) # logging.DEBUG
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')
#格式化输出
logger.debug('format print : Hello {0}, {1}!'.format('World', 'Congratulations'))#不推荐
logger.debug('persent print : Hello %s, %s!'%('World', 'Congratulations'))#不推荐
logger.debug('advice print : Hello %s, %s!', 'World', 'Congratulations')#推荐
logger.debug("sleep cycle in %s",__name__)
for t in range(10):
    time.sleep(1)
    print("second %d"%(t))
    logger.debug("second %s",t)
    #logger.debug("second %02d", t)
