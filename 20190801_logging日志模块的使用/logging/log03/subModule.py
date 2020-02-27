#coding=utf-8
"""
logging模块
Author: elesun
logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等；
相比print，具备如下优点：
    可以通过设置不同的日志等级，在release版本中只输出重要信息，而不必显示大量的调试信息；
    print将所有信息都输出到标准输出中，严重影响开发者从标准输出中查看其它数据；logging则可以由开发者决定将信息输出到什么地方，以及怎么输出
"""
import logging

class SubModuleClass(object):#在类中定义方法，方法里面打印log
    def __init__(self):
        self.logger = logging.getLogger("mainModule.sub.module")# mainModule必须要与主模块中一致 sub.module不可改
        self.logger.info("creating an instance in SubModuleClass")
    def doSomething(self):
        self.logger.info("do something in SubModule.Class.Method")
        a = []
        a.append(1)
        self.logger.debug("list a = " + str(a))
        self.logger.info("finish something in SubModule.Class.Method")

module_logger = logging.getLogger("mainModule.sub") ## mainModule必须要与主模块中一致 sub不可改
def som_function():
    module_logger.info("call function from subModule.Fun")#在函数中打印log