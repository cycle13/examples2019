# -*- coding: utf-8 -*-
'''
https://zhidao.baidu.com/question/70430105.html?qq-pf-to=pcqq.c2c
'''
from __future__ import print_function
__author__ = 'elesun'

id = input("请输入身份证前17位:")
if len(id)!= 17 or id.isdigit()==False :
    print("invalid input")
else:
    c = [7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
    temp = []
    for i in range(17):
        temp.append(int(id[i])*c[i])
    s = sum(temp)
    ck = ["1","0","X","9","8","7","6","5","4","3","2"]
    print("身份证18位为：" + ck[s % 11])

'''
import tensorflow as tf
a = tf.constant(2.1) #定义tensor常量
with tf.Session() as sess:
    print(sess.run(a))

sess=tf.Session()
b=a.eval(session=sess)
print (b)
print (type(b))
#转化为tensor
c=tf.convert_to_tensor(b)
print (c)
print (type(c))
'''
'''
#data 未定义
print ('data' in locals().keys())
#输出：False

if not ("data" in locals().keys()):
    print("var not defined")

#data 定义
data = 1
print ('data' in locals().keys())
#输出：True
'''

'''
import time

print("----------------------日期时间格式转换-------------------------")
# 字符类型的时间 2017-05-19 01:09:12
st_time0 = "2017-05-19 01:09:12"
print("st_time0",st_time0)
# 转为时间数组
timeArray = time.strptime(st_time0, "%Y-%m-%d %H:%M:%S")
print ("timeArray",timeArray)
#timeArray可以调用tm_year等
print ("timeArray.tm_year",timeArray.tm_year)  # 2017
print ("timeArray.tm_mon",timeArray.tm_mon)  # 5
print ("timeArray.tm_mday",timeArray.tm_mday)  # 19
print ("timeArray.tm_hour",timeArray.tm_hour)  # 1
print ("timeArray.tm_min",timeArray.tm_min)  # 12
print ("timeArray.tm_sec",timeArray.tm_sec)  # 4

# 转为时间戳
timeStamp = int(time.mktime(timeArray))
print ("timeStamp",timeStamp)  # 1495127352
'''

'''
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
'''

'''
import re

imgname = "02Conv-LSTM2000_gt01.png"
list1 = re.findall("\d",imgname)
list2 = re.findall("(?<=\d\d)\D+",imgname) #*(?=\d)
print("list1 = ",list1)
print("list2 = ",list2)
str1 = "".join(list1[2:-2])
str2 = "".join(list2[0])

print("str1 = ",str1)
print("str2 = ",str2)
'''