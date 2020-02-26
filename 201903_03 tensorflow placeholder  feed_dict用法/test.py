#coding:UTF-8
import tensorflow as tf

#输入
input1 = tf.placeholder(tf.float32,(2,3))#placeholder(dtype,shape),定义一个3行2列的矩阵
input2 = tf.placeholder(tf.float32,(3,2))#定义一个2行3列的矩阵
a = [[1,2,3],[0,1,1]]
b = [[3,2],[1,2],[0,1]]
print ('****************input*******************')
print ('a = \n',a)
print ('b = \n',b)
#输出
output = tf.matmul(input1,input2)#matmul(),矩阵乘法

#执行
with tf.Session() as sess:
    result = sess.run(output,feed_dict = {input1:a,input2:b})
	#{input1:[[1,2,3],[0,1,1]],input2:[[3,2],[1,2],[0,1]]})
    print ('****************output*******************')
    print ('c = a*b\n',result)

	
