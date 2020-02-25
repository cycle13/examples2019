# -*- coding: utf-8 -*-
'''
sefattn
ref:
    https://mp.weixin.qq.com/s/uscTj1hr_Aogaz98VS0JTQ
    https://zhuanlan.zhihu.com/p/96492170
'''
from __future__ import print_function
__author__ = 'elesun'

import torch
########prepare_inputs############
x = [
    [1,0,1,0],# input 1
    [0,2,0,2],# input 2
    [1,1,1,1],# input 3
]
x = torch.tensor(x,dtype=torch.float32)
print("tensor input\n",x)
print("input shape",x.size())
########initial_weights############
w_key = [
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
]
w_query = [
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1],
]
w_value = [
    [0, 2, 0],
    [0, 3, 0],
    [1, 0, 3],
    [1, 1, 0],
]
w_key = torch.tensor(w_key,dtype=torch.float32)
w_query = torch.tensor(w_query,dtype=torch.float32)
w_value = torch.tensor(w_value,dtype=torch.float32)
print("tensor w_key\n",w_key)
print("w_key shape",w_key.size())
########derive kqv############
# @ 是用来对tensor进行矩阵相乘的
# * 用来对tensor进行矩阵进行逐元素相乘
keys = x @ w_key
querys = x @ w_query
values =  x @ w_value
print("tensor keys\n",keys)
print("keys shape",keys.size())
print("tensor querys\n",querys)
print("querys shape",querys.size())
print("tensor values\n",values)
print("values shape",values.size())
########caculate_attn_scores############
# 矩阵转置：torch.t(input, out=None) → Tensor
# 输入一个矩阵（2维张量），并转置0, 1维。 可以被视为函数 transpose(input, 0, 1) 的简写函数。
attn_scores = querys @ keys.t() # .T
print("tensor keys.t\n",keys.t())
print("tensor attn_scores\n",attn_scores)
print("attn_scores shape",attn_scores.size())
########caculate_softmax############
from torch.nn.functional import softmax
attn_scores_softmax = softmax(attn_scores, dim=1)
print("tensor attn_scores_softmax\n",attn_scores_softmax)
print("attn_scores_softmax shape",attn_scores_softmax.size())
attn_scores_softmax = [
    [0.0, 0.5, 0.5],
    [0.0, 1.0, 0.0],
    [0.0, 0.9, 0.1]
]
attn_scores_softmax = torch.tensor(attn_scores_softmax,dtype=torch.float32)
print("tensor attn_scores_softmax for simple\n",attn_scores_softmax)
print("attn_scores_softmax shape for simple",attn_scores_softmax.size())
########attn_multiply_scores_values############
# None为对应轴增加一个维度
weighted_values = values[:,None] * attn_scores_softmax.t()[:,:,None] # .T
print("tensor values[:,None] \n",values[:,None] )
print("values[:,None]  shape",values[:,None] .size())
print("tensor attn_scores_softmax.t()[:,:,None] \n",attn_scores_softmax.t()[:,:,None] )
print("attn_scores_softmax.t()[:,:,None]  shape",attn_scores_softmax.t()[:,:,None].size())
print("tensor weighted_values\n",weighted_values)
print("weighted_values shape",weighted_values.size())
########sum_weighted_values############
outputs = weighted_values.sum(dim=0)
print("tensor outputs\n",outputs)
print("outputs shape",outputs.size())
