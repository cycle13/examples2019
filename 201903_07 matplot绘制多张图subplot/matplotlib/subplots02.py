# -*- coding: utf-8 -*-
""" 
subplots_matplot_test01
fun:
	draw multi-pictures 
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3
"""
#from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
 
 
def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)
 
if __name__ == '__main__' :
    t1 = np.arange(0, 5, 0.1)
    t2 = np.arange(0, 5, 0.02)
 
    plt.figure(10)
    plt.subplot(2,2,1)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')
 
    plt.subplot(2,2,2)
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
 
    plt.subplot(2,1,2)
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
 
    plt.show()

