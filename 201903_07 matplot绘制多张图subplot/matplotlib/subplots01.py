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

import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for i,color in enumerate("rgby"):
        plt.subplot(221+i, facecolor=color)
    plt.show()

