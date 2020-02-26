# -*- coding: utf-8 -*-
""" 
img_ssim_test
fun:
	get ssim metric 
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3;opencv-python 3.4.3.18;scikit-image 0.14.0
"""
from __future__ import print_function
#import the necessary packages
from skimage.measure import compare_ssim
import argparse
import cv2
  
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
help="first input image")
ap.add_argument("-s", "--second", required=True,
help="second")
args = vars(ap.parse_args())
 
# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
  
# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
 
# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
