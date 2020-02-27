# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'elesun'

try:
    x = int(input('Enter the first number: '))
    y = int(input('Enter the second number: '))
    print(x / y)
except (ZeroDivisionError, TypeError, ValueError) as e:
    print(e)
    raise e

