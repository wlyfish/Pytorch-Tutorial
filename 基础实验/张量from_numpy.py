#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/2 21:39
 @Author  : wly
 @File    : 张量from_numpy.py
 @Description: 
"""
import torch
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
t_from_numpy = torch.from_numpy(arr)
print("numpy array: ", arr)
print("tensor : ", t_from_numpy)
print("\n修改arr")
arr[0, 0] = 0
print("numpy array: ", arr)
print("tensor : ", t_from_numpy)
print("\n修改tensor")
t_from_numpy[0, 0] = -1
print("numpy array: ", arr)
print("tensor : ", t_from_numpy)
