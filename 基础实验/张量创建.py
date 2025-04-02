#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/2 21:37
 @Author  : wly
 @File    : 张量创建.py
 @Description: 
"""
import torch
import numpy as np

l = [[1., -1.], [1., -1.]]
t_from_list = torch.tensor(l)
arr = np.array([[1, 2, 3], [4, 5, 6]])
t_from_array = torch.tensor(arr)
print(t_from_list, t_from_list.dtype)
print(t_from_array, t_from_array.dtype)

arr = np.array([[1, 2, 3], [4, 5, 6]])
t_from_array = torch.tensor(arr, dtype=torch.uint8)
print(t_from_array)
