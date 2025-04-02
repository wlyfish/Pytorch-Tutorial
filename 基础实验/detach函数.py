#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/2 22:29
 @Author  : wly
 @File    : detach函数.py
 @Description: 
"""
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()

w_detach = w.detach()
w_detach.data[0] = 999
print(w)
