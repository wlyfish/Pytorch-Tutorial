#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/2 22:07
 @Author  : wly
 @File    : 梯度清零.py
 @Description: 
"""
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

for i in range(4):
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward()
    print(w.grad)  # 梯度不会自动清零，数据会累加， 通常需要采用 optimizer.zero_grad() 完成对参数的梯度清零

    # w.grad.zero_()
