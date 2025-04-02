#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/2 22:19
 @Author  : wly
 @File    : 叶子结点自加.py
 @Description:
 RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
"""
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

w.add_(1)

y.backward()
