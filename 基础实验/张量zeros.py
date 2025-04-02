#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/2 21:44
 @Author  : wly
 @File    : 张量zeros.py
 @Description:
 通过torch.zeros创建的张量不仅赋给了t，同时赋给了o_t，
 并且这两个张量是共享同一块内存，只是变量名不同。
"""
import torch

o_t = torch.tensor([1])
t = torch.zeros((3, 3), out=o_t)
print(t, '\n', o_t)
print(id(t), id(o_t))

t1 = torch.tensor([[1., -1.], [1., -1.]])
t2 = torch.zeros_like(t1)
print(t2)

print(torch.full((2, 3), 3.141592))

