#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/2 22:00
 @Author  : wly
 @File    : 自动微分.py
 @Description: 
"""
import torch
from torch.autograd.function import Function


class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        # ============== step1: 函数功能实现 ==============
        result = i.exp()
        # ============== step1: 函数功能实现 ==============

        # ============== step2: 结果保存，用于反向传播 ==============
        ctx.save_for_backward(result)
        # ============== step2: 结果保存，用于反向传播 ==============

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # ============== step1: 取出结果，用于反向传播 ==============
        result, = ctx.saved_tensors
        # ============== step1: 取出结果，用于反向传播 ==============

        # ============== step2: 反向传播公式实现 ==============
        grad_results = grad_output * result
        # ============== step2: 反向传播公式实现 ==============

        return grad_results


x = torch.tensor([1.], requires_grad=True)
y = Exp.apply(x)  # 需要使用apply方法调用自定义autograd function
print(y)  # y = e^x = e^1 = 2.7183
y.backward()
print(x.grad)  # 反传梯度,  x.grad = dy/dx = e^x = e^1  = 2.7183

# 关于本例子更详细解释，推荐阅读 https://zhuanlan.zhihu.com/p/321449610
