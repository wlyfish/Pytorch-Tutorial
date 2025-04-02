#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/2 22:02
 @Author  : wly
 @File    : 手动实现2D卷积.py
 @Description: 
"""
import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F


def convolution_backward(grad_out, X, weight):
    """
    将反向传播功能用函数包装起来，返回的参数个数与forward接收的参数个数保持一致，为2个
    """
    grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight)
    return grad_X, grad_input


class MyConv2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight):
        ctx.save_for_backward(X, weight)
        # ============== step1: 函数功能实现 ==============
        ret = F.conv2d(X, weight)
        # ============== step1: 函数功能实现 ==============
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        X, weight = ctx.saved_tensors
        return convolution_backward(grad_out, X, weight)


weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(10, 3, 7, 7, requires_grad=True, dtype=torch.double)
# gradcheck 会检查你实现的自定义操作的前向传播 (forward) 和反向传播 (backward) 方法是否正确计算了梯度。
# 如果返回 True，则表示梯度检查通过，即自定义操作的梯度计算与数值近似梯度之间的一致性在允许的误差范围内；
# 如果返回 False，则说明存在不匹配，需要检查和修正自定义操作的反向传播逻辑。
print("梯度检查: ", torch.autograd.gradcheck(MyConv2D.apply, (X, weight)))  # gradcheck 功能请自行了解，通常写完Function会用它检查一下
y = MyConv2D.apply(X, weight)
label = torch.randn_like(y)
loss = F.mse_loss(y, label)

print("反向传播前，weight.grad: ", weight.grad)
loss.backward()
print("反向传播后，weight.grad: ", weight.grad, weight.grad.shape)
