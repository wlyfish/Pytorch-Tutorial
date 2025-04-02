#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/2 21:18
 @Author  : wly
 @File    : hello_pytorch.py
 @Description: 
"""
import torch

print("Hello World, Hello PyTorch {}".format(torch.__version__))

print("\nCUDA is available:{}, version is {}".format(torch.cuda.is_available(), torch.version.cuda))

print("\ndevice_name: {}".format(torch.cuda.get_device_name(0)))

