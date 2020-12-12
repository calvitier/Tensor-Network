import numpy as np
import copy
import torch as tc
import matplotlib.pyplot as plt
import cv2
import MPS

x = MPS.mps(d = 4, chi = 10, length = 4)
print(x.virtdim)
for n in range(0,x.length):
    print(x.tensors[n].shape,end=' ')
x.orth_right2left(3, cut_dim = 6)
print('\n',x.virtdim)
for n in range(0,x.length):
    print(x.tensors[n].shape,end=' ')
x.orth_right2left(2, cut_dim = 6)
print('\n',x.virtdim)
for n in range(0,x.length):
    print(x.tensors[n].shape,end=' ')
x.orth_right2left(1, cut_dim = 6)
print('\n',x.virtdim)
for n in range(0,x.length):
    print(x.tensors[n].shape,end=' ')