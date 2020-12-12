import numpy as np
import copy
import torch as tc
import matplotlib.pyplot as plt
import cv2
import MPS

x = MPS.mps.init_rand(d = 4, chi = 10, length = 4)
print(x.virtdim)
for n in range(0,x.length):
    print(x.tensors[n].shape,end=' ')
