import numpy as np
import copy
import torch as tc
import matplotlib.pyplot as plt
import cv2
import MPS

x = MPS.mps(d = 5, chi = 10, length = 2)
# tensor0 = copy.deepcopy(x.tensors)
x.center_orth(center = 0, cut_dim = 6)
tensor = x.get_tensor(1)
np.tensordot(tensor,tensor,[[1],[1]])