import numpy as np
import copy
import torch as tc
import matplotlib.pyplot as plt
import cv2
import MPS

d = 2
chi = 10
length = 10
m = MPS.mps.init_rand(d, chi, length)
m.center_orth(0, normalize=True)
m0 = m.mps2tensor()
m0 = m0/np.linalg.norm(m0)
gate = np.eye(d ** 2).reshape([d, d, d, d])
m.evolve_gate(gate, 5)
m1 = m.mps2tensor()
m1 = m1/np.linalg.norm(m1)
err = np.linalg.norm(m1 - m0)
print(err)

