import numpy as np
import copy
import MPS
import Operator as op
from scipy.linalg import expm


d = 2
chi = 10
length = 10
m = MPS.mps.init_rand(d, chi, length)
m.center_orth(0, normalize=True)
m0 = m.mps2tensor()
m0 = m0/np.linalg.norm(m0)
gate = np.eye(d ** 2).reshape([d, d, d, d])
m.evolve_gate(gate, m.length-2)
m1 = m.mps2tensor()
m1 = m1/np.linalg.norm(m1)
err = np.linalg.norm(m1 - m0)
print(err)

"""

H = op.heisenberg_hamilt([1, 1, 1], [0, 0, 0])
print(H)
a = expm(np.zeros((2,2)))
print(a)
"""