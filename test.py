import numpy as np
import torch as tc
import PEPS
import MPS
import copy

"""
d = 2
chi = 10
length = 10
m = MPS.mps.init_rand(d, chi, length)
m.center_orth(m.length - 2, normalize=True)
m2 = m.inner(m)
gate = np.eye(d ** 2).reshape([d, d, d, d])
m0 = copy.deepcopy(m)
m.evolve_gate(gate, m.length - 2, cut_dim=d)
mim = m0.inner(m)
i = mim/m2
print(i)
"""

"""
H = op.heisenberg_hamilt([1, 1, 1], [0, 0, 0])
gs = MPS.ground_state(3, H, times=1e5)
print('success!')
"""

"""
physdim = 2
virtdim = 5
n = 4
m = 4
shape = (n, m)
p = PEPS.peps.init_rand(physdim, virtdim, shape)
print(p.physdim)
print(p.virtdim_horizontal)
print(p.virtdim_vertical)
for i in range(0, n):
    for j in range(0, m):
        print(p.tensors[i][j].shape, end=' ')
    print('\n')
"""

"""
x = np.random.rand(5, 5, 5)
x = x.reshape(5, 25)
u, lm, v = PEPS.svd_cut(x)
print(lm)
x = v.reshape(-1, 5, 5)
x = np.rollaxis(x, 1, 0)
x = x.reshape(5, -1)
u, lm, v = PEPS.svd_cut(x)
print(lm)
"""

"""
x = PEPS.peps.init_rand(2, 5, (3, 3))
x.to_Gamma_Lambda()
(i, j) = (0, 1)
I = np.eye(4).reshape(2, 2, 2, 2)
tensor = copy.deepcopy(x.tensors[i][j])
x.evolve_gate(I, (i, j), (i, j+1), cut_dim=5, debug=True)
diff = tensor - x.tensors[i][j]
norm = np.linalg.norm(diff)
# print(tensor)
print(norm)
"""

pd = 2
vd = 5
shape = (3, 3)
(i, j) = (0, 0)
m = PEPS.peps.init_rand(pd, vd, shape)
gate = np.eye(pd ** 2).reshape([pd, pd, pd, pd])

m.to_Gamma_Lambda()
m0 = copy.deepcopy(m)
m1 = copy.deepcopy(m)
m2 = m0.inner(m0)
m.evolve_gate(gate, (i, j), (i, j+1), cut_dim=vd)
mim = m1.inner(m)
I = mim/m2
print(I)

