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
m0 = copy.deepcopy(m.tensors[-1])
gate = 2 * np.eye(d ** 2).reshape([d, d, d, d])
m.evolve_gate(gate, m.length - 2)
m1 = copy.deepcopy(m.tensors[-1])
diff = m1 - m0
print(np.linalg.norm(diff))
err = m1 - 2*m0
print(np.linalg.norm(err))

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


x = PEPS.peps.init_rand(2, 5, (3, 3))
x.to_Gamma_Lambda()
"""
for i in range(0, 3):
    for j in range(0, 3):
        print(x.tensors[1][1].shape)
"""
(i, j) = (0, 1)
I = np.eye(4).reshape(2, 2, 2, 2)
norm = np.linalg.norm(x.tensors[i][j])
x.evolve_gate(I, (i, j), (i, j+1))
norm1 = np.linalg.norm(x.tensors[i][j])
diff = norm - norm1
# print(tensor)
print(norm)
print(norm1)


"""
x = tc.rand((2, 3, 4, 5, 6))
y = tc.rand((2, 3, 4, 5, 6))
ans = tc.einsum('abcde, abcij -> deij', x, y)
ans = ans.numpy()
print(ans.shape)
"""