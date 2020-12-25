import numpy as np
import PEPS

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
    