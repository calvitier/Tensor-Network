import numpy as np

def spin_operator_one_half():
    op = dict()
    op['i'] = np.eye(2)  # Identity
    op['x'] = np.zeros((2, 2))
    op['x'][0, 1] = 1 / 2
    op['x'][1, 0] = 1 / 2
    op['y'] = np.zeros((2, 2), dtype=np.complex)
    op['y'][0, 1] = 1j / 2
    op['y'][1, 0] = -1j / 2
    op['z'] = np.zeros((2, 2))
    op['z'][0, 0] = 1 / 2
    op['z'][1, 1] = -1 / 2
    return op


def heisenberg_hamilt(j, h):
    """
    :param j: list，耦合参数[Jx, Jy, Jz]
    :param h: list，外磁场[hx, hy, hz]
    :return H: 哈密顿量
    """
    op = spin_operator_one_half()
    H = j[0]*np.kron(op['x'], op['x']) + j[1]*np.kron(op['y'], op['y']) + \
        j[2]*np.kron(op['z'], op['z'])
    H += h[0] * (np.kron(op['x'], op['i']) + np.kron(op['i'], op['x']))
    H += h[1] * (np.kron(op['y'], op['i']) + np.kron(op['i'], op['y']))
    H += h[2] * (np.kron(op['z'], op['i']) + np.kron(op['i'], op['z']))
    if np.linalg.norm(np.imag(H)) < 1e-20:
        H = np.real(H)
    return H